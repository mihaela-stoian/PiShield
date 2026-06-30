"""Lightweight time and GPU-memory profiling.

Provides a hierarchical profiler used to instrument the constraint modules: named
"watches" (context managers, or the :meth:`Profiler.wrap` decorator) record the elapsed
time and peak/net CUDA memory of code regions, aggregating the measurements into a tree
of :class:`Stats`. On CPU-only systems the memory measurements degrade to zero.
"""

import torch
import functools
import datetime


# A stack with abstract operations:
# - push & pop as usual
# - update x: set all values v to max(v, x)
class MaxStack:
    """A stack whose ``update`` raises stored values to a running maximum.

    Supports the usual push/pop plus an ``update(x)`` that bumps the top of stack up to
    ``x``; popping a value also propagates it into the new top, so nested peak measures
    accumulate correctly.

    Attributes:
        stack: The underlying list of values.
    """

    def __init__(self):
        """Initialise an empty stack."""
        self.stack = []

    def __len__(self):
        """Return the number of values on the stack."""
        return len(self.stack)

    def push(self, value):
        """Push a value onto the stack.

        Args:
            value: The value to push.
        """
        self.stack.append(value)

    def update(self, value):
        """Raise the top of the stack to at least ``value``.

        Args:
            value: The lower bound to apply to the current top (no-op if empty).
        """
        if len(self.stack):
            last = self.stack.pop()
            self.stack.append(max(last, value))

    def pop(self):
        """Pop the top value, propagating it into the new top as a maximum.

        Returns:
            The popped value.
        """
        value = self.stack.pop()
        if len(self.stack):
            last = self.stack.pop()
            self.stack.append(max(last, value))
        return value


def singleton(cls):
    """Class decorator that turns a class into a lazily-created singleton.

    Args:
        cls: The class to wrap.

    Returns:
        A constructor function that always returns the single shared instance.
    """
    @functools.wraps(cls)
    def constructor(*args, **kwargs):
        if not hasattr(cls, '_instance_'):
            cls._instance_ = cls(*args, **kwargs)
        return cls._instance_

    constructor.__dict__.update(cls.__dict__)
    return constructor


def no_cuda():
    """Return True if CUDA is unavailable."""
    return not torch.cuda.is_available()


def get_allocated(device=None):
    """Return the currently allocated CUDA memory in bytes (0 without CUDA).

    Args:
        device: Optional CUDA device.

    Returns:
        The allocated memory in bytes, or 0 if CUDA is unavailable.
    """
    if no_cuda(): return 0
    return torch.cuda.memory_allocated(device)


def get_peak(device=None):
    """Return the peak allocated CUDA memory in bytes (0 without CUDA).

    Args:
        device: Optional CUDA device.

    Returns:
        The peak allocated memory in bytes, or 0 if CUDA is unavailable.
    """
    if no_cuda(): return 0
    return torch.cuda.max_memory_allocated(device)


def reset_peak(device=None):
    """Reset the CUDA peak-memory statistics (no-op without CUDA).

    Args:
        device: Optional CUDA device.
    """
    if no_cuda(): return None
    torch.cuda.reset_peak_memory_stats(device)


# Manages the global cuda profiling data
@singleton
class PeakMemoryManager:
    """Singleton tracking nested peak CUDA-memory measurements.

    Uses a :class:`MaxStack` so that entering a region resets the CUDA peak counter and
    exiting returns the peak observed within that region, while still propagating it to
    enclosing regions.

    Attributes:
        stack: The :class:`MaxStack` of in-flight peak measurements.
    """

    def __init__(self):
        """Initialise with an empty measurement stack."""
        self.stack = MaxStack()

    def enter(self):
        """Begin a nested measurement region, resetting the CUDA peak counter."""
        self.stack.update(get_peak())
        reset_peak()
        self.stack.push(0)

    def exit(self):
        """End the current measurement region.

        Returns:
            The peak CUDA memory observed within the region.
        """
        self.stack.update(get_peak())
        return self.stack.pop()


class Stats:
    """Aggregated time and memory statistics for a profiled region.

    Attributes:
        peak: The maximum peak memory observed.
        diff: The maximum net memory change observed.
        sum: The summed net memory change across measurements.
        tdiff: The maximum elapsed time observed.
        tsum: The summed elapsed time across measurements.
    """

    def __init__(self, peak, diff, sum, tdiff, tsum):
        """Store the individual statistics.

        Args:
            peak: The peak memory.
            diff: The net memory change.
            sum: The summed net memory change.
            tdiff: The elapsed time.
            tsum: The summed elapsed time.
        """
        self.peak = peak
        self.diff = diff
        self.sum = sum

        self.tdiff = tdiff
        self.tsum = tsum

    @classmethod
    def single(cls, peak, diff, tdiff):
        """Build a Stats from a single measurement (sum equals the value).

        Args:
            peak: The peak memory.
            diff: The net memory change.
            tdiff: The elapsed time.

        Returns:
            A Stats whose sum fields equal the corresponding single values.
        """
        return cls(peak, diff, diff, tdiff, tdiff)

    @classmethod
    def null(cls):
        """Return a zero-valued Stats (the additive identity)."""
        return cls.single(0, 0, 0)

    def __add__(self, other):
        """Combine two Stats: max the peak/diff fields and sum the totals.

        Args:
            other: The Stats to combine with.

        Returns:
            The combined Stats.
        """
        return Stats(
            max(self.peak, other.peak),
            max(self.diff, other.diff),
            self.sum + other.sum,
            max(self.tdiff, other.tdiff),
            self.tsum + other.tsum)

    def __str__(self):
        """Return a string representation of the stats tuple."""
        return str(self.tuple)

    def memory(self, long=True):
        """Return the memory statistics as a tuple.

        Args:
            long: If True include the summed change; otherwise only peak and diff.

        Returns:
            ``(peak, diff, sum)`` when ``long`` else ``(peak, diff)``.
        """
        if long:
            return (self.peak, self.diff, self.sum)
        else:
            return (self.peak, self.diff)

    def time(self, long=True):
        """Return the timing statistics as a tuple.

        Args:
            long: If True include the summed time; otherwise only the max.

        Returns:
            ``(tdiff, tsum)`` when ``long`` else ``(tdiff,)``.
        """
        if long:
            return (self.tdiff, self.tsum)
        else:
            return (self.tdiff,)

    def tuple(self, long=True):
        """Return the combined time and memory statistics.

        Args:
            long: Whether to include the summed fields.

        Returns:
            A tuple ``(time_tuple, memory_tuple)``.
        """
        return (self.time(long), self.memory(long))


# Profiler that records data from the manager
class Profiler:
    """Hierarchical recorder of timing and memory statistics.

    A profiler owns a tree of named "watches"; each leaf accumulates a list of
    :class:`Stats`. Sub-profilers created via :meth:`branch` share the parent's
    subtree, allowing nested instrumentation. Profiling can be globally toggled with
    :meth:`enable`/:meth:`disable`.

    Attributes:
        watches: The nested dict of recorded watches (subtrees or lists of Stats).
        manager: The shared :class:`PeakMemoryManager`.
    """

    _enabled = True

    def __init__(self, watches=None):
        """Create a profiler over an optional existing watches subtree.

        Args:
            watches: An existing watches dict to attach to, or None for a fresh tree.
        """
        self.watches = dict() if watches is None else watches
        self.manager = PeakMemoryManager()

    @classmethod
    def enable(cls):
        """Globally enable profiling."""
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Globally disable profiling."""
        cls._enabled = False

    @classmethod
    def enabled(cls):
        """Return whether profiling is currently enabled."""
        return cls._enabled

    @classmethod
    def shared(cls):
        """Return the shared singleton profiler instance."""
        if not hasattr(cls, '_shared_'):
            cls._shared_ = cls()
        return cls._shared_

    @staticmethod
    def norm(x):
        """Convert a byte count to mebibytes.

        Args:
            x: A value in bytes.

        Returns:
            The value in MiB.
        """
        return x / 1024 / 1024

    def branch(self, name):
        """Return a sub-profiler for a named subtree, creating it if needed.

        Args:
            name: The name of the branch.

        Returns:
            A Profiler scoped to the named subtree.
        """
        if not name in self.watches:
            self.watches[name] = dict()
        return Profiler(self.watches[name])

    def register(self, name, peak, diff, tdiff):
        """Record one measurement under a named watch.

        Args:
            name: The watch name.
            peak: The peak memory in bytes.
            diff: The net memory change in bytes.
            tdiff: The elapsed time in seconds.
        """
        [peak, diff] = [Profiler.norm(x) for x in [peak, diff]]
        stats = Stats.single(peak, diff, tdiff)

        if not name in self.watches:
            self.watches[name] = [stats]
        else:
            self.watches[name].append(stats)

    def watch(self, name):
        """Return a context manager that times and measures a named region.

        Args:
            name: The watch name to record under.

        Returns:
            A :class:`Watch` context manager.
        """
        return Watch(name, self)

    def wrap(self, f):
        """Decorator that profiles a function under its own name.

        Args:
            f: The function to wrap.

        Returns:
            The wrapped function, profiled under ``f.__name__``.
        """
        @functools.wraps(f)
        def profiled(*args, **kwargs):
            with self.watch(f.__name__):
                return f(*args, **kwargs)

        return profiled

    @classmethod
    def map_dict(cls, f, node):
        """Recursively apply a function to every leaf of a nested dict.

        Args:
            f: The function applied to each non-dict leaf value.
            node: The nested dict to traverse.

        Returns:
            A new nested dict with ``f`` applied to every leaf.
        """
        result = dict()
        for key in node:
            if isinstance(node[key], dict):
                result[key] = cls.map_dict(f, node[key])
            else:
                result[key] = f(node[key])
        return result

    def reset(self):
        """Clear all recorded measurements, leaving a single null Stats per watch."""
        def zero(x):
            x.clear()
            x.append(Stats.null())

        Profiler.map_dict(zero, self.watches)

    def get_kind(self, kind, long):
        """Return a selector mapping a Stats to the requested view.

        Args:
            kind: One of ``'all'`` (time and memory), ``'gpu'`` (memory), or
                ``'time'``.
            long: Whether the returned tuples should include the summed fields.

        Returns:
            A callable mapping a :class:`Stats` to the selected tuple.

        Raises:
            Exception: If ``kind`` is not recognised.
        """
        if kind == 'all':
            return lambda x: x.tuple(long)
        elif kind == 'gpu':
            return lambda x: x.memory(long)
        elif kind == 'time':
            return lambda x: x.time(long)
        else:
            raise Exception(f"Unknown kind {kind}")

    def all(self, kind='all'):
        """Return every individual measurement per watch.

        Args:
            kind: The view to extract (see :meth:`get_kind`).

        Returns:
            A nested dict mapping each watch to the list of its per-measurement tuples.
        """
        kinder = self.get_kind(kind, long=False)
        return Profiler.map_dict(lambda xs: [kinder(x) for x in xs], self.watches)

    def combined(self, kind='all'):
        """Return the per-watch aggregate of all its measurements.

        Args:
            kind: The view to extract (see :meth:`get_kind`).

        Returns:
            A nested dict mapping each watch to its combined Stats tuple.
        """
        kinder = self.get_kind(kind, long=True)
        return Profiler.map_dict(lambda x: kinder(sum(x, Stats.null())), self.watches)

    def total(self, kind='all'):
        """Return the aggregate over every watch in the tree.

        Args:
            kind: The view to extract (see :meth:`get_kind`).

        Returns:
            The combined Stats tuple summed across all watches.
        """
        kinder = self.get_kind(kind, long=True)
        result = Stats.null()

        def update(xs):
            nonlocal result
            for x in xs: result = result + x

        Profiler.map_dict(update, self.watches)
        return kinder(result)


def condition(cond):
    """Build a decorator that only runs the wrapped function when a condition holds.

    Args:
        cond: A boolean, or a callable returning a boolean, evaluated on each call.

    Returns:
        A decorator that runs the function when the condition is truthy, else returns
        None.
    """
    def conditioned(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            should = cond() if callable(cond) else cond
            return f(*args, **kwargs) if should else None

        return decorated

    return conditioned


class Watch:
    """Context manager that records the time and memory of its body.

    On enter it snapshots the time and allocated memory; on exit it computes the
    elapsed time, net memory change and peak memory and registers them with the
    profiler. Both methods are no-ops when profiling is disabled.

    Attributes:
        name: The watch name to register under.
        profiler: The owning :class:`Profiler`.
    """

    def __init__(self, name, profiler):
        """Store the watch name and owning profiler.

        Args:
            name: The watch name.
            profiler: The owning Profiler.
        """
        self.name = name
        self.profiler = profiler

    @condition(Profiler.enabled)
    def __enter__(self):
        """Snapshot start time and memory and begin the peak measurement."""
        self.start = get_allocated()
        self.tstart = datetime.datetime.now()
        self.profiler.manager.enter()

    @condition(Profiler.enabled)
    def __exit__(self, a, b, c):
        """Compute and register the elapsed time and memory of the region."""
        self.peak = self.profiler.manager.exit()
        self.end = get_allocated()
        self.tend = datetime.datetime.now()

        peak = self.peak - self.start
        diff = self.end - self.start
        tdiff = (self.tend - self.tstart).total_seconds()

        self.profiler.register(self.name, peak, diff, tdiff)
