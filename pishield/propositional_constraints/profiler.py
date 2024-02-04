import torch
import functools
import datetime


# A stack with abstract operations:
# - push & pop as usual
# - update x: set all values v to max(v, x)
class MaxStack:
    def __init__(self):
        self.stack = []

    def __len__(self):
        return len(self.stack)

    def push(self, value):
        self.stack.append(value)

    def update(self, value):
        if len(self.stack):
            last = self.stack.pop()
            self.stack.append(max(last, value))

    def pop(self):
        value = self.stack.pop()
        if len(self.stack):
            last = self.stack.pop()
            self.stack.append(max(last, value))
        return value


def singleton(cls):
    @functools.wraps(cls)
    def constructor(*args, **kwargs):
        if not hasattr(cls, '_instance_'):
            cls._instance_ = cls(*args, **kwargs)
        return cls._instance_

    constructor.__dict__.update(cls.__dict__)
    return constructor


def no_cuda():
    return not torch.cuda.is_available()


def get_allocated(device=None):
    if no_cuda(): return 0
    return torch.cuda.memory_allocated(device)


def get_peak(device=None):
    if no_cuda(): return 0
    return torch.cuda.max_memory_allocated(device)


def reset_peak(device=None):
    if no_cuda(): return None
    torch.cuda.reset_peak_memory_stats(device)


# Manages the global cuda profiling data 
@singleton
class PeakMemoryManager:
    def __init__(self):
        self.stack = MaxStack()

    def enter(self):
        self.stack.update(get_peak())
        reset_peak()
        self.stack.push(0)

    def exit(self):
        self.stack.update(get_peak())
        return self.stack.pop()


class Stats:
    def __init__(self, peak, diff, sum, tdiff, tsum):
        self.peak = peak
        self.diff = diff
        self.sum = sum

        self.tdiff = tdiff
        self.tsum = tsum

    @classmethod
    def single(cls, peak, diff, tdiff):
        return cls(peak, diff, diff, tdiff, tdiff)

    @classmethod
    def null(cls):
        return cls.single(0, 0, 0)

    def __add__(self, other):
        return Stats(
            max(self.peak, other.peak),
            max(self.diff, other.diff),
            self.sum + other.sum,
            max(self.tdiff, other.tdiff),
            self.tsum + other.tsum)

    def __str__(self):
        return str(self.tuple)

    def memory(self, long=True):
        if long:
            return (self.peak, self.diff, self.sum)
        else:
            return (self.peak, self.diff)

    def time(self, long=True):
        if long:
            return (self.tdiff, self.tsum)
        else:
            return (self.tdiff,)

    def tuple(self, long=True):
        return (self.time(long), self.memory(long))


# Profiler that records data from the manager
class Profiler:
    _enabled = True

    def __init__(self, watches=None):
        self.watches = dict() if watches is None else watches
        self.manager = PeakMemoryManager()

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def enabled(cls):
        return cls._enabled

    @classmethod
    def shared(cls):
        if not hasattr(cls, '_shared_'):
            cls._shared_ = cls()
        return cls._shared_

    @staticmethod
    def norm(x):
        return x / 1024 / 1024

    def branch(self, name):
        if not name in self.watches:
            self.watches[name] = dict()
        return Profiler(self.watches[name])

    def register(self, name, peak, diff, tdiff):
        [peak, diff] = [Profiler.norm(x) for x in [peak, diff]]
        stats = Stats.single(peak, diff, tdiff)

        if not name in self.watches:
            self.watches[name] = [stats]
        else:
            self.watches[name].append(stats)

    def watch(self, name):
        return Watch(name, self)

    def wrap(self, f):
        @functools.wraps(f)
        def profiled(*args, **kwargs):
            with self.watch(f.__name__):
                return f(*args, **kwargs)

        return profiled

    @classmethod
    def map_dict(cls, f, node):
        result = dict()
        for key in node:
            if isinstance(node[key], dict):
                result[key] = cls.map_dict(f, node[key])
            else:
                result[key] = f(node[key])
        return result

    def reset(self):
        def zero(x):
            x.clear()
            x.append(Stats.null())

        Profiler.map_dict(zero, self.watches)

    def get_kind(self, kind, long):
        if kind == 'all':
            return lambda x: x.tuple(long)
        elif kind == 'gpu':
            return lambda x: x.memory(long)
        elif kind == 'time':
            return lambda x: x.time(long)
        else:
            raise Exception(f"Unknown kind {kind}")

    def all(self, kind='all'):
        kinder = self.get_kind(kind, long=False)
        return Profiler.map_dict(lambda xs: [kinder(x) for x in xs], self.watches)

    def combined(self, kind='all'):
        kinder = self.get_kind(kind, long=True)
        return Profiler.map_dict(lambda x: kinder(sum(x, Stats.null())), self.watches)

    def total(self, kind='all'):
        kinder = self.get_kind(kind, long=True)
        result = Stats.null()

        def update(xs):
            nonlocal result
            for x in xs: result = result + x

        Profiler.map_dict(update, self.watches)
        return kinder(result)


def condition(cond):
    def conditioned(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            should = cond() if callable(cond) else cond
            return f(*args, **kwargs) if should else None

        return decorated

    return conditioned


class Watch:
    def __init__(self, name, profiler):
        self.name = name
        self.profiler = profiler

    @condition(Profiler.enabled)
    def __enter__(self):
        self.start = get_allocated()
        self.tstart = datetime.datetime.now()
        self.profiler.manager.enter()

    @condition(Profiler.enabled)
    def __exit__(self, a, b, c):
        self.peak = self.profiler.manager.exit()
        self.end = get_allocated()
        self.tend = datetime.datetime.now()

        peak = self.peak - self.start
        diff = self.end - self.start
        tdiff = (self.tend - self.tstart).total_seconds()

        self.profiler.register(self.name, peak, diff, tdiff)
