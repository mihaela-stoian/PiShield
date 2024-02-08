import time

import pytest
import torch

from pishield.propositional_requirements.profiler import PeakMemoryManager, Profiler


def test_one_manager():
    manger = PeakMemoryManager()
    manager2 = PeakMemoryManager()
    assert id(manger) == id(manager2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_one_profiler():
    device = 'cuda'
    profiler = Profiler()

    for i in range(3):
        with profiler.watch('out + A + B + C'):
            out = torch.rand(512, 512, device=device)
            with profiler.watch('A'):
                a = torch.rand(1024, 1024, device=device)
            out2 = torch.rand(512, 512, device=device)
            with profiler.watch('B + C'):
                with profiler.watch('B'):
                    b = torch.rand(2048, 2048, device=device)
                with profiler.watch('C'):
                    c = torch.rand(1024, 2048, device=device)

    results = profiler.all(kind='gpu')
    assert results['A'] == [(4.0, 4.0), (4.0, 0.0), (4.0, 0.0)]
    assert results['B'] == [(16.0, 16.0), (16.0, 0.0), (16.0, 0.0)]
    assert results['C'] == [(8.0, 8.0), (8.0, 0.0), (8.0, 0.0)]
    assert results['B + C'] == [(24.0, 24.0), (16.0, 0.0), (16.0, 0.0)]
    assert results['out + A + B + C'] == [(30.0, 30.0), (16.0, 0.0), (16.0, 0.0)]

    results = profiler.combined(kind='gpu')
    assert results['A'] == (4., 4., 4.)
    assert results['B'] == (16., 16., 16.)
    assert results['C'] == (8., 8., 8.)
    assert results['B + C'] == (24., 24., 24.)
    assert results['out + A + B + C'] == (30., 30., 30.)

    results = profiler.total(kind='gpu')
    assert results == (30., 30., 82.)

    profiler.reset()
    assert profiler.total(kind='gpu') == (0, 0, 0)


def _test_nested(profiler, profiler2):
    device = 'cuda'

    with profiler.watch('test'):
        a = torch.rand(1024, 1024, device=device)
        with profiler2.watch('test2'):
            b = torch.rand(512, 512, device=device)
            del b
        with profiler2.watch('test3'):
            b = torch.rand(512 // 2, 512 // 2, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_different_profilers():
    profiler = Profiler()
    profiler2 = Profiler()

    _test_nested(profiler, profiler2)
    assert profiler.total(kind='gpu') == (5., 4.25, 4.25)
    assert profiler2.total(kind='gpu') == (1., 0.25, 0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared():
    profiler = Profiler.shared()
    profiler2 = Profiler.shared()

    _test_nested(profiler, profiler2)

    assert 'test' in profiler.all()
    assert 'test2' in profiler.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wrap():
    device = 'cuda'
    profiler = Profiler()

    @profiler.wrap
    def f(n, a):
        if n <= 1: return a
        return f(n - 1, a) + f(n - 2, a)

    a = torch.rand(1024, 1024, device=device)
    f(4, a)

    assert profiler.all(kind='gpu')['f'] == [(0.0, 0.0), (0.0, 0.0), (4.0, 4.0), (0.0, 0.0), (8.0, 4.0), (0.0, 0.0),
                                             (0.0, 0.0), (4.0, 4.0), (12.0, 4.0)]


def test_time():
    profiler = Profiler()

    def sleeper(s, profiler):
        with profiler.watch('test'):
            time.sleep(s)

    for s in range(1, 10):
        ss = 0.01 * s
        ss = min(ss, 0.1 - ss)
        sleeper(ss, profiler)

    print(profiler.all('time'))

    total = profiler.total(kind='time')
    assert 0.05 <= total[0] and total[0] <= 0.06
    assert 0.25 <= total[1] and total[1] <= 0.27


def test_disable():
    profiler = Profiler()

    Profiler.enable()
    with profiler.watch('test'):
        time.sleep(0.05)
    assert profiler.total('time')[1] > 0.05

    Profiler.disable()
    with profiler.watch('test'):
        time.sleep(0.05)
    assert profiler.total('time')[1] < 0.10

    Profiler.enable()
    with profiler.watch('test'):
        time.sleep(0.05)
    assert profiler.total('time')[1] > 0.10
