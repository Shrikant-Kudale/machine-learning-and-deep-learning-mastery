"""Day 29 - Edge/microservice inference: dynamic micro-batching simulation, latency/throughput trade-off, TorchScript."""
import argparse
import heapq
import time
import numpy as np
import torch
import torch.nn as nn


def measure_service_model(model, d_in=64, max_b=64, reps=30):
    """Fit service_time(b) = a + c*b (ms) from real measurements -> used by the simulator."""
    bs = np.array([1, 4, 16, 64]); ts = []
    with torch.inference_mode():
        for b in bs:
            x = torch.randn(int(b), d_in)
            for _ in range(5):
                model(x)
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            ts.append((time.perf_counter() - t0) / reps * 1000)
    c, a = np.polyfit(bs, ts, 1)
    return max(a, 1e-3), max(c, 1e-5)


def simulate(arrivals, a, c, max_batch, max_wait, workers=1):
    """Discrete-event simulation of a dynamic batcher: dispatch when batch is full or the oldest request waited max_wait."""
    n, free = len(arrivals), [0.0] * workers
    heapq.heapify(free)
    lat, i = np.zeros(n), 0
    while i < n:
        w_free = heapq.heappop(free)
        start = max(w_free, arrivals[i])
        deadline = arrivals[i] + max_wait
        j = i + 1
        while j < n and j - i < max_batch and arrivals[j] <= max(deadline, start):
            j += 1
        launch = max(start, min(deadline, arrivals[j - 1])) if j - i < max_batch else max(start, arrivals[j - 1])
        done = launch + a + c * (j - i)
        lat[i:j] = done - arrivals[i:j]
        heapq.heappush(free, done)
        i = j
    return lat


def main(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(64, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)).eval()
    a, c = measure_service_model(model)
    print(f"measured service time: {a:.3f} ms fixed + {c * 1000:.2f} us per sample")

    ts = torch.jit.trace(model, torch.randn(1, 64)).eval()
    xt = torch.randn(8, 64)
    with torch.inference_mode():
        assert torch.allclose(ts(xt), model(xt), atol=1e-6)
    print("TorchScript trace matches eager.")

    rng = np.random.default_rng(seed)
    rate = 0.8 / (a + c) * 1000 * 2.5                                     # req/s: ~2.5x capacity of unbatched serving
    arrivals = np.cumsum(rng.exponential(1000 / rate, 20000))             # ms
    print(f"offered load = {rate:,.0f} req/s | single-request capacity = {1000 / (a + c):,.0f} req/s")
    print(f"{'policy':<28s} {'p50 ms':>9s} {'p99 ms':>9s}")
    res = {}
    for name, mb, mw in [("no batching (b=1)", 1, 0.0), ("batch<=8, wait 1ms", 8, 1.0), ("batch<=32, wait 2ms", 32, 2.0)]:
        lat = simulate(arrivals, a, c, mb, mw)
        res[name] = np.percentile(lat, 99)
        print(f"{name:<28s} {np.percentile(lat, 50):9.2f} {np.percentile(lat, 99):9.2f}")
    assert res["batch<=32, wait 2ms"] < res["no batching (b=1)"], "batching must rescue an overloaded server"
    low = np.cumsum(rng.exponential(50.0, 2000))                          # light load: batching only adds wait
    assert np.percentile(simulate(low, a, c, 32, 5.0), 50) >= np.percentile(simulate(low, a, c, 1, 0.0), 50)
    print("Under overload batching wins on tail latency; under light load it only adds waiting time.")
    print("Day 29 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
