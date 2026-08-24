import argparse
import asyncio
from itertools import cycle, islice
from pathlib import Path
import time

import httpx
import pandas as pd


TASK_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = TASK_ROOT / "reports" / "results" / "fastapi_benchmark.csv"

SAMPLE_REVIEWS = [
    "A thoughtful story with excellent performances and a satisfying ending.",
    "The plot was dull, predictable, and far too long.",
    "I loved the direction and would happily watch it again.",
    "The characters were flat and the dialogue felt unnatural.",
    "A warm, funny film that exceeded my expectations.",
    "Nothing in this movie worked for me.",
    "Beautiful cinematography supports a genuinely moving story.",
    "It started badly and somehow became even worse.",
    "Smart writing and strong acting made this memorable.",
    "A frustrating waste of a promising premise.",
]


async def _send_request(client, semaphore, text, url):
    async with semaphore:
        started = time.perf_counter()
        response = await client.post(url, json={"text": text})
        elapsed = time.perf_counter() - started
    response.raise_for_status()
    return elapsed


async def run_benchmark(
    url="http://127.0.0.1:8000/predict",
    concurrency_levels=(5, 10, 20, 40, 60, 80, 100),
    total_requests=1000,
):
    rows = []
    texts = list(islice(cycle(SAMPLE_REVIEWS), total_requests))

    for parallel_requests in concurrency_levels:
        limits = httpx.Limits(
            max_connections=parallel_requests,
            max_keepalive_connections=parallel_requests,
        )
        semaphore = asyncio.Semaphore(parallel_requests)
        async with httpx.AsyncClient(timeout=30, limits=limits) as client:
            await client.post(url, json={"text": SAMPLE_REVIEWS[0]})
            started = time.perf_counter()
            latencies = await asyncio.gather(
                *(
                    _send_request(client, semaphore, text, url)
                    for text in texts
                )
            )
            total_seconds = time.perf_counter() - started

        ordered_latencies = sorted(latencies)
        p95_index = int(0.95 * (len(ordered_latencies) - 1))
        rows.append(
            {
                "parallel_requests": parallel_requests,
                "total_requests": total_requests,
                "total_seconds": total_seconds,
                "throughput_requests_per_second": total_requests / total_seconds,
                "mean_latency_ms": 1000 * sum(latencies) / len(latencies),
                "p95_latency_ms": 1000 * ordered_latencies[p95_index],
                "successful_requests": len(latencies),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--total-requests", type=int, default=1000)
    arguments = parser.parse_args()
    results = asyncio.run(
        run_benchmark(url=arguments.url, total_requests=arguments.total_requests)
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(results.to_string(index=False))
