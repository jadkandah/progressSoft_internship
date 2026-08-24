# 12 - FastAPI Serving and Concurrency Experiments

## Objective

Package the tuned IMDB sentiment classifier behind a validated FastAPI endpoint, create a separate inference client, and measure completion time and throughput for 1,000 requests at increasing parallelism levels.

## Concepts

- FastAPI request and response models
- Application lifespan and loading a model once at startup
- Input validation and health checks
- Asynchronous HTTP clients and bounded concurrency
- Throughput, mean latency, p95 latency, and saturation
- Reproducible loopback benchmarking with a fixed workload

## Tasks

- [x] Refit and serialize the selected IMDB linear SVM pipeline
- [x] Reuse the existing preprocessing, regex tokenizer, and custom TF-IDF vectorizer
- [x] Expose health and prediction endpoints through FastAPI
- [x] Validate successful predictions and reject blank input
- [x] Create a separate Python inference client
- [x] Send 1,000 requests at 5, 10, 20, 40, 60, 80, and 100-way concurrency
- [x] Record completion time, throughput, mean latency, and p95 latency
- [x] Plot total time and throughput against parallel requests
- [x] Identify the measured maximum-throughput setting

## Project Files

- [`notebooks/fastapi_serving.ipynb`](notebooks/fastapi_serving.ipynb): executed training, validation, benchmark, plots, and observations
- [`src/training.py`](src/training.py): reproducible model training and serialization
- [`src/api.py`](src/api.py): FastAPI application
- [`src/client.py`](src/client.py): single-request inference client
- [`src/benchmark.py`](src/benchmark.py): reusable concurrency benchmark
- [`reports/results/fastapi_benchmark.csv`](reports/results/fastapi_benchmark.csv): measured benchmark results

## Run

From this directory:

```text
python src/training.py
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1
python src/client.py "A thoughtful and beautifully acted film."
python src/benchmark.py
```

## Notes

The serving model keeps the previous task's validation-selected `C=1` linear SVM. It was refitted on 39,665 development reviews with a 30,000-feature vocabulary and evaluated on the untouched 9,917-review test set. It achieved test accuracy `0.8945` and F1 `0.8957`. The serialized pipeline is stored under `models/` and remains ignored by Git.

The NVIDIA GeForce RTX 3050 Ti was detected. The experiment uses CPU because scikit-learn's `LinearSVC` does not support CUDA inference.

The benchmark used one local Uvicorn worker, persistent loopback connections, one excluded warm-up request per level, and exactly 1,000 measured requests at each setting. All 7,000 measured requests succeeded.

| Parallel requests | Total time (s) | Throughput (requests/s) | Mean latency (ms) | p95 latency (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 4.439 | 225.28 | 20.71 | 39.21 |
| 10 | 6.242 | 160.20 | 60.43 | 144.66 |
| 20 | 6.775 | 147.60 | 130.70 | 314.88 |
| 40 | 6.806 | 146.92 | 262.31 | 758.75 |
| 60 | 6.920 | 144.51 | 388.96 | 1219.69 |
| 80 | 8.965 | 111.55 | 688.07 | 2151.82 |
| 100 | 7.459 | 134.07 | 708.96 | 2087.40 |

Maximum measured throughput was `225.28` requests per second at `5` parallel requests. Higher concurrency increased waiting and tail latency on this single-process CPU workload instead of adding useful capacity. These loopback results describe this machine and setup rather than production capacity.

## Resources

- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Uvicorn documentation](https://www.uvicorn.org/)
- [HTTPX asynchronous support](https://www.python-httpx.org/async/)
