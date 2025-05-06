# HPC-ColPali

## Overview

Hierarchical Patch Compression for ColPali: Indexing and Retrieval with HPC-Colpali(Base on ColQwwen2.5)

## Code Structure

```bash
src/
├─ data/
├─ indexes/
│   ├─ faiss_hnsw.idx
│   ├─ faiss_pq.idx       # FAISS PQ index
│   └─ metadata.db        # SQLite metadata
├─ ingest.py              # script indexing offline
├─ query.py               # script/Flask API
├─ benchmark.py           # benchmark
├─ retriever_adapter.py   # wrapper adapter
├─ requirements.txt       # dependencies
└─ README.md
```

## Set up Env

- Using python 3.11 and conda for manage.

## Indexing

```
python ingest.py
```

## Retrieval

```
uvicorn query:app --host 0.0.0.0 --port 8000 --reload
```

## BenchMark (BEIR)

```
python benchmark.py
```

## Notes:

- You can use other base model.
- Benchmark Dataset must be download from [BEIR]
