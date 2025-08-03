# HPC-ColPali: Hierarchical Patch Compression for Document Retrieval

[![Paper](https://img.shields.io/badge/Paper-KDIR%202025-blue)](https://kdir.scitevents.org/Guidelines.aspx)
[![arXiv](https://img.shields.io/badge/arXiv-2506.21601-b31b1b)](https://arxiv.org/abs/2506.21601)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎉 News: Paper Accepted at KDIR 2025!

We are excited to announce that our paper **"Hierarchical Patch Compression for ColPali: Indexing and Retrieval with HPC-ColPali"** has been accepted for presentation at the **17th International Conference on Knowledge Discovery and Information Retrieval (KDIR 2025)** in Lisbon, Portugal!

**Conference Details:**

- **Event:** 17th International Conference on Knowledge Discovery and Information Retrieval (KDIR 2025)
- **Location:** Lisbon, Portugal
- **Date:** November 2025
- **Proceedings:** SCITEPRESS - Science and Technology Publications
- **Indexing:** SCOPUS, Google Scholar, DBLP, Semantic Scholar, CrossRef, and others

## 📖 Abstract

This repository implements **HPC-ColPali**, a novel hierarchical patch compression approach for document retrieval based on the ColQwen2.5 multilingual model. Our method introduces hierarchical patch-level embeddings and attention mechanisms to improve retrieval accuracy while maintaining computational efficiency.

**Key Contributions:**

- 🎯 **Hierarchical Patch Compression**: Novel approach to compress document patches while preserving semantic information
- 🌍 **Multilingual Support**: Based on ColQwen2.5-3B multilingual model for cross-lingual retrieval
- ⚡ **Efficient Indexing**: Dual-index strategy with HNSW and Product Quantization (PQ)
- 📊 **Comprehensive Evaluation**: Benchmark results on BEIR datasets

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/HPC-ColPali.git
cd HPC-ColPali

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Indexing Documents**

```bash
# Place your PDF documents in src/data/
python src/ingest.py
```

2. **Start Retrieval API**

```bash
cd src
uvicorn query:app --host 0.0.0.0 --port 8000 --reload
```

3. **Run Benchmarks**

```bash
python src/benchmark.py
```

## 📁 Project Structure

```
HPC-ColPali/
├── docs/
│   └── HPC_Copali_v2.pdf          # Conference paper
├── src/
│   ├── data/                       # Input PDF documents
│   ├── indexes/                    # Generated indices
│   │   ├── faiss_hnsw.idx         # HNSW index
│   │   ├── faiss_pq.idx           # Product Quantization index
│   │   └── metadata.db            # SQLite metadata
│   ├── ingest.py                   # Document indexing pipeline
│   ├── query.py                    # FastAPI retrieval service
│   ├── benchmark.py                # BEIR benchmark evaluation
│   └── retriever_adapter.py       # Retrieval adapter
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🔧 Technical Details

### Architecture

Our approach consists of three main components:

1. **Document Processing**: PDF parsing and chunking with overlap
2. **Embedding Generation**: Hierarchical patch-level embeddings using ColQwen2.5
3. **Indexing Strategy**: Dual-index approach with HNSW and PQ for efficiency

### Model Configuration

- **Base Model**: `tsystems/colqwen2.5-3b-multilingual-v1.0`
- **Chunk Size**: 500 characters with 50 character overlap
- **Batch Size**: 8 (configurable)
- **Index Types**: HNSW (32 neighbors) + Product Quantization (8 subspaces, 8 bits)

### API Endpoints

The FastAPI service provides:

- `POST /search`: Document retrieval with configurable parameters
  - `query`: Search query text
  - `top_k`: Number of results (default: 5)
  - `prune_ratio`: Attention pruning ratio (default: 1.0)

_Results will be updated after conference publication_

## 🎯 Key Features

- **Multilingual Retrieval**: Support for multiple languages through ColQwen2.5
- **Attention-Aware**: Hierarchical attention mechanisms for better semantic understanding
- **Scalable Indexing**: Efficient dual-index strategy for large-scale datasets
- **RESTful API**: Easy integration with existing systems
- **Comprehensive Evaluation**: BEIR benchmark integration

## 📚 Citation

If you use this code in your research, please cite our paper:

**arXiv Version:**

```bibtex
@misc{bach2025hierarchicalpatchcompressioncolpali,
      title={Hierarchical Patch Compression for ColPali: Efficient Multi-Vector Document Retrieval with Dynamic Pruning and Quantization},
      author={Duong Bach},
      year={2025},
      eprint={2506.21601},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2506.21601},
}
```

**Conference Version (KDIR 2025):**

Not yet

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ColQwen2.5 model by T-Systems
- FAISS library for efficient similarity search
- BEIR benchmark for evaluation framework
- KDIR 2025 conference committee

---

**For questions or collaboration opportunities, please contact us or open an issue on GitHub.**
