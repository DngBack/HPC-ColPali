import faiss
import torch
import sqlite3
import numpy as np
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

class ColPaliRetriever:
    def __init__(self, index_path="indexes/faiss_hnsw.idx", db_path="indexes/metadata.db"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ColQwen2_5.from_pretrained(
            'tsystems/colqwen2.5-3b-multilingual-v1.0',
            torch_dtype=torch.bfloat16,
            device_map='auto'
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(
            'tsystems/colqwen2.5-3b-multilingual-v1.0')
        self.index = faiss.read_index(index_path)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cur = self.conn.cursor()

    def embed_query(self, query):
        inputs = self.processor.process_queries([query], return_tensors='pt').to(self.device)
        with torch.no_grad():
            embs, _ = self.model.get_patch_embeddings_and_attentions(**inputs)
        return embs.cpu().numpy().reshape(-1, embs.size(-1))

    def fetch_doc_texts(self, ids):
        placeholders = ','.join('?'*len(ids))
        self.cur.execute(
            f"SELECT chunk_id, source_file FROM metadata WHERE chunk_id IN ({placeholders})",
            ids
        )
        return {row[0]: row[1] for row in self.cur.fetchall()}

    def retrieve(self, corpus, queries, top_k=10):
        results = {}
        for qid, query in queries.items():
            vecs = self.embed_query(query)
            D, I = self.index.search(vecs, top_k)
            doc_scores = {}
            for i in range(len(I[0])):
                doc_id = str(I[0][i])
                score = float(D[0][i])
                doc_scores[doc_id] = score
            results[qid] = doc_scores
        return results