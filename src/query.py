from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import sqlite3
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    prune_ratio: float = 1.0

class SearchResult(BaseModel):
    source_file: str
    chunk_index: int
    start_char: int
    score: float

class QueryResponse(BaseModel):
    results: list[SearchResult]

# App init
device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = FastAPI()

# Load model
model = ColQwen2_5.from_pretrained(
    'tsystems/colqwen2.5-3b-multilingual-v1.0',
    torch_dtype=torch.bfloat16,
    device_map='auto'
).eval()
processor = ColQwen2_5_Processor.from_pretrained(
    'tsystems/colqwen2.5-3b-multilingual-v1.0'
)

# Load index
index = faiss.read_index('indexes/faiss_hnsw.idx')

# Load metadata
def get_db():
    conn = sqlite3.connect('indexes/metadata.db', check_same_thread=False)
    return conn
conn = get_db()
cur = conn.cursor()

def fetch_metadata(ids):
    placeholders = ','.join('?'*len(ids))
    cur.execute(
        f"SELECT chunk_id, source_file, chunk_index, start_char FROM metadata WHERE chunk_id IN ({placeholders})",
        ids
    )
    return cur.fetchall()

@app.post('/search', response_model=QueryResponse)
def search(req: QueryRequest):
    # Embed
    inputs = processor.process_queries([req.query], return_tensors='pt').to(device)
    with torch.no_grad():
        embs, atts = model.get_patch_embeddings_and_attentions(**inputs)
    # Prune
    if 0 < req.prune_ratio < 1.0:
        M = atts.shape[-1]
        keep = int(M * req.prune_ratio)
        idxs = torch.argsort(atts, descending=True)[:, :keep]
        embs = torch.gather(embs, 1, idxs.unsqueeze(-1).expand(-1, -1, embs.size(-1)))
    vecs = embs.cpu().numpy().reshape(-1, embs.size(-1))
    # Search
    D, I = index.search(vecs, req.top_k)
    scores = D.mean(axis=0).tolist()
    chunk_ids = I[0].tolist()
    meta = fetch_metadata(chunk_ids)
    results = []
    for cid, sc in zip(chunk_ids, scores):
        for r in meta:
            if r[0] == cid:
                results.append(SearchResult(
                    source_file=r[1], chunk_index=r[2], start_char=r[3], score=sc
                ))
                break
    if not results:
        raise HTTPException(status_code=404, detail="No results")
    return QueryResponse(results=results)

# To run: uvicorn query:app --reload