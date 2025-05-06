import os
import math
import sqlite3
import numpy as np
import faiss
import torch
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Config
PDF_DIR = "data/"
INDEX_DIR = "indexes/"
os.makedirs(INDEX_DIR, exist_ok=True)

# 1. Load PDFs
loader = UnstructuredPDFLoader
all_docs = []
for fname in os.listdir(PDF_DIR):
    if fname.lower().endswith('.pdf'):
        path = os.path.join(PDF_DIR, fname)
        docs = loader(path).load()
        for doc in docs:
            doc.metadata['source_file'] = fname
        all_docs.extend(docs)

# 2. Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
texts = [chunk.page_content for chunk in chunks]

# 3. Load model & processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ColQwen2_5.from_pretrained(
    'tsystems/colqwen2.5-3b-multilingual-v1.0',
    torch_dtype=torch.bfloat16,
    device_map='auto'
).eval()
processor = ColQwen2_5_Processor.from_pretrained(
    'tsystems/colqwen2.5-3b-multilingual-v1.0'
)

# 4. Extract embeddings & attention
batch_size = 8
all_embs = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    inputs = processor.process_queries(batch, return_tensors='pt').to(device)
    with torch.no_grad():
        embs, atts = model.get_patch_embeddings_and_attentions(**inputs)
    # flatten patches
    bs, m, d = embs.shape
    all_embs.append(embs.cpu().numpy().reshape(bs*m, d))
all_embs = np.vstack(all_embs).astype('float32')

# 5. Build indices
d = all_embs.shape[1]
# 5.1 HNSW index
hnsw = faiss.IndexHNSWFlat(d, 32)
print('Training HNSW...')
hnsw.add(all_embs)
faiss.write_index(hnsw, os.path.join(INDEX_DIR, 'faiss_hnsw.idx'))

# 5.2 PQ index
m = 8
nbits = 8
pq = faiss.IndexPQ(d, m, nbits)
print('Training PQ...')
pq.train(all_embs)
pq.add(all_embs)
faiss.write_index(pq, os.path.join(INDEX_DIR, 'faiss_pq.idx'))

# 6. Save metadata
conn = sqlite3.connect(os.path.join(INDEX_DIR, 'metadata.db'))
cur = conn.cursor()
cur.execute('''
CREATE TABLE IF NOT EXISTS metadata (
    chunk_id INTEGER PRIMARY KEY,
    source_file TEXT,
    chunk_index INTEGER,
    start_char INTEGER
)
''')
records = []
for idx, chunk in enumerate(chunks):
    records.append((idx, chunk.metadata['source_file'], chunk.metadata.get('chunk', 0), chunk.metadata.get('start', 0)))
cur.executemany('INSERT INTO metadata VALUES (?,?,?,?)', records)
conn.commit()
conn.close()
print('Indexing complete.')