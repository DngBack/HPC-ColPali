from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from retriever_adapter import ColPaliRetriever

# Load benchmark data (e.g., FiQA)
data_path = "beir_datasets/fiqa"
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Instantiate retriever
retriever = ColPaliRetriever()

# Run retrieval
results = retriever.retrieve(corpus, queries, top_k=10)

# Evaluate
evaluator = EvaluateRetrieval()
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[1,3,5,10])
print("nDCG@10:", ndcg["10"])
print("Recall@10:", recall["10"])
print("MAP@10:", _map["10"])