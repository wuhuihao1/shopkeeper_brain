from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-reranker-large", local_dir=r"W:\LocalModels\RerankerModels\bge-reranker-large")

print(local_dir)