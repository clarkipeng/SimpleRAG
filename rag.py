import os
import sys


import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import subprocess

class Rag:
    def __init__(self, config):
        df = pd.read_csv('reuters.csv')
        self.corpus = df['text'].dropna().unique().tolist()

        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.model_path = "./all-MiniLM-L6-v2"
        if not os.path.exists(self.model_path):
            get_path_script = f"""
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='{model_id}',local_dir='{self.model_path}'))
"""
            self.model_path = subprocess.check_output(
                [sys.executable, "-c", get_path_script],
                text=True,
                close_fds=True
            ).strip()
        
        self.model = SentenceTransformer(self.model_path, device='cpu')
        self.ref_embeddings = self.model.encode(self.corpus, convert_to_tensor=True, show_progress_bar=False)
        self.cutoff = config.similarity_cutoff
    def get_reference_text(self, prompt):
        quer_emb = self.model.encode(prompt, convert_to_tensor=True, show_progress_bar=False)
        search_results = util.semantic_search(quer_emb, self.ref_embeddings, top_k=1)
        
        for result in search_results[0]:
            # result structure: {'corpus_id': 123, 'score': 0.85}
            doc_id = result['corpus_id']
            score = result['score']
            if score>self.cutoff:
                return self.corpus[doc_id]
        return ""

if __name__ == "__main__":
    import yaml
    from launch import ModelArguments
    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    app = Rag(ModelArguments(**data))