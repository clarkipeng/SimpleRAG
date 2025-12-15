import os, sys


import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False
from litellm import completion

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
import litellm

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


from huggingface_hub import hf_hub_download


import transformers

import subprocess
import threading
from pathlib import Path
    
class Backend:
    # def __init__(self,config):
    #     self.config = config
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
    #     self.llm = AutoModelForCausalLM.from_pretrained(self.config.model,)
    #     self.streamer = TextIteratorStreamer(
    #         self.tokenizer,
    #         skip_prompt=True,
    #         skip_special_tokens=True,
    #     )
    #     self.device = 'cpu'
    # def call_model(self, messages):
    #     input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    #     generation_kwargs = dict(
    #         input_ids=input_ids,
    #         streamer=self.streamer,
    #         max_new_tokens=4096,
    #         temperature=0.7,
    #         do_sample=True,
    #     )
        
    #     thread = threading.Thread(target=self.llm.generate, kwargs=generation_kwargs)
    #     thread.start()
        
    #     for new_text in self.streamer:
    #         yield new_text
        
    #     thread.join()
    def __init__(self, config):
        repo_id = '/'.join(config.model.split('/')[:-1])
        model_name = config.model.split('/')[-1]
        self.model_path = f'./{model_name}'
        if not os.path.exists(self.model_path):
            get_path_script = f"""
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id='{repo_id}', filename='{model_name}',local_dir='./'))
"""
            self.model_path = subprocess.check_output(
                [sys.executable, "-c", get_path_script],
                text=True,
                close_fds=True
            ).strip()
        
        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", self.model_path,
            "--n_ctx", "8192",
            "--n_gpu_layers", "32",  # Enable GPU offload
            "--port", "8081"
        ]
        self.log_file = open("server_log.txt", "w")
        self.process = subprocess.Popen(cmd,
            stdout=self.log_file,
            stderr=self.log_file,
            close_fds=True,
            env=os.environ,
        )
        
    def call_model(self, messages):
        stream = completion(
            model="openai/local-model",
            api_base="http://localhost:8081/v1",
            api_key="x",
            messages=messages,
            stream=True,
            temperature=0.7,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    def shutdown(self):
        self.process.terminate()
        self.log_file.close()
        

if __name__ == "__main__":
    import yaml
    from launch import ModelArguments
    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    app = Backend(ModelArguments(**data))
    app.run()