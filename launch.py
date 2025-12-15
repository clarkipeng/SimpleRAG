
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["UVICORN_LOG_LEVEL"] = "critical"
os.environ["LLAMA_LOG_LEVEL"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] ="false"

from transformers.utils.logging import disable_progress_bar
disable_progress_bar()
import transformers
transformers.logging.set_verbosity_error()
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

from datetime import datetime

from dataclasses import asdict, dataclass, field
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Digits, RichLog, Input, Markdown

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer

from serve import Backend
from rag import Rag

import logging, sys
logging.basicConfig(
    filename="debug.log",
    filemode="w",              # Overwrite log on every run
    level=logging.INFO,        # Capture INFO and DEBUG logs (not just ERROR)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True                 # Force override any previous config
)

# 2. Redirect stdout/stderr to the log file
# This catches libraries that use 'print()' instead of 'logging'
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
    def flush(self): pass

sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

@dataclass
class ModelArguments:
    model: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "HF model name"},
    )
    dtype: str = field(
        default="fp8",
        metadata={"help": "quantization"},
    )
    device: str = field(
        default = "cpu",
        metadata={"help": "device type"}
    )
    
class TUIApp(App):
    CSS = """
    Screen { align: center middle; }
    #chat_container { 
        margin: 1 1; 
        border: solid green; 
        height: 80%; 
        overflow-y: auto; # Allow scrolling
    }
    Input { margin: 1 1; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit application"),
                ]
    
    
    
    def __init__(self, config:ModelArguments):
        self.config = config
        
        self.messages = []
        
        self.chat_container = VerticalScroll(id="chat_container")
        self.input = Input(id="input")
        
        self.system_prompt = "You are a financial helper agent, each user input you receive will be coupled with some reference text, use the text to inform your responses."

        self.backend = Backend(self.config)
        self.rag = Rag(self.config)
        super().__init__()

    def compose(self) -> ComposeResult:
        # yield RichLog(id="chat")
        yield self.chat_container
        yield self.input

    def on_mount(self) -> None:
        self.messages.append({"role":"system","content":self.system_prompt})
        self.add_md_message("SYSTEM",self.system_prompt)
        # self.load_model()
        # self.call_from_thread(self.update_input_state, True, "Loading model weights...")
        
        
        # self.call_from_thread(self.update_input_state, False, "Type a message...")
        
        
    def on_stop(self) -> None:
        self.backend.shutdown()
        
    # @work(exclusive=True, thread=True)
    # def load_model(self) -> None:
    
    @on(Input.Submitted)
    def start_generation(self, event: Input.Submitted) -> None:
        self.input.value = ""
        self.update_input_state(True, "Generating...")
        
        text = event.value
        self.add_md_message("User", text)
        
        ref_text = self.rag.get_reference_text(text)
        if ref_text:
            self.add_md_message("Rag", (ref_text[:200]+'...' if len(ref_text)>200 else ref_text))
            text += f"Reference text: {ref_text}"
        
        self.messages.append({"role": "user", "content": text})
        
        
        self.run_worker(self.infer, exclusive=True, thread=True)
        
    @work(exclusive=True, thread=True)
    def infer(self):
        self.call_from_thread(self.add_md_message, "Assistant")
        
        response_text = ""
        stream = self.backend.call_model(self.messages)
        
        for new_text in stream:
            response_text += new_text
            self.call_from_thread(self.stream_update, response_text)
        self.messages.append({"role": "assistant", "content": response_text})
        self.call_from_thread(self.update_input_state, False, "Type a new message...")
        
    def update_input_state(self, disabled: bool, placeholder: str = "") -> None:
        """Helper to update Input widget from the main thread."""
        inp = self.input
        inp.disabled = disabled
        if placeholder:
            inp.placeholder = placeholder
        if not disabled:
            inp.focus()
    def add_md_message(self, role, text = ""):
        """Adds the user's message to the chat."""
        md = Markdown(f"**{role}:**\n{text}")
        if role == "Assistant":
            self.current_message = md
        self.chat_container.mount(
            md
        )
    def stream_update(self, text: str):
        text = text.replace("<think>", "\n> **Thinking:**\n> ")
        text = text.replace("</think>", "\n> **Thinking Done:**\n> ")
        
        self.current_message.update(f"**Assistant:**\n{text}")
        self.current_message.scroll_visible()


if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    app = TUIApp(ModelArguments(**data))
    app.run()