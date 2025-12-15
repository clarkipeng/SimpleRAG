

# Instructions 

Install uv if you don't have it,
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# How to run the tool.

```
uv run launch.py config.yaml
```

# Examples of usage and expected outputs. 

My implementation isn't exactly a CLI, its a TUI, so when you run launch, you should be prompted with a message bar and the chat history. Then you should be able to ask questions to the agent.

# Description of the RAG mechanism

High level flow -> prompt -> embeded using `all-MiniLM-L6-v2` -> find most similar text sample from Reuters-21578 (Financial) -> append to user prompt.

Since we always provide the top-1 sample as context, some prompts might not be similar to any text sample, and might get appeneded a unhelpful text sample. This can be circumvented by modifying the similarity_cutoff in the config which is the cosine-sim cutoff for appending the rag context.

# Implementation details

Textual is used for the tui
Model is served on a separate thread, to support non-python libraries like llama.cpp.
HF and llama.cpp are supported for LLM serving.