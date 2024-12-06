# teRAG: Topic Enhanced Retrieval Augmented Generation

Our research comprised of two attempts at the model. The first is contained in `models/te_rag.py`. In the paper, we describe how this version underperformed vanilla baseline due to the introduction of biases when topic distributions failed to find quality answers. The next version of the model, `models/llm_te_rag.py` fixes this issue by using a self-knowledge agent to evaluate the prompts before performing RAG. This led to improvements over the baseline by allowing the model to look for query-relevant data only when RAG is necessary.

## Requirements
The package relies on the `gensim` implementation of Latent Dirichlet Allocation, which can be installed via pip.

## Usage
To perform local evaluation, run `python local_evaluation.py` from the command line. To use the model in your own projects, import the model object `LLMPreCheckTopicModelingRAG` from `llm_te_rag.py`.
