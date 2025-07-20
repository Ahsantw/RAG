# Retrieval-Augmented Generation (RAG)

This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) pipeline. Main features of this repository are:

1. Download the LLaMA model (meta-llama/Llama-3.1-8B-Instruct) from Hugging Face.
2. Convert the LLaMA model to OpenVINO INT4 format.
3. If the model has already been converted, load the existing INT4 version directly from folder.
4. Use a Sentence-Transformers embedding model to generate embeddings and store them using FAISS.
5. Build a question-answering (QA) system using LangChain to complete the RAG pipeline.
6. Logs for all the steps are documented in all_logs folder.
### Installation

1. Clone the Repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2. Create and Activate Conda Environment
```
conda env create -f environment.yaml
conda activate cbot
```

### Inference
All steps are done with one command.
```
python rag_loop.py
```

### Sample Output

```
Question ('exit'): whats procyon?
Answer:  UL Procyon is a suite of benchmark tests for professional users in various industries, designed to measure the performance of computers and devices.
```

### Tested Environment

This RAG pipeline was tested successfully on the following system:

- **OS**: Windows 10
- **Processor**: Intel Core i7 10th Gen
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3090
