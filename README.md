# Retrieval-Augmented Generation (RAG)

This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) pipeline. Main features of this repository are:

1. Download the LLaMA model (meta-llama/Llama-3.1-8B-Instruct) from Hugging Face.
2. Convert the LLaMA model to OpenVINO INT4 format.
3. If the model has already been converted, load the existing INT4 version directly from folder.
4. Use a Sentence-Transformers embedding model to generate embeddings and store them using FAISS.
5. Build a question-answering (QA) system using LangChain to complete the RAG pipeline.
6. Logs for all the steps are documented in all_logs folder. So very easy to debug any issues.
### Installation

1. Clone the Repository
```
git clone https://github.com/Ahsantw/RAG
cd RAG
```
2. Install Python 3.10
3. Install required Pakages
```
pip install -r requirements.txt
```
4. Log into to hugginface accout (optional if you do not have HF model). You will be asked to paste [HF token](https://huggingface.co/docs/hub/en/security-tokens).
```
huggingface-cli login
```


### Inference
All steps are done with one command. At the end when all steps are done, you can easly type questions.
```
python rag_cli.py
```

### Sample Output

```
Question ('exit'): whats procyon
 - Page 1, File: data/procyon_guide.pdf
 - Page 39, File: data/procyon_guide.pdf
 - Page 13, File: data/procyon_guide.pdf
Answer:  UL Procyon is a suite of benchmark tests for professional users in various industries, designed to measure the performance of computers and devices.
```

### Hardware Specs.

This RAG pipeline was tested successfully on the following system:

- **OS**: Windows 10
- **Processor**: Intel Core i7 10th Gen
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3090
