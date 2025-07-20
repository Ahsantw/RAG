from src.convert_llama_to_open import OpenVINOLLMLoader
from src.log_setup import setup_logger
from src.vector_db import PDFVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.globals import set_verbose
import yaml
import re

set_verbose(False)
# Configure logging
logger = setup_logger(__name__, '')
logger.info(f"-----------------------STARTED LOGGING---------------------------")

def main(logger):
    loader = OpenVINOLLMLoader(logger)
    llm_model = loader.get_llm_model()


    db_class = PDFVectorStore(logger)
    db_class.create_db()

    db = db_class.read_db()


    prompt_template = PromptTemplate.from_template(
            """<|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are an expert assistant. Use the following extracted parts of a document to answer the question accurately and concisely. 
            If the answer is not found, say you don't know. Don't try to make up an answer.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Context:
            {context}

            Question: {question}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
            )

    while True:
        question = input("Question ('exit'): ")
        logger.info(f'Question {question}')
        if question.strip().lower() == "exit":
            break

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt_template},
            verbose=False,
        )

        result = qa_chain.invoke(question)
        answer = result['result']

        logger.info(f"Answer {answer}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    try:
        main(logger)
    except Exception as e:
        logger.error(f"Failed to Run the pipeline : {e}")
