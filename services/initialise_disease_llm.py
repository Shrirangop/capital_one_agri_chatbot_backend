# services/initialise_llm.py

import logging
from typing import AsyncGenerator
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import config
import json

def initialize_llm_and_embeddings_for_disease():
    """
    Initializes the Groq LLM (Llama 3.1) and a HuggingFace Embeddings model.
    
    Returns:
        tuple: (llm_instance, embeddings_model_instance)
    """
    try:
        llm = ChatGroq(
            model_name=config.LLM_MODEL, 
            groq_api_key=config.GROQ_API_KEY, 
            temperature=0.2
        )
        
        embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs ={"normalize_embeddings": True}
        )
        
        logging.info(f"LLM ({config.LLM_MODEL}) and Embedding model ({config.EMBEDDING_MODEL}) initialized successfully.")
        return llm, embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        raise

def create_rag_chain_for_disease(llm):
    """
    Creates a RAG chain that extracts disease and crop from a JSON query,
    and generates a description and cure based on the provided context.

    Args:
        llm: An initialized LangChain compatible language model.

    Returns:
        A runnable LangChain object.
    """
    logging.info("Creating RAG chain for disease description and cure...")

    # 1. --- FOCUSED PROMPT TEMPLATE ---
    # This template is specifically designed to ask for a description and cure.
    template = """You are a specialized agricultural bot. Your task is to provide a clear description of the specified plant disease and recommend effective cures using ONLY the provided context.

INSTRUCTIONS:
1. Use the `Retrieved Knowledge` as your single source of truth to answer.
2. Structure your answer with two distinct sections: `## Description` and `## Cure`.
3. If the context is empty or insufficient to answer, Use general agricultural knowledge, but start with: "Based on general agricultural knowledge:"

---

## CONTEXT
- **Crop Name**: {crop_name}
- **Disease Name**: {disease_name}
- **Retrieved Knowledge**: {context}

---

## ANSWER
"""

    prompt = PromptTemplate.from_template(template)




    rag_chain = (
        {

            "context": itemgetter("context"),
            "crop_name": itemgetter("crop_name"),
            "disease_name": itemgetter("disease_name"),
            
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("RAG chain is ready.")
    return rag_chain

