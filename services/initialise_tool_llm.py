# services/initialise_llm.py

import logging
from typing import AsyncGenerator
from operator import itemgetter

# --- MODIFIED IMPORTS ---
# Removed: google.generativeai, langchain_google_genai
# Added: langchain_groq, langchain_huggingface
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# --- END MODIFIED IMPORTS ---

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config

def initialize_llm_and_embeddings_for_tool():
    """
    Initializes the Groq LLM (Llama 3.1) and a HuggingFace Embeddings model.
    
    Returns:
        tuple: (llm_instance, embeddings_model_instance)
    """
    try:
        # The API key is passed directly to the ChatGroq constructor.
        llm = ChatGroq(
            model_name=config.LLM_MODEL, 
            groq_api_key=config.GROQ_API_KEY, 
            temperature=0
        )
        
        # Groq does not provide an embedding API. We use a popular open-source 
        # model from HuggingFace which runs locally.
        embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            # To run on CPU, uncomment the following line
            # model_kwargs={'device': 'cpu'} 
        )
        
        logging.info(f"LLM ({config.LLM_MODEL}) and Embedding model ({config.EMBEDDING_MODEL}) initialized successfully.")
        return llm, embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        raise

# UNCHANGED: This function's logic is provider-agnostic.
def create_rag_chain_for_tool(llm, vector_store):
    """
    Creates the complete RAG chain. The same chain can be used for invoke() and stream().
    
    Args:
        llm: The initialized language model.
        vector_store: The primary vector store (e.g., Pinecone).
    
    Returns:
        A runnable LangChain object.
    """
    logging.info("Creating RAG chain...")

    general_retriever = vector_store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs=config.RETRIEVER_SEARCH_KWARGS
    )

    template = """You are an expert Agri-Tech Specialist AI. Your function is to analyze a farmer's on-field situation and recommend a list of specific agricultural tools, technologies, and equipment, including their purpose and estimated prices.

INSTRUCTIONS:
1.  Carefully analyze the farmer's query and the on-field situation described in the `AVAILABLE CONTEXT` block. Pay close attention to the actions implied in the 'Details of Immediate Issues' and 'Future Planning Needs'.
2.  Based on the required tasks (e.g., spraying pesticides, improving irrigation, monitoring pests, soil preparation), identify a list of suitable tools and technologies from your knowledge base.
3.  For each tool, you must provide its name, type, a brief description of what it does, and an estimated price range.
4.  The final output MUST be a JSON array. Each element in the array will be an object representing one recommended tool.
5.  If the situation does not call for any specific tools, return an empty array `[]`.

---

## FARMER'S QUERY
{query}

---

## AVAILABLE CONTEXT (Farmer's On-Field Situation)
- **Topic Classification**: {general_context}
- **Location**: {location}
- **Crop Name**: {crop_name}
- **Relevant Weather**: {weather}
- **Details of Immediate Issues**: {short_term_answers}
- **Details of Future Planning Needs**: {long_term_a}

---

## OUTPUT SPECIFICATIONS
- **Format**: Your final output MUST be a valid JSON array `[...]`.
- **Object Structure**: Each object within the array must contain these and only these four keys: `tool_name` (string), `tool_type` (string, e.g., "Sprayer", "Sensor", "Tillage", "Trap"), `description` (string), and `estimated_price` (string, e.g., "₹5,000 - ₹8,000").
- **Relevance**: Recommendations must be directly relevant to solving the problems or implementing the plans described in the `FARMER'S QUERY` and `AVAILABLE CONTEXT`.
- **Pricing**: Provide a realistic price range in the local currency (e.g., INR for India) to account for market variations.

JSON_TOOL_RECOMMENDATIONS:
"""
    
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "prioritized_context": lambda x: format_docs(x['query_doc_retriever'].get_relevant_documents(x['question'])),
            "general_context": itemgetter("question") | general_retriever | format_docs,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Hybrid Q&A chain is ready.")
    return rag_chain

# UNCHANGED: This function's logic is provider-agnostic.
async def astream_rag_response_for_tool(rag_chain, question: str, query_doc_retriever) -> AsyncGenerator[str, None]:
    """
    Invokes the RAG chain using the .astream() method for asynchronous streaming.

    Args:
        rag_chain: The runnable RAG chain object.
        question: The user's question.
        query_doc_retriever: The dynamically created retriever for the uploaded document.

    Yields:
        str: Chunks of the response as they are generated by the LLM.
    """
    chain_input = {
        "question": question,
        "query_doc_retriever": query_doc_retriever
    }
    
    async for chunk in rag_chain.astream(chain_input):
        yield chunk