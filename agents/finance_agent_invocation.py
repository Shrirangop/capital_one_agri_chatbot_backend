# agents/finance_agent_invocation.py

import logging
from typing import AsyncGenerator

# NOTE: In a real app, you would create a dedicated RAG chain for finance
# in 'services/initialise_llm.py'. For simplicity, we can reuse the generic one.
# from services.initialise_llm import create_generic_rag_chain

async def invoke_finance_agent_chain(
    rag_chain,
    retriever, # This will be the finance-specific retriever
    query: str,
) -> AsyncGenerator[str, None]:
    """
    Gathers context and invokes the Finance RAG chain.

    Args:
        rag_chain: The initialized, runnable RAG chain.
        retriever: The retriever for the finance knowledge base.
        query (str): The user's question.

    Yields:
        str: Chunks of the response.
    """
    logging.info(f"Finance Agent invoked for query: '{query}'")

    # 1. Retrieve relevant documents from the finance vector store
    retrieved_docs = retriever.invoke(query)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    context_str = format_docs(retrieved_docs)

    # 2. Construct the input for the RAG chain.
    # Note: We omit location/weather as they might be irrelevant for finance.
    chain_input = {
        "query": query,
        "context": context_str,
        # Provide default/empty values for unused keys in the generic prompt
        "location": "N/A",
        "crop_name": "N/A",
        "weather": "N/A",
    }

    # 3. Stream the response from the RAG chain
    async for chunk in rag_chain.astream(chain_input):
        yield chunk

    logging.info("Finance Agent streaming finished.")