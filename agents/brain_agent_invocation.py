# agents/brain_agent_invocation.py

import logging
from typing import AsyncGenerator
from services.brain_agent_chain import create_brain_agent_chain
from agents.crop_agent_invocation import invoke_crop_agent_chain

async def route_query(
    query: str,
    brain_llm,
    crop_rag_chain,
    ensemble_retriever
) -> AsyncGenerator[str, None]:
    """
    First, uses the brain agent to classify the query, then routes to the
    appropriate agent (Crop Agent or a general response).

    Args:
        query (str): The user's incoming question.
        brain_llm: The LLM instance for the fast routing agent.
        crop_rag_chain: The fully initialized, runnable RAG chain for the crop agent.
        ensemble_retriever: The retriever for the crop agent's knowledge base.

    Yields:
        A stream of strings forming the final answer.
    """
    logging.info("Routing query with Brain Agent...")
    brain_chain = create_brain_agent_chain(brain_llm)
    
    # Get the classification from the brain agent
    routing_decision = await brain_chain.ainvoke({"query": query})
    
    # Clean up the output in case of extra whitespace
    routing_decision = routing_decision.strip()
    logging.info(f"Brain Agent decision: {routing_decision}")

    if "crop_agent" in routing_decision:
        # If routed to crop agent, invoke it with all necessary context
        logging.info("Query routed to Crop Agent. Invoking...")
        async for chunk in invoke_crop_agent_chain(crop_rag_chain, ensemble_retriever, query):
            yield chunk

    elif "general_agent" in routing_decision:
        # Handle general queries here.
        # This could be a different agent or a simple canned response.
        logging.info("Query routed to General Agent.")
        yield "I am an agricultural assistant. Please ask me questions about farming, crops, soil, or pests."

    elif "finance_agent" in routing_decision:
        # Handle finance-related queries here.
        # This could be a different agent or a specialized finance chain.
        logging.info("Query routed to Finance Agent.")
        yield "I can help with agricultural finances and government schemes. Please ask your question about farm loans, subsidies, or financial planning."
    
    elif "tool_agent" in routing_decision:
        # Handle tool-related queries here.
        # This could be a different agent or a specialized tool chain.
        logging.info("Query routed to Tool Agent.")
        yield "I can assist with questions about farming tools and machinery. Please ask your question about equipment selection, maintenance, or operation."

        
    else:
        # Handle general queries here.
        # This could be a different agent or a simple canned response.
        logging.info("Query routed to General Agent.")
        yield "I am an agricultural assistant. Please ask me questions about farming, crops, soil, or pests."