# agents/brain_agent_invocation.py

import logging
from typing import AsyncGenerator
from services.brain_agent_chain import create_brain_agent_chain
from agents.crop_agent_invocation import invoke_crop_agent_chain
from agents.finance_agent_invocation import invoke_finance_agent_chain
from agents.tool_agent_invocation import invoke_tool_agent_chain
from agents.disease_agent_invocation import invoke_disease_agent_chain
from database.init_db import chat_histories_collection,users_collection
import config

def build_onboarding_form_message(user_id: str) -> str:
    """Return a structured message containing a link to the onboarding form.

    The front-end can detect this pattern (e.g., `FORM_LINK:`) and render a UI form.
    The form should collect: location (pincode/district), crop type, irrigation type, browser (auto-captured).
    """
    form_url = f"{getattr(config, 'USER_FORM_URL', 'https://your-form-url')}"
    instructions = (
        "Please complete the initial farm profile so I can tailor advice. "
        "Provide location (pincode), current crop, and irrigation type."
    )
    return (
        "ONBOARDING_REQUIRED\n" +
        f"FORM_LINK: {form_url}?user_id={user_id}\n" +
        "FIELDS: location, crop_type, irrigation_type\n" +
        f"MESSAGE: {instructions}"
    )

async def route_query(
    query: str,
    brain_llm,
    crop_rag_chain, crop_retriever, crop_embeddings,
    finance_rag_chain, finance_retriever, finance_embeddings,
    tools_rag_chain, tools_retriever, tools_embeddings,
    disease_rag_chain, disease_retriever, disease_embeddings,
    user_id: int,
    image_file: bytes = None
) -> AsyncGenerator[str, None]:
    """
    Routes the query to the correct agent based on brain agent classification.

    Args:
        query (str): The user's incoming question.
        brain_llm: The LLM instance for the fast routing agent.
        crop_rag_chain: The fully initialized, runnable RAG chain for the crop agent.
        crop_retriever: The retriever for the crop agent's knowledge base.
        crop_embeddings: The embeddings for the crop agent's knowledge base.
        finance_rag_chain: The fully initialized, runnable RAG chain for the finance agent.
        finance_retriever: The retriever for the finance agent's knowledge base.
        finance_embeddings: The embeddings for the finance agent's knowledge base.
        tools_rag_chain: The fully initialized, runnable RAG chain for the tool agent.
        tools_retriever: The retriever for the tool agent's knowledge base.
        tools_embeddings: The embeddings for the tool agent's knowledge base.
        disease_rag_chain: The fully initialized, runnable RAG chain for the disease agent.
        disease_retriever: The retriever for the disease agent's knowledge base.
        disease_embeddings: The embeddings for the disease agent's knowledge base.
        user_id (int): The ID of the user, used to fetch context and for routing.
        image_file (bytes, optional): An optional image file for disease diagnosis.

    Yields:
        A stream of strings forming the final answer.
    """
    logging.info("Routing query with Brain Agent...")
    brain_chain = create_brain_agent_chain(brain_llm)

    # Onboarding: if no user history, prompt for form
    if not await users_collection.find_one({"phone_number": user_id}):
        yield build_onboarding_form_message(user_id)
        return

    # Classify the query
    routing_decision = await brain_chain.ainvoke({"query": query})
    routing_decision = routing_decision.strip().lower()
    logging.info(f"Brain Agent decision: {routing_decision}")

    # Route to the correct agent
    if "crop_agent" in routing_decision:
        async for chunk in invoke_crop_agent_chain(
            crop_rag_chain, crop_retriever, crop_embeddings, query, user_id
        ):
            yield chunk
    elif "finance_agent" in routing_decision:
        async for chunk in invoke_finance_agent_chain(
            finance_rag_chain, finance_retriever, finance_embeddings, query, user_id
        ):
            yield chunk
    elif "tool_agent" in routing_decision:
        async for chunk in invoke_tool_agent_chain(
            tools_rag_chain, tools_retriever, tools_embeddings, query, user_id
        ):
            yield chunk
    elif "disease_agent" in routing_decision:
        if image_file is None:
            yield "Please upload an image of the crop for disease diagnosis."
            return
        async for chunk in invoke_disease_agent_chain(
            disease_rag_chain, disease_retriever, disease_embeddings, query, user_id, image_file
        ):
            yield chunk
    else:
        yield "I am an agricultural assistant. Please ask me questions about farming, crops, soil, pests, finances, or tools."