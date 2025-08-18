import logging
from typing import AsyncGenerator

from services.brain_agent_chain import create_brain_agent_chain
from database.init_db import chat_histories_collection, users_collection
import config

# We no longer need to import the component variables directly
from services.call_different_agents import (
    test_crop_agent,
    test_finance_agent,
    test_tools_agent,
    CropTestRequest,
    FinanceTestRequest,
    ToolsTestRequest,
    _init_disease_services_if_needed  # Only import the init function
)
from agents.disease_agent_invocation import invoke_disease_agent_chain


def build_onboarding_form_message(user_id: str) -> str:
    """Return a structured message containing a link to the onboarding form."""
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
    user_id: int,
    image_file: bytes = None
) -> AsyncGenerator[str, None]:
    """
    Routes the query to the correct agent based on brain agent classification.
    """
    logging.info("Routing query with Brain Agent...")
    brain_chain = create_brain_agent_chain(brain_llm)

    if not await users_collection.find_one({"phone_number": user_id}):
        yield build_onboarding_form_message(str(user_id))
        return

    routing_decision = await brain_chain.ainvoke({"query": query})
    routing_decision = routing_decision.strip().lower()
    logging.info(f"Brain Agent decision: {routing_decision}")

    if "crop_agent" in routing_decision:
        req = CropTestRequest(question=query, user_id=user_id)
        response = await test_crop_agent(req)
        yield response["answer"]

    elif "finance_agent" in routing_decision:
        req = FinanceTestRequest(question=query, user_id=user_id)
        response = await test_finance_agent(req)
        yield response["answer"]

    elif "tool_agent" in routing_decision:
        req = ToolsTestRequest(question=query, user_id=user_id)
        response = await test_tools_agent(req)
        yield response["answer"]

    elif "disease_agent" in routing_decision:
        if image_file is None:
            yield "Please upload an image of the crop for disease diagnosis."
            return
        
        # 1. Call the init function to get the required components
        rag_chain, retriever, embeddings_model = await _init_disease_services_if_needed()
        
        # 2. Pass these initialized components explicitly to the agent chain
        chunks = []
        async for part in invoke_disease_agent_chain(
            rag_chain=rag_chain,
            retriever=retriever,
            embeddings_model=embeddings_model,
            crop_name=query,
            phone_number=user_id,
            image_file=image_file
        ):
            chunks.append(str(part))
        
        answer = "".join(chunks)
        
        yield answer

    else:
        yield "I am an agricultural assistant. Please ask me questions about farming, crops, soil, pests, finances, or tools."