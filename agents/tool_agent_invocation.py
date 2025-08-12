import logging
from typing import AsyncGenerator
from database.init_db import users_collection

async def _fetch_user_data(phone_number: int) -> dict:
	user = await users_collection.find_one({"phone_number": phone_number})
	if not user:
		return {"phone_number": phone_number, "location": "", "curr_crop_name": ""}
	return {
		"phone_number": user.get("phone_number", ""),
		"location": user.get("location", ""),
		"curr_crop_name": user.get("curr_crop_name", ""),
	}

async def invoke_tool_agent_chain(
	rag_chain,
	retriever,
	query: str,
	phone_number: int
) -> AsyncGenerator[str, None]:
	"""Invoke tools RAG chain using capital-tools-index retriever only."""
	logging.info(f"Tools Agent invoked for query: '{query}'")
	user_data = await _fetch_user_data(phone_number)
	location = user_data.get("location", "") or "N/A"
	crop_name = user_data.get("curr_crop_name", "") or "N/A"

	docs = retriever.invoke(query)
	context = "\n\n".join(d.page_content for d in docs)

	chain_input = {
		"query": query,
		"general_context": context,
		"location": location,
		"crop_name": crop_name,
		"weather": "N/A",
		"short_term_answers": "",
		"long_term_a": ""
	}
	try:
		async for chunk in rag_chain.astream(chain_input):
			yield chunk
	except Exception as e:
		logging.error(f"Tools RAG chain streaming failed: {e}")
		yield f"Error: {e}"
