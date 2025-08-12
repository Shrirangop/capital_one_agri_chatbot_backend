import logging
import os
from typing import Optional
import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.schema import Document
import config


def _ensure_index(pc: Pinecone, name: str, dimension: int):
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name in existing:
        logging.info(f"Finance index '{name}' already exists.")
        return
    logging.info(f"Creating finance index '{name}' ...")
    pc.create_index(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )


def init_finance_index() -> tuple[Pinecone, any]:
    """Initialize Pinecone client and ensure finance index exists (capital-finance-index)."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    _ensure_index(pc, config.FINANCE_INDEX_NAME, config.EMBEDDING_DIMENSION)
    return pc, pc.Index(config.FINANCE_INDEX_NAME)


def _load_finance_docs() -> list[Document]:
    directory = config.FINANCE_DIRECTORY
    if not os.path.isdir(directory):
        logging.warning(f"Finance directory '{directory}' not found.")
        return []
    loader = DirectoryLoader(directory, loader_cls=UnstructuredFileLoader, show_progress=True)
    try:
        return loader.load()
    except Exception as e:
        logging.error(f"Failed loading finance directory '{directory}': {e}")
        return []


def ingest_finance_index_if_empty(index, embeddings):
    stats = index.describe_index_stats()
    if stats.get('total_vector_count', 0) > 0:
        logging.info("Finance index already populated; skipping ingestion.")
        return
    docs = _load_finance_docs()
    if not docs:
        logging.info("No finance documents to ingest.")
        return
    store = PineconeVectorStore(index=index, embedding=embeddings)
    store.add_documents(docs)
    logging.info(f"Ingested {len(docs)} finance documents.")


def build_finance_retriever(index, embeddings, search_kwargs=None):
    store = PineconeVectorStore(index=index, embedding=embeddings)
    return store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs=search_kwargs or config.RETRIEVER_SEARCH_KWARGS,
    )
