# /services/vector_db.py

import time
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import config
from utils.document_processor import load_and_split_pdfs

def initialize_pinecone_indexes():
    """Initialize Pinecone client and ensure all configured indexes exist.

    Returns:
        tuple[Pinecone, dict[str, pinecone.Index]]: Client and mapping of index name to Index objects.
    """
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    existing = {idx["name"] for idx in pc.list_indexes()}
    indexes = {}
    for name in config.PINECONE_INDEX_NAMES:
        if name not in existing:
            logging.info(f"Creating Pinecone index: {name} ...")
            pc.create_index(
                name=name,
                dimension=config.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not pc.describe_index(name).status["ready"]:
                logging.info(f"Waiting for index '{name}' to be ready...")
                time.sleep(5)
            logging.info(f"Index '{name}' created.")
        else:
            logging.info(f"Index '{name}' already exists.")
        indexes[name] = pc.Index(name)
    return pc, indexes

def get_vector_store(index, embeddings):
    """Return langchain PineconeVectorStore wrapper for a single index."""
    return PineconeVectorStore(index=index, embedding=embeddings)

def build_multi_index_retriever(index_map, embeddings, search_type=None, search_kwargs=None, weights=None):
    """Create an EnsembleRetriever over multiple Pinecone indexes.

    Args:
        index_map (dict[str, pinecone.Index]): Mapping of index name to Index objects.
        embeddings: Embedding function/model.
        search_type: Optional override for search type; defaults to config.RETRIEVER_SEARCH_TYPE.
        search_kwargs: Optional override for search kwargs; defaults to config.RETRIEVER_SEARCH_KWARGS.
        weights: List of float weights for each retriever (same order as index_map.keys()); defaults to equal weights.

    Returns:
        EnsembleRetriever
    """
    if not index_map:
        raise ValueError("index_map is empty; cannot build retriever.")
    search_type = search_type or config.RETRIEVER_SEARCH_TYPE
    search_kwargs = search_kwargs or config.RETRIEVER_SEARCH_KWARGS

    retrievers = []
    names = list(index_map.keys())
    for name in names:
        vs = get_vector_store(index_map[name], embeddings)
        retrievers.append(vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs))

    if weights and len(weights) != len(retrievers):
        raise ValueError("Length of weights must match number of indexes")
    if not weights:
        weights = [1.0] * len(retrievers)

    return EnsembleRetriever(retrievers=retrievers, weights=weights)

def setup_knowledge_base_for_index(index, embeddings):
    """Populate a single index with foundational documents if empty."""
    try:
        if index.describe_index_stats().get('total_vector_count', 0) > 0:
            logging.info(f"Index '{index._name}' already populated; skipping ingestion.")
            return
        chunks = load_and_split_pdfs()
        if not chunks:
            logging.info("No documents found to ingest.")
            return
        logging.info(f"Adding {len(chunks)} chunks to index '{index._name}'.")
        get_vector_store(index, embeddings).add_documents(documents=chunks)
        logging.info(f"Ingestion complete for index '{index._name}'.")
    except Exception as e:
        logging.error(f"Failed to setup knowledge base for index '{getattr(index,'_name','?')}': {e}")

def setup_all_knowledge_bases(index_map, embeddings, target_indexes=None):
    """Ingest documents into selected indexes (or all if target_indexes None)."""
    names = target_indexes or list(index_map.keys())
    for name in names:
        setup_knowledge_base_for_index(index_map[name], embeddings)

def multi_index_similarity_search(query: str, index_map, embeddings, k: int = 4, score_key: str = 'score'):
    """Run a similarity search across all provided indexes and merge results.

    Args:
        query: Natural language query string.
        index_map: Dict of name -> pinecone.Index.
        embeddings: Embedding function/model (same used to build the indexes).
        k: Top k per index to retrieve before merging.
        score_key: Metadata key name to place the similarity score (for downstream use).

    Returns:
        List[Document]: Combined and re-sorted documents with added metadata:
            source_index: which index produced the doc
            score_key (default 'score'): similarity score
    """
    all_docs = []
    for name, index in index_map.items():
        vs = get_vector_store(index, embeddings)
        try:
            results = vs.similarity_search_with_score(query, k=k)
        except Exception as e:
            logging.error(f"Similarity search failed on index '{name}': {e}")
            continue
        for doc, score in results:
            doc.metadata = dict(doc.metadata) if doc.metadata else {}
            doc.metadata['source_index'] = name
            doc.metadata[score_key] = score
            all_docs.append(doc)

    # Sort descending by score (assuming higher is more similar for cosine in Pinecone)
    all_docs.sort(key=lambda d: d.metadata.get(score_key, 0), reverse=True)
    return all_docs