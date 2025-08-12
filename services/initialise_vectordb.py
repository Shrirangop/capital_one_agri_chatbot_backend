# /services/vector_db.py

import time
import logging
from typing import Dict, List, Optional, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
import config
from utils.document_processor import load_and_split_pdfs
import numpy as np
try:  # optional FAISS
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

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

# ---------------- FAISS GLOBAL RE-RANKING RETRIEVER -----------------
class MultiIndexFAISSRetriever(BaseRetriever):
    """Collects candidates from each Pinecone index then re-ranks globally with FAISS (or numpy fallback).

    Pydantic (v2) compatible: define fields instead of assigning dynamically in __init__.
    """

    index_map: Dict[str, Any]
    embeddings: Any
    per_index_k: int = 6
    final_k: int = 4
    search_kwargs: Optional[dict] = None

    # Allow arbitrary types (Pinecone Index objects, embedding models)
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context):  # type: ignore[override]
        if self.search_kwargs is None:
            self.search_kwargs = {}

    def _gather_candidates(self, query: str):
        candidates: List[Document] = []
        for name, index in self.index_map.items():
            try:
                vs = get_vector_store(index, self.embeddings)
                results = vs.similarity_search_with_score(query, k=self.per_index_k)
            except Exception as e:
                logging.warning(f"Similarity search failed on index {name}: {e}")
                continue
            for doc, score in results:
                doc.metadata = dict(doc.metadata) if doc.metadata else {}
                doc.metadata['source_index'] = name
                doc.metadata['pinecone_score'] = score
                candidates.append(doc)
        return candidates

    def _rerank(self, query_emb: np.ndarray, docs: List[Document]):
        if not docs:
            return []
        try:
            doc_texts = [d.page_content for d in docs]
            doc_embs = self.embeddings.embed_documents(doc_texts)
            mat = np.array(doc_embs, dtype=np.float32)
        except Exception as e:
            logging.error(f"Embedding doc candidates failed: {e}")
            return docs[: self.final_k]

        def _normalize(m: np.ndarray):
            n = np.linalg.norm(m, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return m / n

        try:
            qn = _normalize(query_emb.reshape(1, -1))[0]
            dn = _normalize(mat)
        except Exception:
            qn, dn = query_emb, mat

        if _FAISS_AVAILABLE:
            try:
                index_flat = faiss.IndexFlatIP(dn.shape[1])
                index_flat.add(dn)
                sims, idxs = index_flat.search(qn.reshape(1, -1), min(self.final_k, dn.shape[0]))
                for i, sc in zip(idxs[0], sims[0]):
                    docs[i].metadata['faiss_score'] = float(sc)
                return [docs[i] for i in idxs[0]]
            except Exception as e:
                logging.warning(f"FAISS ranking failed, fallback numpy: {e}")

        sims = dn @ qn
        order = np.argsort(-sims)[: self.final_k]
        for i, sc in enumerate(sims):
            docs[i].metadata['faiss_score'] = float(sc)
        return [docs[i] for i in order]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            q_emb = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        except Exception as e:
            logging.error(f"Query embedding failed: {e}")
            return []
        candidates = self._gather_candidates(query)
        return self._rerank(q_emb, candidates)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

def build_faiss_multi_index_retriever(index_map, embeddings, per_index_k=6, final_k=4, search_kwargs=None):
    if not index_map:
        raise ValueError("index_map empty")
    return MultiIndexFAISSRetriever(
        index_map=index_map,
        embeddings=embeddings,
        per_index_k=per_index_k,
        final_k=final_k,
        search_kwargs=search_kwargs,
    )

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