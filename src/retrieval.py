import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from src.reranking import LLMReranker, JinaReranker

_log = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, bm25_db_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.bm25_index, self.chunk_catalog = self._load_index_and_catalog()

    def _load_index_and_catalog(self):
        index_path = self.bm25_db_dir / "bm25.pkl"
        catalog_path = self.bm25_db_dir / "bm25_chunks.json"

        if not index_path.exists() or not catalog_path.exists():
            raise FileNotFoundError(f"Global BM25 artifacts not found in {self.bm25_db_dir}")

        with open(index_path, "rb") as file:
            bm25_index = pickle.load(file)
        with open(catalog_path, "r", encoding="utf-8") as file:
            chunk_catalog = json.load(file)

        return bm25_index, chunk_catalog

    def retrieve(self, query: str, top_n: int = 5) -> List[Dict]:
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)

        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]

        results = []
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = self.chunk_catalog[index]
            results.append({
                "distance": score,
                "document_id": chunk.get("document_id"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
            })

        return results


class VectorRetriever:
    def __init__(self, vector_db_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.vector_db, self.chunk_catalog = self._load_index_and_catalog()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        proxy_pass = os.getenv("PROXY_PASSWORD")
        proxy_user = os.getenv("PROXY_USERNAME")
        if not proxy_pass:
            raise RuntimeError("PROXY_PASSWORD missing")
        if not proxy_user:
            raise RuntimeError("PROXY_USERNAME missing")

        os.environ["HTTP_PROXY"] = f"http://{proxy_user}:{proxy_pass}@5.129.219.79:3128"
        os.environ["HTTPS_PROXY"] = f"http://{proxy_user}:{proxy_pass}@5.129.219.79:3128"

        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=None, max_retries=2)

    def _load_index_and_catalog(self):
        index_path = self.vector_db_dir / "global.faiss"
        catalog_path = self.vector_db_dir / "global_chunks.json"

        if not index_path.exists() or not catalog_path.exists():
            raise FileNotFoundError(f"Global vector DB artifacts not found in {self.vector_db_dir}")

        vector_db = faiss.read_index(str(index_path))
        with open(catalog_path, "r", encoding="utf-8") as file:
            chunk_catalog = json.load(file)

        if vector_db.ntotal != len(chunk_catalog):
            _log.warning(
                "Vector DB entries (%s) and catalog entries (%s) mismatch.",
                vector_db.ntotal,
                len(chunk_catalog),
            )

        return vector_db, chunk_catalog

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=None, max_retries=2)
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return round(similarity_score, 4)

    def retrieve(self, query: str, top_n: int = 5) -> List[Dict]:
        actual_top_n = min(top_n, len(self.chunk_catalog))
        embedding = self.llm.embeddings.create(input=query, model="text-embedding-3-large").data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.vector_db.search(x=embedding_array, k=actual_top_n)

        results = []
        for distance, index in zip(distances[0], indices[0]):
            if index < 0 or index >= len(self.chunk_catalog):
                continue
            chunk = self.chunk_catalog[index]
            results.append({
                "distance": round(float(distance), 4),
                "document_id": chunk.get("document_id"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
            })

        return results


class HybridRetriever:
    def __init__(self, vector_db_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir)
        self.reranker = LLMReranker()

    def retrieve(self, query: str, llm_reranking_sample_size: int = 28, documents_batch_size: int = 2,
                 top_n: int = 6, llm_weight: float = 0.7) -> List[Dict]:
        vector_results = self.vector_retriever.retrieve(query=query, top_n=llm_reranking_sample_size)

        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight,
        )

        return reranked_results[:top_n]


class JinaHybridRetriever:
    def __init__(self, vector_db_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir)
        self.reranker = JinaReranker()

    def retrieve(
        self,
        query: str,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7
    ) -> List[Dict]:
        # documents_batch_size and llm_weight are kept for interface compatibility
        _ = documents_batch_size
        _ = llm_weight

        vector_results = self.vector_retriever.retrieve(query=query, top_n=llm_reranking_sample_size)
        if not vector_results:
            return []

        documents = [doc.get("text", "") for doc in vector_results]
        jina_response = self.reranker.rerank(query=query, documents=documents, top_n=min(top_n, len(documents)))
        jina_results = jina_response.get("results", [])
        print(len(jina_results))

        reranked_results = []
        for rank_item in jina_results:
            doc_index = rank_item.get("index")
            if doc_index is None or doc_index < 0 or doc_index >= len(vector_results):
                continue

            doc = vector_results[doc_index].copy()
            relevance_score = round(float(rank_item.get("relevance_score", 0.0)), 4)
            doc["relevance_score"] = relevance_score
            doc["combined_score"] = relevance_score
            reranked_results.append(doc)

        return reranked_results[:top_n]
