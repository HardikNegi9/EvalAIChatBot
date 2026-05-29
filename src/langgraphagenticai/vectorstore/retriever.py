import json
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Any
import asyncio

class HybridCohereRetriever(BaseRetriever):
    faiss_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    compressor: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Fetch from FAISS
        faiss_docs = self.faiss_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        # 2. Fetch from BM25
        bm25_docs = self.bm25_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        
        # 3. Merge and deduplicate by content
        unique_docs = {}
        for doc in faiss_docs + bm25_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        merged_docs = list(unique_docs.values())
        
        # 4. Rerank using Cohere
        compressed_docs = self.compressor.compress_documents(merged_docs, query)
        
        # Cohere returns a list of Document objects with metadata 'relevance_score'
        return list(compressed_docs)


class Retriever:
    retriever = None
    embeddings = None

    def __init__(self):
        pass

    # Load urls documents
    def load_urls_documents(self, urls):
        docs = []
        for url in urls:
            docs.extend(WebBaseLoader(url).load())
        return docs

    # Load pdf documents
    async def load_pdf_documents(self, pdf_file):
        docs = []
        loader = PyPDFLoader(pdf_file)
        async for page in loader.alazy_load():
            docs.append(page)
        return docs

    # Load embeddings
    def load_embeddings(self):
        if Retriever.embeddings is None:
            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}  # Use 'cuda' for GPU
            encode_kwargs = {'normalize_embeddings': False}
            Retriever.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        return Retriever.embeddings

    # Split documents
    def split_documents(self, docs_list):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits

    # Add to vectorDB
    async def set_retriever(self):
        if Retriever.retriever is None:
            with open("data.json", "r") as f:
                data = json.load(f)

            docs_list = []
            if "urls" in data:
                docs_list.extend(self.load_urls_documents(data["urls"]))
            if "pdf_file" in data:
                docs_list.extend(await self.load_pdf_documents(data["pdf_file"]))

            splits = self.split_documents(docs_list)
            
            # Stage 1: Vector Search (Dense)
            vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=self.load_embeddings(),
            )
            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            
            # Stage 2: Keyword Search (Sparse)
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 10
            
            # Stage 3 & 4: Custom Hybrid Reranker
            compressor = CohereRerank(model="rerank-english-v3.0", top_n=4)
            compression_retriever = HybridCohereRetriever(
                faiss_retriever=faiss_retriever,
                bm25_retriever=bm25_retriever,
                compressor=compressor
            )

            Retriever.retriever = compression_retriever

    async def async_get_retriever(self):
        if Retriever.retriever is None:
            await self.set_retriever()
        return Retriever.retriever

    def get_retriever(self):
        return asyncio.run(self.async_get_retriever())