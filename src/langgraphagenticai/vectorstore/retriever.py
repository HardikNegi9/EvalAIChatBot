import json
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio

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

            vectorstore = FAISS.from_documents(
                documents=self.split_documents(docs_list),
                embedding=self.load_embeddings(),
            )

            Retriever.retriever = vectorstore.as_retriever()

    async def async_get_retriever(self):
        if Retriever.retriever is None:
            await self.set_retriever()
        return Retriever.retriever

    def get_retriever(self):
        return asyncio.run(self.async_get_retriever())