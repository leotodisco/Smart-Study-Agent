# RAG PIPELINE
# 1. Get and embed data
# 2. Index and store data
# 3. query data
# 4. evaluation (?)

from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import chromadb
from settings import DOCUMENT_FOLDER

import asyncio

async def main():
    reader = SimpleDirectoryReader(input_dir=DOCUMENT_FOLDER)
    documents = reader.load_data()
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("MyDocuments")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    ingestion_pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store
    )

    await ingestion_pipeline.arun(documents=documents)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    Settings.llm = Ollama(model='gemma3:4b', base_url='http://localhost:11434', temperature=0.0)
    query_engine = index.as_query_engine(response_mode="refine")

    res = query_engine.query("How can I use a tool in a workflow?")
    print(res)


asyncio.run(main())