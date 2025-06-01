import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from langchain_ollama import ChatOllama

from lazy_embeddings import *

loader = PyPDFLoader("/Users/leopoldotodisco/Documents/Smart-Study-Agent/test_documents/RAD.pdf")
pages = []
for page in loader.load():
    pages.append(page)

# una volta letti i doc devo splittarli e computare gli embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(pages)

# Index chunks
start_time = time.time()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = LazyEmbeddings()
vector_store = Chroma(
    collection_name="documents",
    persist_directory="./chroma_langchain_db",
)
_ = vector_store.add_documents(documents=all_splits)

question = """
@gau.route('/crea_area_assistenza', methods=['GET', 'POST'])
@login_required
def crea_area_assistenza():
    if request.method == 'POST' and session['isApicoltore']:
        descrizione = request.form.get('descrizione')
        apicoltore = get_apicoltore_by_id(current_user.id)
        if inserisci_area_assistenza(descrizione, apicoltore):
            current_user.descrizione = descrizione
            current_user.assistenza = True
        else:
            return crea_area_assistenza_page()
    return area_personale()
"""

question_4_embedding = "crea_area_assistenza"

retrieved_docs = vector_store.similarity_search(question_4_embedding, k=2)
print(f"### Embeddings required time = {time.time() - start_time}")

docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
messages = [
    {
        "role": "system", 
        "content": "You are a friendly assistant. Answer user's question."
     },
     {
        "role": "user", 
        "content": f"Explain this code: {question} \nUse this Context for providing better insights but do not mention it in the answer: {docs_content}"
     }
]

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

response = llm.invoke(messages)

print("\n========= RISPOSTA DEL LLM CON EMBEDDINGS =========\n")
print(response)

print(f"Total required time = {time.time() - start_time}")

print("\n========= DEBUG: DOCUMENTI RECUPERATI =========\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"[Documento {i}]\n{doc.page_content}\n")
