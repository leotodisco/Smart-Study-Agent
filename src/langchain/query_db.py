from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from langchain_ollama import ChatOllama
from lazy_embeddings import LazyEmbeddings

start_time = time.time()

vector_store = Chroma(
    collection_name="documents",
    persist_directory="./chroma_langchain_db"
)
retriever = vector_store.as_retriever()

embeddings = LazyEmbeddings()

query = "crea_area_assistenza"
query_vector = embeddings.embed_query(query)

results = vector_store.similarity_search_by_vector(query_vector, k=1)

print("\n========= DEBUG: DOCUMENTI RECUPERATI =========\n")
for i, doc in enumerate(results, 1):
    print(f"[Documento {i}]\n{doc.page_content}\n")

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

print("EMBEDDINGS TIME = ", time.time() - start_time)

docs_content = "\n\n".join(doc.page_content for doc in results)
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


print("TOTAL TIME = ", time.time() - start_time)