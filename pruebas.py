
import os
from contextcache import ContextCache

from main4 import RAGSystem

cache= ContextCache()

#print(cache.get_history("user123",10))

#print(cache.get_ultimate_question("user123"))

historial = cache.get_history("user123",10)

ultimatequestion = cache.get_ultimate_question("user123")

anteriorrespuesta = cache.get_ultimate_answerfull("user123")

#anteriorrespuesta=""
#cache.clear_user_history("user123")

CONFIG = {
    "FILE_PATH": os.getenv("RAG_FILE_PATH", "datoscompletos.txt"),
    "PERSIST_DIR": os.getenv("RAG_PERSIST_DIR", "chroma_db"),
    "CHUNK_SIZE": int(os.getenv("RAG_CHUNK_SIZE", "500")),
    "CHUNK_OVERLAP": int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
    "FEEDBACK_DB": os.getenv("FEEDBACK_DB", "feedback.db")
}

# Initialize RAG system
rag_system = RAGSystem(
    file_path=CONFIG["FILE_PATH"],
    persist_dir=CONFIG["PERSIST_DIR"],
    chunk_size=CONFIG["CHUNK_SIZE"],
    chunk_overlap=CONFIG["CHUNK_OVERLAP"],
    feedback_db_path=CONFIG["FEEDBACK_DB"]
)
print (RAGSystem.in_context( None, ultimatequestion , " y cuanto es la multa en zona escolar?", anteriorrespuesta))