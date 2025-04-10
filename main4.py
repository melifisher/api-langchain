import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from filtro import filtro_palabras

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self, 
        file_path: str, 
        persist_dir: str = "chroma_db",
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ):
        """Initialize the RAG system with configurable parameters."""
        self.file_path = file_path
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        
        # Ensure persist directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
    def get_file_hash(self) -> str:
        """Calculate MD5 hash of input file to detect changes."""
        with open(self.file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_documents(self) -> List[Document]:
        """Load and split documents from the source file."""
        start_time = time.time()
        loader = TextLoader(self.file_path, encoding="utf-8")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        logger.info(f"Loaded {len(texts)} text chunks in {time.time() - start_time:.2f} seconds")
        return texts
    
    def initialize(self) -> None:
        """Initialize or load the vector database based on file changes."""
        current_hash = self.get_file_hash()
        hash_file_path = os.path.join(self.persist_dir, "hash.txt")
        
        # Check if database exists and if source file has changed
        if os.path.exists(self.persist_dir) and os.path.exists(hash_file_path):
            with open(hash_file_path, "r") as f:
                saved_hash = f.read().strip()
            
            if saved_hash == current_hash:
                logger.info("✅ Using existing embeddings (no changes in source file)")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                return
            
            logger.info("🔄 Source file changed, updating embeddings")
        else:
            logger.info("🆕 First run, creating new embeddings")
        
        # Create or update embeddings
        texts = self.load_documents()
        start_time = time.time()
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        # Save the current hash
        with open(hash_file_path, "w") as f:
            f.write(current_hash)
            
        logger.info(f"Embeddings created in {time.time() - start_time:.2f} seconds")
    
    def search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """Perform similarity search with the given query."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")
            
        start_time = time.time()
        docs = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Search completed in {time.time() - start_time:.2f} seconds")
        
        # Convert LangChain documents to serializable format
        results = []
        for i, doc in enumerate(docs, start=1):
            results.append({
                "id": i,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return results
    
    def proces_data_result_openIA(self ,results: List[Dict[str, Any]], query: str):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        respuestas = []

        # Combina todos los contenidos en un solo string
        combined_content = "\n\n".join([item.get("content", "") for item in results])

        # Crear el prompt con todos los contenidos juntos
        prompt = f"""
        Eres un experto en leyes de tránsito. con base en el siguiente texto:

        \"\"\"{combined_content}\"\"\"

        Responde a la siguiente pregunta del usuario, solo dame  la respuesta:

        \"{query}\"
         Instrucciones adicionales:
        1. Responde solo con hechos basados en la información proporcionada
        2. Mantén un tono profesional
        3. Limita tu respuesta a 150 palabras máximo
        4. Si tiene Articulo en parentesis nombra al articulo
        
        """
        #logger.info(f"Pront enviado : {prompt} ")
        respuesta = llm.invoke(prompt)
        respuestas.append(respuesta.content)
        #logger.info(respuestas) #muestra el resultado de la espuesta
        return respuestas
    

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter integration

# Default configuration
CONFIG = {
    "FILE_PATH": os.getenv("RAG_FILE_PATH", "datoscompletos.txt"),
    "PERSIST_DIR": os.getenv("RAG_PERSIST_DIR", "chroma_db"),
    "CHUNK_SIZE": int(os.getenv("RAG_CHUNK_SIZE", "500")),
    "CHUNK_OVERLAP": int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
}

# Initialize RAG system
rag_system = RAGSystem(
    file_path=CONFIG["FILE_PATH"],
    persist_dir=CONFIG["PERSIST_DIR"],
    chunk_size=CONFIG["CHUNK_SIZE"],
    chunk_overlap=CONFIG["CHUNK_OVERLAP"]
)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({
        "status": "ok",
        "service": "RAG API",
        "config": {
            "file_path": CONFIG["FILE_PATH"],
            "chunk_size": CONFIG["CHUNK_SIZE"],
            "chunk_overlap": CONFIG["CHUNK_OVERLAP"]
        }
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_api():
    """API endpoint to initialize or refresh the RAG system."""
    try:
        start_time = time.time()
        rag_system.initialize()
        return jsonify({
            "status": "success",
            "message": "RAG system initialized successfully",
            "time_taken": f"{time.time() - start_time:.2f} seconds"
        })
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search_api():
    """API endpoint to search the RAG system."""
    try:
        data = request.get_json()
        
        
        if not data or "query" not in data:
            return jsonify({
                "status": "error",
                "message": "Query parameter is required"
            }), 400
            
        query = data["query"]
        k = data.get("k", 2)  # Number of results, default is 2

         
        reemplazador  = filtro_palabras()
        query = reemplazador.reemplazar_palabras(query)   
      
        
        # Ensure RAG system is initialized
        if not rag_system.vectorstore:
            rag_system.initialize()
        
        # Perform search
        results = rag_system.search(query, k=k)
        resultsprocess = rag_system.proces_data_result_openIA(results,query)
        return jsonify({
            "status": "success",
            "query": query,
            "results": results,
            "result_count": len(results),
            "processIA":resultsprocess,
        })
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/update', methods=['POST'])
def update_data():
    """API endpoint to update the data file."""
    try:
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file part in the request"
            }), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400
            
        # Save the uploaded file
        file.save(CONFIG["FILE_PATH"])
        logger.info(f"File saved to {CONFIG['FILE_PATH']}")
        
        # Force reinitialization by clearing the vectorstore
        rag_system.vectorstore = None
        
        # Reinitialize the RAG system
        rag_system.initialize()
        
        return jsonify({
            "status": "success",
            "message": "Data file updated and RAG system reinitialized"
        })
    except Exception as e:
        logger.error(f"Update error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)

        }), 500
if __name__ == '__main__':
    # Initialize before starting the server
    rag_system.initialize()
    
    # Get port from environment variable or use 5000 as default
    port = int(os.getenv("PORT", "5000"))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")