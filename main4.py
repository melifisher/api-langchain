import os
import time
import hashlib
import logging
import json
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
from contextcache import ContextCache
from feedback import FeedbackDB
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

import chromadb
from chromadb.config import Settings
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
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
        chunk_overlap: int = 50,
        feedback_db_path: str = "feedback.db"
    ):
        """Initialize the RAG system with configurable parameters."""
        self.file_path = file_path
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.feedback_db = FeedbackDB(feedback_db_path)
        
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

    def in_context(self, prev_question: str, new_question: str, answerfull: str, formatted_history: str) -> bool:
        """
        Determina si una nueva pregunta está en el mismo contexto que la anterior
        utilizando un enfoque híbrido de embeddings y LLM
        
        Args:
            prev_question: Pregunta anterior en la conversación
            new_question: Nueva pregunta del usuario
            answerfull: Respuesta completa a la pregunta anterior
            formatted_history: Historial formateado de la conversación
            
        Returns:
            bool: True si la nueva pregunta está en contexto
        """
        
        # 2. Para casos intermedios, usamos el LLM para un análisis semántico profundo
        # Sistema de votación: combinamos resultados de embeddings y LLM
        votes = 0
        
        
        # Análisis semántico con LLM para casos difíciles
        mensajes = [
            SystemMessage(
                content="""Eres un clasificador experto en determinar si una nueva pregunta está relacionada 
                con la conversación anterior. Tu trabajo es analizar si la nueva pregunta:
                1. Es una continuación natural del tema anterior
                2. Solicita clarificación o detalles adicionales sobre lo discutido
                3. Se refiere a conceptos mencionados previamente
                4. Está completamente fuera de contexto
                
                Responde únicamente con "Relacionado" o "Nuevo tema"."""
                        ),
            HumanMessage(
                content=f"""
                Analiza cuidadosamente si la nueva pregunta está relacionada con el contexto previo.
                
                Pregunta anterior: "{prev_question}"
                Nueva pregunta: "{new_question}"
                
                Respuesta anterior: 
                {answerfull}
                
                Historial de conversación previo:
                {formatted_history}
                
                ¿La nueva pregunta está relacionada con el contexto actual o es un tema nuevo?
                """
            )
        ]

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        respuesta = llm.invoke(mensajes)
        
        logger.info(f"Pregunta anterior: {prev_question}. Nueva pregunta: {new_question}")
        logger.info(f"Respuesta del clasificador: {respuesta.content}")
        
        if "relacionado" in respuesta.content.lower():
            votes += 2  # El LLM tiene más peso en la decisión
        
        # Sistema de votación final
        # Si embeddings y LLM concuerdan, o si el LLM está muy seguro (2 votos)
        return votes >= 2
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula la similitud coseno entre dos textos usando embeddings.
        
        """
        model_name: str = "text-embedding-3-small",
        
        embedder = self.embeddings
        emb1 = embedder.embed_query(text1)
        emb2 = embedder.embed_query(text2)
        return cosine_similarity([emb1], [emb2])[0][0]

    def in_context_embeddings(self, prev_question: str, new_question: str, answerfull: str) -> bool:
        """
        Determina si una nueva pregunta está en contexto usando embeddings.
        
        Args:
            prev_question: Pregunta anterior en la conversación
            new_question: Nueva pregunta a evaluar
            answerfull: Respuesta completa relacionada
            
        Returns:
            bool: True si la nueva pregunta está en contexto
        """
        threshold: float = 0.8,
        # Calcula similitudes
        sim_questions = self.get_similarity(prev_question, new_question)
        sim_new_to_answer = self.get_similarity(new_question, answerfull)
        
        # Puntuación combinada ponderada
        weights = (0.6, 0.4)
        combined_score = (weights[0] * sim_questions + 
                         weights[1] * sim_new_to_answer)
        
        return combined_score >= threshold
    
    def process_data_result_openIA_Continue_Context(self, results: str, query: str, oldquery :str, formatted_history : str):
        """Process the search results with OpenAI, incorporating feedback and Continue Context Using Old Anaswers of question Initial (lee fisher :U)."""
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        # Get similar feedback to influence the response
        similar_feedback = self.feedback_db.get_similar_feedback(query)
        feedback_guidance = ""
        
        if similar_feedback:
            # Extract learnings from previous feedback
            high_rated_examples = [f for f in similar_feedback if f["rating"] >= 4]
            low_rated_examples = [f for f in similar_feedback if f["rating"] <= 2]
            
            if high_rated_examples or low_rated_examples:
                feedback_guidance = "Basado en feedback previo de usuarios:\n"
                
                if high_rated_examples:
                    feedback_guidance += "Los usuarios prefieren respuestas que:\n"
                    for ex in high_rated_examples[:2]:
                        feedback_guidance += f"- Sean similares a: '{ex['response'][:100]}...'\n"
                
                if low_rated_examples:
                    feedback_guidance += "Los usuarios NO prefieren respuestas que:\n"
                    for ex in low_rated_examples[:2]:
                        feedback_guidance += f"- Sean similares a: '{ex['response'][:100]}...'\n"
        
        # Combina todos los contenidos en un solo string
        combined_content =  results

        # Crear el prompt con todos los contenidos juntos y el feedback
        prompt = f"""
        Actua como un experto en leyes de tránsito con base en el siguiente texto, el cual es sacado de documentos de leyes de transito de Bolivia:

        \"\"\"{combined_content}\"\"\"
        
        Responde a la siguiente pregunta del usuario:

        \"{query}\"

        Tomando en cuenta que esta era la anterior pregunta:

        \"{oldquery}\"
         
        Instrucciones adicionales:
        1. Responde con hechos basados en la información proporcionada
        2. Mantén un tono profesional
        3. Limita tu respuesta a 150 palabras máximo
        4. Si tiene Articulo, nombralo
        5. Solo responde a la pregunta del usuario
        6. La Pregunta es la continuacion al contexto de la pregunta principal
        7. si es necesario Basa tu respuesta en este contexto previo 
        
        Contexto previo:\n{formatted_history}
        
        {feedback_guidance}
        """

        respuesta = llm.invoke(prompt)
        return respuesta.content
    
    def process_data_result_openIA(self, results: List[Dict[str, Any]], query: str, conversation_history: str):
        """
        Process the search results with OpenAI, incorporating feedback and continual learning
        
        Args:
            results: Resultados de la búsqueda vectorial
            query: Consulta del usuario
            user_id: ID del usuario para tracking
        """
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        # Analizar feedback similar
        similar_feedback = self.feedback_db.get_similar_feedback(query)
        
        learning_patterns = self.feedback_db.extract_feedback_patterns()
        
        feedback_guidance = ""
        
        if similar_feedback:
            high_rated_examples = [f for f in similar_feedback if f["rating"] >= 4]
            low_rated_examples = [f for f in similar_feedback if f["rating"] <= 2]
            
            if high_rated_examples or low_rated_examples:
                feedback_guidance = "Basado en feedback previo de usuarios:\n"
                
                if high_rated_examples:
                    feedback_guidance += "Los usuarios prefieren respuestas que:\n"
                    for ex in high_rated_examples[:2]:
                        feedback_guidance += f"- Sean similares a: '{ex['response'][:100]}...'\n"
                
                if low_rated_examples:
                    feedback_guidance += "Los usuarios NO prefieren respuestas que:\n"
                    for ex in low_rated_examples[:2]:
                        feedback_guidance += f"- Sean similares a: '{ex['response'][:100]}...'\n"
        
        if learning_patterns["positive_patterns"]:
            feedback_guidance += "\nPatrones efectivos identificados:\n"
            for i, pattern in enumerate(learning_patterns["positive_patterns"][:3], 1):
                feedback_guidance += f"- {pattern}\n"
        
        if learning_patterns["negative_patterns"]:
            feedback_guidance += "\nPatrones a evitar:\n"
            for i, pattern in enumerate(learning_patterns["negative_patterns"][:3], 1):
                feedback_guidance += f"- {pattern}\n"
        
        # Combinar contenidos de búsqueda
        combined_content = "\n\n".join([item.get("content", "") for item in results])
        
        # Crear prompt
        prompt = f"""
        Actúa como un experto en leyes de tránsito con base en el siguiente texto, el cual es sacado de documentos de leyes de tránsito de Bolivia:

        \"\"\"{combined_content}\"\"\"

        Responde a la siguiente pregunta del usuario:

        \"{query}\"
        
        Resumen de la conversacion:
        {conversation_history}
        
        Instrucciones adicionales:
        1. Responde con hechos basados en la información proporcionada
        2. Mantén un tono profesional y preciso
        3. Limita tu respuesta a 150 palabras máximo
        4. Si corresponde a un Artículo específico, cítalo explícitamente
        5. Enfócate específicamente en responder la pregunta actual
        6. Si hay información previa relevante en el historial, úsala para contextualizar
        7. Prioriza la precisión y claridad en la respuesta
        
        {feedback_guidance}
        """

        # 8. Generar respuesta
        respuesta = llm.invoke(prompt)
        
        
        return respuesta.content

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter integration

# Default configuration
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

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    # Add feedback stats to health check
    feedback_stats = rag_system.feedback_db.get_feedback_stats()
    
    return jsonify({
        "status": "ok",
        "service": "RAG API",
        "config": {
            "file_path": CONFIG["FILE_PATH"],
            "chunk_size": CONFIG["CHUNK_SIZE"],
            "chunk_overlap": CONFIG["CHUNK_OVERLAP"]
        },
        "feedback_stats": feedback_stats
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
        k = data.get("k", 5) 
        user_id = data.get("user_id", "anonymous")  # Añadir identificación de usuario
        old_question = data.get("oldquestion", "").strip()
        old_response_full = data.get("oldresponsefull", "").strip()
        hay_contexto = bool(old_question) and bool(old_response_full)
        
        history = data.get("summaries", [])
        formatted_history = "\n \n".join(history)

        # Aplicar filtro de normalización de palabras
        reemplazador = filtro_palabras()
        queryFiltrado = reemplazador.reemplazar_palabras(query)   
      
        # Ensure RAG system is initialized
        if not rag_system.vectorstore:
            rag_system.initialize()
        
        # Generate unique identifier for this response
        response_id = hashlib.md5(f"{query}-{time.time()}".encode()).hexdigest()
        
        newcontext = True
        incontext = False
        
        # Verificar si hay contexto previo y si la nueva pregunta sigue ese contexto
        if hay_contexto:
            incontext = rag_system.in_context(old_question, queryFiltrado, old_response_full, formatted_history)
        
        if hay_contexto and incontext: 
            # Si estamos en el mismo contexto, expandimos la pregunta y buscamos más resultados
            logger.info(f"Detectado contexto continuo. Expandiendo búsqueda.")
            combined_query = f"{old_question} {queryFiltrado}"  # Combinar preguntas
            results = rag_system.search(combined_query, k=k+3)
            combined_content = "\n\n".join([item.get("content", "") for item in results])
            
            # Procesar con contexto continuo
            response_content = rag_system.process_data_result_openIA_Continue_Context(
                combined_content, query, old_question, formatted_history)
            newcontext = False
        else:
            # Nueva conversación o cambio de tema
            logger.info(f"Nuevo contexto de conversación o cambio de tema")
            results = rag_system.search(queryFiltrado, k=k)
            response_content = rag_system.process_data_result_openIA(results, query, formatted_history)
            newcontext = True
        
        # Generar datos para la respuesta
        return jsonify({
            "status": "success",
            "query": query,
            "results": results,
            "result_count": len(results),
            "response": response_content,
            "response_id": response_id,
            "isnewcontext": newcontext,
            "user_id": user_id
        })
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """API endpoint to submit feedback on responses."""
    try:
        data = request.get_json()
        
        if not data or "query" not in data or "response" not in data or "rating" not in data or "contexts" not in data:
            return jsonify({
                "status": "error",
                "message": "Required parameters: query, response, rating, contexts"
            }), 400
        
        query = data["query"]
        response = data["response"]
        rating = data["rating"]
        contexts = data["contexts"]
        
        # Validate rating
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({
                "status": "error",
                "message": "Rating must be an integer between 1 and 5"
            }), 400
            
        # Save feedback
        rag_system.feedback_db.save_feedback(query, response, rating, contexts)
        
        return jsonify({
            "status": "success",
            "message": f"Feedback (rating: {rating}) saved successfully"
        })
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """API endpoint to get feedback statistics."""
    try:
        stats = rag_system.feedback_db.get_feedback_stats()
        return jsonify({
            "status": "success",
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Feedback stats error: {str(e)}")
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

# Añadir al main4.py
from apscheduler.schedulers.background import BackgroundScheduler

def analyze_feedback_periodically():
    """
    Función que se ejecuta periódicamente para analizar feedback
    y actualizar los patrones de aprendizaje
    """
    try:
        logger.info("Iniciando análisis periódico de feedback")
        
        # Extraer y procesar patrones de feedback
        patterns = rag_system.feedback_db.extract_feedback_patterns()
        
        with open("learning_patterns.json", "w") as f:
            json.dump(patterns, f, indent=2)
            
        logger.info(f"Análisis completo. Patrones positivos: {len(patterns['positive_patterns'])}, " +
                   f"Patrones negativos: {len(patterns['negative_patterns'])}")
    except Exception as e:
        logger.error(f"Error en análisis periódico: {str(e)}")


# Add this endpoint to your existing Flask application

@app.route('/api/export_fragments/<collection_name>', methods=['GET'])
def export_fragments(collection_name: str):
    """
    Export document fragments with their embeddings for a specific collection.
    This endpoint allows the Flutter app to download fragments for local searching.
    
    Returns:
        A JSON object containing all document fragments, their embeddings, and metadata.
    """
    try:
        # Initialize ChromaDB client with the same settings as your RAG system
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )
        
        # Get the collection
        collection = chroma_client.get_collection(name=collection_name)
        
        # Get all documents with their embeddings
        # Note: 'ids' is always included by default, so we don't need to specify it
        result = collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        
        # Format the response for Flutter
        fragments = []
        for i in range(len(result["documents"])):
            # Convert NumPy array to a regular Python list for JSON serialization
            embedding = result["embeddings"][i]
            if hasattr(embedding, 'tolist'):  # Check if it's a NumPy array
                embedding = embedding.tolist()
                
            fragments.append({
                "id": result["ids"][i],
                "document_id": result["metadatas"][i].get("document_id", result["ids"][i]),
                "text": result["documents"][i],
                "embedding": embedding,
                "metadata": result["metadatas"][i]
            })
        
        return jsonify({
            "status": "success",
            "collection": collection_name,
            "fragments_count": len(fragments),
            "fragments": fragments
        })
    except Exception as e:
        app.logger.error(f"Error exporting collection: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error exporting collection: {str(e)}"
        }), 500

# Optional endpoint to get list of available collections
@app.route('/api/collections', methods=['GET'])
def list_collections():
    """
    List all available collections in the ChromaDB instance.
    
    Returns:
        A JSON object with a list of collection names.
    """
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )
        # chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=CONFIG["PERSIST_DIR"]
        # ))
        
        collection_names = chroma_client.list_collections()
        # collection_names = [c.name for c in collections]
        
        return jsonify({
            "status": "success",
            "collections": collection_names
        })
    except Exception as e:
        app.logger.error(f"Error listing collections: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error listing collections: {str(e)}"
        }), 500


# Iniciar tarea programada
scheduler = BackgroundScheduler()
scheduler.add_job(
    analyze_feedback_periodically, 
    'interval', 
    hours=24  # Ejecutar una vez al día
)
scheduler.start()

# Detener el scheduler al cerrar la aplicación
import atexit
atexit.register(lambda: scheduler.shutdown())


if __name__ == '__main__':
    # Initialize before starting the server
    rag_system.initialize()
    
    # Get port from environment variable or use 5000 as default
    port = int(os.getenv("PORT", "5000"))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")