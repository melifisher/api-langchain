import os
import time
import hashlib
import logging
import sqlite3
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
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

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

class FeedbackDB:
    """Class to handle feedback storage and retrieval"""
    def __init__(self, db_path="feedback.db"):
        """Initialize the feedback database"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create the feedback table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for storing feedback
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            rating INTEGER NOT NULL,
            contexts TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Feedback database initialized at {self.db_path}")
    
    def save_feedback(self, query: str, response: str, rating: int, contexts: List[Dict]):
        """Save feedback for a response"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save the feedback
        cursor.execute(
            "INSERT INTO feedback (query, response, rating, contexts) VALUES (?, ?, ?, ?)",
            (query, response, rating, json.dumps(contexts))
        )
        
        conn.commit()
        conn.close()
        logger.info(f"Saved feedback (rating: {rating}) for query: {query[:50]}...")
        return True
    
    def get_similar_feedback(self, query: str, limit=5):
        """
        Get feedback for similar queries
        Simple implementation using substring matching
        In a production system, this would use embeddings similarity
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple approach: look for queries containing similar keywords
        words = set(query.lower().split())
        results = []
        
        cursor.execute("SELECT query, response, rating, contexts FROM feedback ORDER BY timestamp DESC")
        all_feedback = cursor.fetchall()
        
        for fb_query, fb_response, fb_rating, fb_contexts in all_feedback:
            fb_words = set(fb_query.lower().split())
            # Calculate basic similarity (intersection of words)
            similarity = len(words.intersection(fb_words)) / max(len(words), 1)
            if similarity > 0.3:  # Arbitrary threshold
                results.append({
                    "query": fb_query,
                    "response": fb_response,
                    "rating": fb_rating,
                    "contexts": json.loads(fb_contexts),
                    "similarity": similarity
                })
                if len(results) >= limit:
                    break
        
        conn.close()
        return results
    
    def get_feedback_stats(self):
        """Get statistics about the feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM feedback")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(rating) as avg_rating FROM feedback")
        avg_rating = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT rating, COUNT(*) as count 
            FROM feedback 
            GROUP BY rating 
            ORDER BY rating DESC
        """)
        rating_distribution = {rating: count for rating, count in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_feedback": total,
            "average_rating": round(avg_rating, 2),
            "rating_distribution": rating_distribution
        }

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
    def in_context(self,prev_question: str, new_question: str,answerfull :str ,formatted_history : str) -> bool:
       
        mensajes = [
            SystemMessage(
                content="""Eres un clasificador de una IA para saber si la pregunta sigue el contexto o no. Determina si la nueva pregunta podría estar relacionada con la pregunta anterior o
                si es un nuevo tema. Tambien podrias basarte en si la respuesta a la nueva pregunta esta contenida en la respuesta anterior.
                Responde únicamente con "Relacionado" o "Nuevo tema"."""
                        ),
            HumanMessage(
                content=f"""
                Pregunta anterior o principal:\n“{prev_question}”\nNueva pregunta:\n“{new_question}”
                
                Respuesta anterior para verificar si sigue en el contexto: \n{ answerfull }
                Instrucciones adicionales:
                1. Responde basado en la Nueva pregunta , solo si la respuesta correcta esta ahi
                2. te pase la Respuesta anterior para que veas si esta en el contexto tambien
                3. Si ves que la respuesta a la nueva pregunta no esta entre las respuesta anterior ,considera que no esta en el contexto
                4. si es necesario Basa tu respuesta en este contexto previo, si se aleja mucho tomalo como Nuevo tema 
        
                 Contexto previo:\n{formatted_history}
        
                
                """
                # Instrucciones adicionales:
                # 1. Responde basado en la Nueva pregunta , solo si la respuesta correcta esta ahi
                # 2. te pase la Respuesta anterior para que veas si esta en el contexto tambien
                # 3. Si ves que la respuesta a la nueva pregunta no esta entre las respuesta anterior ,considera que no esta en el contexto
                #                o si la respuesta a la nueva pregunta está contenida en la respuesta anterior.

            )
        ]

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        respuesta = llm.invoke(mensajes)

        print("→ Respuesta del modelo:", respuesta.content)
        print(f"""Pregunta anterior o principal:\n“{prev_question}”\nNueva pregunta:\n“{new_question}”""")

        return "relacionado" in respuesta.content.lower()
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
    
    def proces_data_result_openIA_Continue_Context(self, results: str, query: str, oldquery :str, formatted_history : str):
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
    
    def proces_data_result_openIA(self, results: List[Dict[str, Any]], query: str):
        """Process the search results with OpenAI, incorporating feedback."""
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
        combined_content = "\n\n".join([item.get("content", "") for item in results])

        # Crear el prompt con todos los contenidos juntos y el feedback
        prompt = f"""
        Actua como un experto en leyes de tránsito con base en el siguiente texto, el cual es sacado de documentos de leyes de transito de Bolivia:

        \"\"\"{combined_content}\"\"\"

        Responde a la siguiente pregunta del usuario:

        \"{query}\"
         
        Instrucciones adicionales:
        1. Responde con hechos basados en la información proporcionada
        2. Mantén un tono profesional
        3. Limita tu respuesta a 150 palabras máximo
        4. Si tiene Articulo, nombralo
        5. Solo responde a la pregunta
        
        {feedback_guidance}
        """

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
        
        print(data)
        
        if not data or "query" not in data:
            return jsonify({
                "status": "error",
                "message": "Query parameter is required"
            }), 400
            
        query = data["query"]
        k = data.get("k", 5) 
        old_question = data.get("oldquestion", "").strip()
        old_response_full = data.get("oldresponsefull", "").strip()
        hay_contexto = bool(old_question) and bool(old_response_full)
        
        history = data.get("summaries", [])
        formatted_history = "\n \n".join(history)


        reemplazador = filtro_palabras()
        cache = ContextCache()
        newcontext=True
        queryFiltrado = reemplazador.reemplazar_palabras(query)   
      
        # Ensure RAG system is initialized
        if not rag_system.vectorstore:
            rag_system.initialize()
        
        # Generate unique identifier for this response
        response_id = hashlib.md5(f"{query}-{time.time()}".encode()).hexdigest()
        
        # Perform search
        
        # Combina todos los contenidos en un solo string
        #combined_content = "\n\n".join([item.get("content", "") for item in results])
        #cache.add_entry("user123", query , combined_content,response_content)


        
        incontext = rag_system.in_context(old_question,queryFiltrado , old_response_full,formatted_history)
        
        if  (hay_contexto) and incontext: 
            newquestioon = old_question + queryFiltrado
            results = rag_system.search(newquestioon, k=k+3) # nuevos resutados de la pregunta principal conbinada con la nueva pregunta relacionada
            combined_content = "\n\n".join([item.get("content", "") for item in results])
            response_content = rag_system.proces_data_result_openIA_Continue_Context( combined_content , old_question , queryFiltrado,formatted_history)
            logger.info(f"In Cache Answer ")
            results =[]
            newcontext=False
        else:
            #cache.clear_user_history("user123") #elimina la cache del contexto actual
            results = rag_system.search(queryFiltrado, k=k)
            response_content = rag_system.proces_data_result_openIA(results, query)
            logger.info(f"From VectorDB Answer ")
            #combined_content = "\n\n".join([item.get("content", "") for item in results])
            #cache.add_entry("user123", query , combined_content,response_content) #inicia un nuevo contexto de pregunta 
            newcontext=True
        
        
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": results,
            "result_count": len(results),
            "response": response_content,
            "response_id": response_id,
            "isnewcontext":newcontext,
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

if __name__ == '__main__':
    # Initialize before starting the server
    rag_system.initialize()
    
    # Get port from environment variable or use 5000 as default
    port = int(os.getenv("PORT", "5000"))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")