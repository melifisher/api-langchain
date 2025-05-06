import sqlite3
import logging
import json
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class FeedbackDB:
    def __init__(self, db_path="feedback.db"):
        """Initialize the feedback database"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO feedback (query, response, rating, contexts) VALUES (?, ?, ?, ?)",
            (query, response, rating, json.dumps(contexts))
        )
        
        conn.commit()
        conn.close()
        logger.info(f"Saved feedback (rating: {rating}) for query: {query[:50]}...")
        return True
    
    def get_similar_feedback(self, query: str, limit=5):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obtener todos los feedbacks ordenados por tiempo
        cursor.execute("SELECT query, response, rating, contexts FROM feedback ORDER BY timestamp DESC")
        all_feedback = cursor.fetchall()
        conn.close()
        
        if not all_feedback:
            return []
        
        # Usar embeddings para encontrar similitudes más precisas
        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_query(query)
        
        results = []
        for fb_query, fb_response, fb_rating, fb_contexts in all_feedback:
            try:
                # Calcular similitud por embeddings
                fb_embedding = embeddings.embed_query(fb_query)
                similarity = cosine_similarity([query_embedding], [fb_embedding])[0][0]
                
                if similarity > 0.6:  # Umbral de similitud más alto
                    results.append({
                        "query": fb_query,
                        "response": fb_response,
                        "rating": fb_rating,
                        "contexts": json.loads(fb_contexts),
                        "similarity": float(similarity)
                    })
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.error(f"Error calculating similarity: {str(e)}")
        
        # Ordenar por similitud
        results.sort(key=lambda x: x["similarity"], reverse=True)
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

    def extract_feedback_patterns(self):
        """
        Extrae patrones de aprendizaje de todo el feedback existente
        para identificar características de respuestas bien valoradas
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obtener todo el feedback
        cursor.execute("SELECT query, response, rating FROM feedback")
        all_feedback = cursor.fetchall()
        conn.close()
        
        if not all_feedback:
            return {
                "positive_patterns": [],
                "negative_patterns": []
            }
        
        # Clasificar por rating
        high_rated = [item for item in all_feedback if item[2] >= 4]
        low_rated = [item for item in all_feedback if item[2] <= 2]
        
        # Extraer patrones con LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        positive_patterns = []
        negative_patterns = []
        
        if high_rated:
            # Seleccionar máximo 10 ejemplos para el análisis
            samples = high_rated[:10] if len(high_rated) > 10 else high_rated
            samples_text = "\n\n".join([f"Query: {q}\nResponse: {r}\nRating: {rating}" 
                                        for q, r, rating in samples])
            
            prompt = f"""
            Analiza las siguientes respuestas bien valoradas (calificación 4-5) por usuarios:
            
            {samples_text}
            
            Identifica 3-5 patrones o características comunes que hacen que estas respuestas sean efectivas.
            Responde únicamente con una lista numerada de patrones, sin introducción ni conclusión.
            """
            
            response = llm.invoke(prompt)
            if response:
                positive_patterns = [line.strip() for line in response.content.split("\n") 
                                    if line.strip() and not line.strip().isdigit()]
        
        if low_rated:
            # Mismo proceso para respuestas mal valoradas
            samples = low_rated[:10] if len(low_rated) > 10 else low_rated
            samples_text = "\n\n".join([f"Query: {q}\nResponse: {r}\nRating: {rating}" 
                                    for q, r, rating in samples])
            
            prompt = f"""
            Analiza las siguientes respuestas mal valoradas (calificación 1-2) por usuarios:
            
            {samples_text}
            
            Identifica 3-5 patrones o problemas comunes que hacen que estas respuestas sean inefectivas.
            Responde únicamente con una lista numerada de patrones, sin introducción ni conclusión.
            """
            
            response = llm.invoke(prompt)
            if response:
                negative_patterns = [line.strip() for line in response.content.split("\n") 
                                    if line.strip() and not line.strip().isdigit()]
        
        return {
            "positive_patterns": positive_patterns,
            "negative_patterns": negative_patterns
        }
