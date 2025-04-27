# contextcache.py
from typing import Dict, List, Optional, Any
import time

class ContextCache:
    """Maneja el contexto de conversaciones para dar continuidad a las respuestas"""
    
    def __init__(self, max_history: int = 5, expiry_time: int = 3600):
        """
        Inicializa el caché de contexto
        
        Args:
            max_history: Número máximo de entradas de historial por usuario
            expiry_time: Tiempo en segundos antes de que una entrada expire
        """
        self.contexts = {}  # Diccionario para almacenar contextos por usuario
        self.max_history = max_history
        self.expiry_time = expiry_time  # 1 hora por defecto
    
    def add_entry(self, user_id: str, query: str, contexts: str, response: str) -> None:
        """
        Añade una nueva entrada al contexto del usuario
        
        Args:
            user_id: Identificador único del usuario
            query: Pregunta realizada
            contexts: Contextos utilizados para responder
            response: Respuesta generada
        """
        timestamp = time.time()
        
        if user_id not in self.contexts:
            self.contexts[user_id] = []
        
        # Añadir nueva entrada
        self.contexts[user_id].append({
            "query": query,
            "contexts": contexts,
            "response": response,
            "timestamp": timestamp
        })
        
        # Mantener solo las últimas max_history entradas
        if len(self.contexts[user_id]) > self.max_history:
            self.contexts[user_id].pop(0)  # Eliminar la más antigua
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene el historial completo de conversación para un usuario
        
        Args:
            user_id: Identificador único del usuario
        
        Returns:
            Lista de entradas de conversación ordenadas cronológicamente
        """
        if user_id not in self.contexts:
            return []
        
        # Filtrar entradas expiradas
        current_time = time.time()
        valid_entries = [
            entry for entry in self.contexts[user_id]
            if current_time - entry["timestamp"] <= self.expiry_time
        ]
        
        # Actualizar el contexto sin entradas expiradas
        self.contexts[user_id] = valid_entries
        
        return valid_entries
    
    def get_recent_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene la entrada de contexto más reciente para un usuario
        
        Args:
            user_id: Identificador único del usuario
        
        Returns:
            La entrada más reciente o None si no hay entradas
        """
        history = self.get_conversation_history(user_id)
        return history[-1] if history else None
    
    def generate_conversation_summary(self, user_id: str) -> str:
        """
        Genera un resumen de la conversación para usar en prompts
        
        Args:
            user_id: Identificador único del usuario
            
        Returns:
            Resumen formateado de la conversación
        """
        history = self.get_conversation_history(user_id)
        if not history:
            return ""
        
        summary = "Historial de conversación reciente:\n\n"
        for i, entry in enumerate(history, 1):
            summary += f"Pregunta {i}: {entry['query']}\n"
            summary += f"Respuesta {i}: {entry['response']}\n\n"
        
        return summary
    
    def clear_user_history(self, user_id: str) -> None:
        """Elimina todo el historial de un usuario"""
        if user_id in self.contexts:
            del self.contexts[user_id]