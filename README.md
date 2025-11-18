# 🚦 RAG API – Sistema de Recuperación Aumentada con Contexto, Feedback y Clasificación Semántica

Esta API implementa un **sistema RAG (Retrieval-Augmented Generation)** especializado en **leyes de tránsito de Bolivia**, con características avanzadas como:

* Vector store persistente con **ChromaDB**
* Detección de cambios en el archivo fuente con **hash MD5**
* Clasificación automática de continuidad de contexto mediante **LLM**
* Recuperación semántica mediante **embeddings OpenAI**
* Sistema de **feedback con aprendizaje continuo**
* Compatibilidad nativa con **Flutter** mediante CORS
* Manejo de conversaciones basadas en contexto y multiconsulta

La API es construida en **Flask + LangChain + OpenAI + ChromaDB**.

---

# 📁 Estructura general

```
project/
 ├─ app.py                 # API principal (Flask)
 ├─ datoscompletos.txt     # Base textual de leyes (editable)
 ├─ filtro.py              # Normalización de palabras
 ├─ contextcache.py        # Manejo de historial
 ├─ feedback.py            # DB SQLite para feedback
 ├─ chroma_db/             # Embeddings persistentes
 ├─ feedback.db            # Base de datos SQLite
 └─ .env                   # Variables de entorno
```

---

# ⚙️ 1. Configuración previa

## 1.1 Variables de entorno (`.env`)

Crea un archivo:

```
OPENAI_API_KEY=tu_api_key
RAG_FILE_PATH=datoscompletos.txt
RAG_PERSIST_DIR=chroma_db
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
FEEDBACK_DB=feedback.db
```

---

# 🚀 2. Inicializar el sistema

Antes de hacer consultas, debes inicializar la base vectorial:

### Endpoint

```
POST /api/initialize
```

### Respuesta esperada

```json
{
  "status": "success",
  "message": "RAG system initialized successfully",
  "time_taken": "3.42 seconds"
}
```

Esto:

* Carga el archivo `datoscompletos.txt`
* Lo divide en chunks
* Genera embeddings con OpenAI
* Guarda todo en `chroma_db/`

Si ya existe un hash igual, no regenera embeddings.

---

# 🔍 3. Búsqueda y generación de respuesta

### Endpoint

```
POST /api/search
```

### Body requerido

```json
{
  "query": "¿Cuál es la sanción por no usar casco?",
  "k": 5,
  "oldquestion": "",
  "oldresponsefull": "",
  "summaries": []
}
```

## Campos:

| Campo             | Descripción                             |
| ----------------- | --------------------------------------- |
| `query`           | Pregunta del usuario                    |
| `k`               | Cantidad de documentos a recuperar      |
| `oldquestion`     | Pregunta anterior (si hay conversación) |
| `oldresponsefull` | Respuesta anterior completa             |
| `summaries`       | Historial reducido de la conversación   |

La API automáticamente:

* filtra palabras (`filtro_palabras`)
* detecta si es una **continuación del contexto**
* usa embeddings y clasificación LLM

---

## Respuesta típica

```json
{
  "status": "success",
  "query": "¿Cuál es la sanción por no usar casco?",
  "results": [...],
  "result_count": 5,
  "response": "Según el Artículo 92...",
  "response_id": "9dc3a1f7b1...",
  "isnewcontext": true
}
```

---

# 4. Modo "Continuación de contexto"

Cuando se envía:

```json
"oldquestion": "¿Qué documentos debo llevar?",
"oldresponsefull": "Debes portar licencia...",
```

La API detecta si la nueva pregunta **está relacionada** usando un clasificador LLM interno.

Si está en contexto:

✔ amplía la consulta combinando ambas preguntas
✔ aumenta `k` automáticamente
✔ genera una respuesta basada en contexto histórico

---

# 5. Enviar feedback de usuario

La API aprende usando un sistema de retroalimentación que se guarda en SQLite (`feedback.db`).

### Endpoint

```
POST /api/feedback
```

### Body

```json
{
  "query": "¿Cuál es la sanción por no usar casco?",
  "response": "Según el Artículo 92...",
  "rating": 5,
  "contexts": ["uso de casco", "seguridad vial"]
}
```

### Respuesta

```json
{
  "status": "success",
  "message": "Feedback (rating: 5) saved successfully"
}
```

El feedback es usado para:

* Ajustar estilo de respuesta
* Aprender patrones positivos/negativos
* Mejorar la precisión futura

---

# 6. Estadísticas de feedback

### Endpoint

```
GET /api/feedback/stats
```

Respuesta:

```json
{
  "status": "success",
  "stats": {
    "total_feedback": 12,
    "avg_rating": 4.6,
    "most_common_topics": ["casco", "licencia"]
  }
}
```

---

# 7. Health Check

### Endpoint

```
GET /api/health
```

Respuesta:

```json
{
  "status": "ok",
  "service": "RAG API",
  "config": {
    "file_path": "datoscompletos.txt",
    "chunk_size": 500
  },
  "feedback_stats": {...}
}
```

---

# 8. Ejecución local

Instalar dependencias:

```
pip install -r Requirements.txt
```

Levantar API:

```
python main4.py
```

---
# 9. Tecnologías utilizadas

* **Flask** – API backend
* **OpenAI GPT-3.5** – generación + clasificación
* **OpenAI Embeddings** – similitud semántica
* **LangChain** – pipelines RAG
* **ChromaDB** – vector store persistente
* **SQLite** – feedback learning
* **scikit-learn** – similitud coseno
* **Flask-CORS** – soporte para Flutter
