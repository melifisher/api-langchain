import requests
import json

# 1. Realizar la búsqueda
while True:
    query = input("Ingresa tu pregunta (o 'q' para salir): ")
    
    if query.lower() == 'q':
        break
        
    search_url = 'http://localhost:5000/api/search'
    search_data = {"query": query, "k": 3}

    search_response = requests.post(search_url, json=search_data)
    search_results = search_response.json()

    # Imprimir la respuesta para el usuario
    print("\nPregunta:", search_results["query"])
    print("\nRespuesta:")
    print(search_results["response"])
    print("\n---")

    
# 2. Solicitar feedback al usuario (1-5 estrellas)
# user_rating = input("¿Cómo calificarías esta respuesta? (1-5): ")
# try:
#     # Convertir a entero y validar
#     rating = int(user_rating)
#     if rating < 1 or rating > 5:
#         print("Por favor, ingresa un número entre 1 y 5")
#         exit()
# except ValueError:
#     print("Por favor, ingresa un número válido")
#     exit()

# # 3. Enviar el feedback
# feedback_url = 'http://localhost:5000/api/feedback'
# feedback_data = {
#     "query": search_results["query"],
#     "response": search_results["response"],
#     "rating": rating,
#     "contexts": search_results["results"]  # Pasamos los contextos que se usaron
# }

# feedback_response = requests.post(feedback_url, json=feedback_data)
# feedback_result = feedback_response.json()

# print("\nFeedback enviado:", feedback_result["message"])

# # Obtener estadísticas del sistema de feedback
# stats_url = 'http://localhost:5000/api/feedback/stats'
# stats_response = requests.get(stats_url)
# stats_results = stats_response.json()

# print("\nEstadísticas de feedback:")
# print(f"Total de calificaciones: {stats_results['stats']['total_feedback']}")
# print(f"Calificación promedio: {stats_results['stats']['average_rating']}")
# print("Distribución de calificaciones:")
# for rating, count in stats_results['stats']['rating_distribution'].items():
#     print(f"  {rating} estrellas: {count} respuestas")

# response = requests.post('http://localhost:5000/api/initialize')
# print(response.json())
