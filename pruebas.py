
from contextcache import ContextCache

from main4 import RAGSystem

cache= ContextCache()

#print(cache.get_history("user123",10))

#print(cache.get_ultimate_question("user123"))

historial = cache.get_history("user123",10)

ultimatequestion = cache.get_ultimate_question("user123")

anteriorrespuesta = cache.get_ultimate_answerfull("user123")

#anteriorrespuesta=""
cache.clear_user_history("user123")
print (RAGSystem.in_context( None , ultimatequestion , " y si me doy a la fuga  ?", anteriorrespuesta))