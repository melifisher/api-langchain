import re

class filtro_palabras:
    reemplazoslist = None
    def __init__(self ):
        """
        Inicializa la clase con dos tipos de reemplazos:
        """
        self.reemplazoslist = {
            # Formas de decir "licencia de conducir"
            'brevet': 'licencia de conducir',
            'brevete': 'licencia de conducir',
            'carnet de manejo': 'licencia de conducir',
            'licencia de manejo': 'licencia de conducir',
            'permiso de conducir': 'licencia de conducir',
            'carnet de chofer': 'licencia de conducir',
            'registro de conducir': 'licencia de conducir',
            'pase de conducir': 'licencia de conducir',

            # Formas de decir "policías"
             'paco': 'policía',
            'policia': 'policía',
            'canas': 'policía',
            'yuta': 'policía',
            'tombos': 'policía',
            'oficial': 'policía',
            #'cerdo' : 'policia',
            'agente de tránsito': 'policía',

            # Formas de decir "accidente / choque"
            'choque': 'accidente',
            'colisión': 'accidente',
            'percance': 'accidente',
            'sinestro': 'accidente',
            'impacto': 'accidente',
            'crash': 'accidente',

            # Formas de decir "multa"
            'boleta': 'multa',
            'infracción': 'multa',
            'sanción': 'multa',
            'ticket': 'multa',
        }

    
    def add_reemplazo(self, palabra, reemplazo):
        """Añade un reemplazo directo de palabra exacta"""
        self.reemplazoslist[palabra] = reemplazo
    
    def eliminar_reemplazo(self, palabra):
        """Elimina un reemplazo directo"""
        if palabra in self.reemplazoslist:
            del self.reemplazoslist[palabra]
            return True
        return False
    
    def reemplazar_palabras(self, texto ):
      
        texto_procesado = texto
        
        texto_procesado = texto.lower()
        
        # Reemplazos 
        for palabra, reemplazo in self.reemplazoslist.items():
            texto_procesado = texto_procesado.replace(palabra, reemplazo)
        return texto_procesado
    
    def mostrar_reemplazos(self):
        """Muestra todos los reemplazos registrados"""
        return {
            'directos': self.reemplazoslist
        }

"""
reemplazador  = Regex()

reemplazador.add_reemplazo('carro', 'automóvil')

texto = "Mi Brevet está en el CARRO mibReveth"
print(reemplazador.reemplazar_palabras(texto)) 

"""

