import re
class filtro_palabras:
    reemplazoslist = None
    def __init__(self):
        """
        Inicializa la clase con varios tipos de reemplazos relacionados con documentos legales de tránsito
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
            'carné de conducción': 'licencia de conducir',
            'carnet de conducción': 'licencia de conducir',
            'permiso de manejo': 'licencia de conducir',
            'licencia para conducir': 'licencia de conducir',
            
            # Formas de decir "policías"
            'paco': 'policía',
            'policia': 'policía',
            'canas': 'policía',
            'yuta': 'policía',
            'tombos': 'policía',
            'oficial': 'policía',
            'agente de tránsito': 'policía',
            'guardia civil': 'policía',
            'carabinero': 'policía',
            'autoridad vial': 'policía',
            'agente vial': 'policía',
            
            # Formas de decir "accidente / choque"
            'choque': 'accidente',
            'colisión': 'accidente',
            'percance': 'accidente',
            'sinestro': 'accidente',
            'impacto': 'accidente',
            'crash': 'accidente',
            'evento vial': 'accidente',
            'incidente vehicular': 'accidente',
            'colisión vehicular': 'accidente',
            
            # Formas de decir "multa"
            'boleta': 'multa',
            'infracción': 'multa',
            'sanción': 'multa',
            'ticket': 'multa',
            'citación': 'multa',
            'papeleta': 'multa',
            'parte': 'multa',
            'penalización': 'multa',
            'boleto': 'multa',
            'comparendo': 'multa',
            
            # Documentos de vehículos
            'tarjeta de propiedad': 'tarjeta de circulación',
            'tarjeta verde': 'tarjeta de circulación',
            'documentación vehicular': 'tarjeta de circulación',
            'papeles del carro': 'documentos del vehículo',
            'papeles del auto': 'documentos del vehículo',
            'papeles del coche': 'documentos del vehículo',
            'tarjeta de rodaje': 'tarjeta de circulación',
            'permiso de circulación': 'tarjeta de circulación',
            
            # Seguro vehicular
            'seguro obligatorio': 'seguro vehicular',
            # 'soat': 'seguro obligatorio de accidentes de tránsito',
            'póliza vehicular': 'seguro vehicular',
            'seguro automotriz': 'seguro vehicular',
            'seguro del carro': 'seguro vehicular',
            'seguro del auto': 'seguro vehicular',
            
            # Vehículos
            'carro': 'vehículo',
            'auto': 'vehículo',
            'coche': 'vehículo',
            'automóvil': 'vehículo',
            'carruaje': 'vehículo',
            'máquina': 'vehículo',
            'nave': 'vehículo',
            
            # Infracciones
            'exceso de velocidad': 'infracción por velocidad',
            'pasarse el rojo': 'infracción por semáforo',
            'cruze prohibido': 'infracción por cruce prohibido',
            'cruce prohibido': 'infracción por cruce prohibido',
            'pasarse el semáforo': 'infracción por semáforo',
            'no respetar señal': 'infracción de señalización',
            'conducir ebrio': 'conducir bajo influencia del alcohol',
            'manejar borracho': 'conducir bajo influencia del alcohol',
            'alcoholemia': 'prueba de alcohol en sangre',
            
            # Autoridades y procesos
            'juzgado de tránsito': 'tribunal de tránsito',
            'corte de tráfico': 'tribunal de tránsito',
            'audiencia vial': 'audiencia de tránsito',
            'apelación de multa': 'recurso contra sanción',
            'recurso de multa': 'recurso contra sanción',
            'impugnación': 'recurso contra sanción',
            'departamento de tránsito': 'autoridad de tránsito',
            'dirección de tránsito': 'autoridad de tránsito',
            
            # Términos legales
            'decreto ley': 'normativa legal',
            'reglamento vial': 'código de tránsito',
            'código de tráfico': 'código de tránsito',
            'ley de tránsito': 'código de tránsito',
            'ordenanza municipal': 'normativa local',
            'disposición legal': 'normativa legal',
            'acta': 'documento oficial',
            'testificación': 'testimonio',
            'declaración jurada': 'declaración bajo juramento',
            
            # Situaciones de tránsito
            'alcoholemia': 'control de alcoholemia',
            'control vehicular': 'revisión vehicular',
            'operativo': 'control policial',
            'retén': 'control policial',
            'punto de control': 'control policial',
            'decomiso': 'confiscación',
            'incautación': 'confiscación',
            'retirada del vehículo': 'confiscación del vehículo',
            'grúa': 'servicio de remolque',
            'depósito vehicular': 'corralón',
            'corralón': 'depósito oficial de vehículos',
            'detención vehicular': 'inmovilización del vehículo',

            'revasar': 'adelantar',
            'rebasar': 'adelantar',
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
   
    def reemplazar_palabras(self, texto):
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
# Ejemplo de uso
reemplazador = filtro_palabras()
reemplazador.add_reemplazo('carro', 'automóvil')
texto = "Mi Brevet está en el CARRO y la tarjeta verde junto con el SOAT vencido"
print(reemplazador.reemplazar_palabras(texto))
"""