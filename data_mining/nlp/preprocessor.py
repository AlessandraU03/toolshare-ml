import re
import unicodedata

STOPWORDS_ES = {
    'de', 'la', 'en', 'el', 'que', 'y', 'un', 'una', 'con', 'para', 'por', 'es', 'al',
    'los', 'las', 'un', 'una', 'unos', 'unas', 'este', 'esta', 'estos', 'estas',
    'del', 'lo', 'como', 'o', 'su', 'sus', 'a', 'para', 'no', 'si'
}

TIPO_TO_SECTOR = {
    "Rotomartillo": "Eléctrico",
    "Taladro": "Eléctrico",
    "Esmeriladora": "Eléctrico",
    "Sierra": "Eléctrico",
    "Mezcladora": "Eléctrico",
    "Vibrador": "Eléctrico",
    "Segueta": "Corte",
    "Serrucho": "Corte",
    "Cortadora": "Corte",
    "Lijadora": "Acabado",
    "Generador": "Energía",
    "Soldadora": "Energía",
    "Compresor": "Neumático",
    "Clavadora": "Neumático",
    "Martillo": "Manual",
    "Mazo": "Manual",
    "Marro": "Manual",
    "Cincel": "Manual",
    "Llana": "Manual",
    "Espatula": "Manual",
    "Cuchara": "Manual",
    "Destornillador": "Manual",
    "Llave": "Manual",
    "Pinza": "Manual",
    "Pistola": "Manual",
    "Cepillo": "Manual",
    "Nivel": "Medición",
    "Plomada": "Medición",
    "Flexometro": "Medición",
    "Escuadra": "Medición",
    "Pala": "Otro",
    "Pico": "Otro",
    "Azadon": "Otro",
    "Carretilla": "Otro",
    "Cubeta": "Otro",
    "Andamio": "Otro",
    "Escalera": "Otro"
}

def preprocesar_texto(texto: str) -> str:
    """Pipeline de preprocesamiento NLP alineado con la clase de Minería de Datos."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    tokens = re.findall(r'\b[a-z0-9ñ]{2,}\b', texto)
    tokens_filtrados = [t for t in tokens if t not in STOPWORDS_ES]
    return " ".join(tokens_filtrados)

def inferir_tipo_herramienta(nombre: str) -> str:
    """Clasificador simple basado en reglas NLP para mapear la herramienta con el dataset."""
    nombre_clean = preprocesar_texto(nombre).lower()
    
    # Listado de clases/tipos definidos en nuestro dataset real de construcción
    tipos_validos = {
        "rotomartillo": "Rotomartillo",
        "taladro": "Taladro",
        "esmeriladora": "Esmeriladora",
        "esmeril": "Esmeriladora",
        "amoladora": "Esmeriladora",
        "sierra": "Sierra",
        "segueta": "Segueta",
        "serrucho": "Serrucho",
        "cortadora": "Cortadora",
        "lijadora": "Lijadora",
        "cepillo": "Cepillo",
        "pulidora": "Pulidora",
        "vibrador": "Vibrador",
        "generador": "Generador",
        "soldadora": "Soldadora",
        "extension": "Extension",
        "compresor": "Compresor",
        "clavadora": "Clavadora",
        "clavos": "Clavadora",
        "impacto": "Pistola de Impacto",
        "juego": "Juego de Herramientas",
        "autocle": "Juego de Herramientas",
        "llave": "Llave",
        "perica": "Llave",
        "inglesa": "Llave",
        "stilson": "Llave",
        "martillo": "Martillo",
        "mazo": "Mazo",
        "marro": "Marro",
        "cincel": "Cincel",
        "llana": "Llana",
        "espatula": "Espatula",
        "cuchara": "Cuchara",
        "paleta": "Cuchara",
        "desarmador": "Destornillador",
        "destornillador": "Destornillador",
        "pinza": "Pinza",
        "alicates": "Pinza",
        "tenazas": "Pinza",
        "multimetro": "Multimetro",
        "nivel": "Nivel",
        "burbuja": "Nivel",
        "plomada": "Plomada",
        "flexometro": "Flexometro",
        "metro": "Flexometro",
        "cinta": "Flexometro",
        "escuadra": "Escuadra",
        "distanciometro": "Distanciometro",
        "escalera": "Escalera",
        "carretilla": "Carretilla",
        "pala": "Pala",
        "pico": "Pico",
        "azadon": "Azadon",
        "cubeta": "Cubeta",
        "mezcladora": "Mezcladora",
        "andamio": "Andamio",
        "silicon": "Pistola",
        "alambre": "Cepillo"
    }
    
    for kw, tipo in tipos_validos.items():
        if kw in nombre_clean:
            return tipo
    return "Otros"
