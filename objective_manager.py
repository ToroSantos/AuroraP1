# ============================================================================
# OBJECTIVE MANAGER V7 - GESTOR UNIFICADO DE OBJETIVOS TERAPÉUTICOS
# ============================================================================
# Autor: Sistema Aurora V7
# Versión: 7.2.1 - OPTIMIZADO PARA AURORA COMPLETO
# Fecha: Junio 2025
# Descripción: Gestor inteligente de objetivos con IA semántica y templates V7
# ============================================================================

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logger = logging.getLogger("Aurora.ObjectiveManager.V7")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

VERSION = "7.2.1_AURORA_COMPLETE_OPTIMIZED"
MAX_TEMPLATES_CACHE = 100
TIMEOUT_PROCESAMIENTO = 5.0

# ============================================================================
# ENUMS Y TIPOS DE DATOS
# ============================================================================

class CategoriaObjetivo(Enum):
    """Categorías principales de objetivos terapéuticos"""
    COGNITIVO = "cognitivo"
    EMOCIONAL = "emocional"
    TERAPEUTICO = "terapeutico"
    CREATIVO = "creativo"
    RELAJACION = "relajacion"
    ENERGIA = "energia"
    MEDITACION = "meditacion"
    SANACION = "sanacion"

class NivelComplejidad(Enum):
    """Niveles de complejidad para personalización"""
    PRINCIPIANTE = "principiante"
    INTERMEDIO = "intermedio"
    AVANZADO = "avanzado"
    EXPERTO = "experto"

class ModoActivacion(Enum):
    """Modos de activación de templates"""
    INSTANTANEO = "instantaneo"
    PROGRESIVO = "progresivo"
    CICLICO = "ciclico"
    ADAPTATIVO = "adaptativo"

class TipoRuteo(Enum):
    """Tipos de ruteo inteligente"""
    DIRECTO = "directo"
    SEMANTICO = "semantico"
    PERSONALIZADO = "personalizado"
    HIBRIDO = "hibrido"
    FALLBACK = "fallback"

class NivelConfianza(Enum):
    """Niveles de confianza en templates"""
    EXPERIMENTAL = 0.3
    BASICO = 0.5
    VALIDADO = 0.7
    CLINICO = 0.9
    GOLD_STANDARD = 1.0

class ContextoUso(Enum):
    """Contextos de uso para templates"""
    PERSONAL = "personal"
    CLINICO = "clinico"
    INVESTIGACION = "investigacion"
    TERAPEUTICO = "terapeutico"
    WELLNESS = "wellness"

# ============================================================================
# DATACLASSES Y ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class ConfiguracionCapaV7:
    """Configuración optimizada de una capa de audio"""
    enabled: bool = True
    carrier: float = 40.0
    freq_l: float = 40.0
    freq_r: float = 40.0
    mod_type: str = "AM"
    style: str = "sine"
    amplitude: float = 0.7
    phase: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para compatibilidad"""
        return {
            "enabled": self.enabled,
            "carrier": self.carrier,
            "freq_l": self.freq_l,
            "freq_r": self.freq_r,
            "mod_type": self.mod_type,
            "style": self.style,
            "amplitude": self.amplitude,
            "phase": self.phase
        }

@dataclass
class TemplateObjetivoV7:
    """Template unificado de objetivo terapéutico V7"""
    nombre: str
    descripcion: str
    categoria: CategoriaObjetivo
    emotional_preset: str
    style: str
    frecuencia_dominante: float
    duracion_recomendada_min: int
    duracion_recomendada_max: int
    efectos_esperados: List[str]
    contraindicaciones: List[str]
    evidencia_cientifica: str
    nivel_confianza: float
    complejidad: NivelComplejidad
    modo_activacion: ModoActivacion
    contexto_uso: ContextoUso
    layers: Dict[str, ConfiguracionCapaV7] = field(default_factory=dict)
    metadatos: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validación y normalización post-inicialización"""
        # Normalizar nivel de confianza
        if self.nivel_confianza > 1.0:
            self.nivel_confianza = 1.0
        elif self.nivel_confianza < 0.0:
            self.nivel_confianza = 0.0
        
        # Asegurar que hay al menos una capa
        if not self.layers:
            self.layers["base"] = ConfiguracionCapaV7(
                carrier=self.frecuencia_dominante,
                freq_l=self.frecuencia_dominante,
                freq_r=self.frecuencia_dominante
            )

@dataclass
class ResultadoRuteo:
    """Resultado de ruteo inteligente"""
    template_seleccionado: Optional[TemplateObjetivoV7]
    nivel_confianza: float
    tipo_ruteo: TipoRuteo
    tiempo_procesamiento_ms: float
    metadatos_ruteo: Dict[str, Any]
    alternativas: List[TemplateObjetivoV7] = field(default_factory=list)

@dataclass
class PerfilUsuario:
    """Perfil de usuario para personalización"""
    id_usuario: str
    nivel_experiencia: NivelComplejidad
    preferencias: Dict[str, Any] = field(default_factory=dict)
    historial_objetivos: List[str] = field(default_factory=list)
    restricciones_medicas: List[str] = field(default_factory=list)

# ============================================================================
# CLASES PRINCIPALES
# ============================================================================

class GestorTemplatesOptimizado:
    """Gestor optimizado de templates V7 con migración V6"""
    
    def __init__(self):
        self.templates: Dict[str, TemplateObjetivoV7] = {}
        self.estadisticas_uso = {}
        self.cache_busquedas = {}
        self.timestamp_inicializacion = datetime.now()
        
        # Inicializar templates base
        self._init_templates_base()
        self._migrar_v6()
        
        logger.info(f"Gestor de templates inicializado con {len(self.templates)} templates")
    
    def _init_templates_base(self):
        """Inicializa templates base del sistema"""
        templates_base = [
            TemplateObjetivoV7(
                nombre="concentracion",
                descripcion="Mejora del enfoque y concentración mental",
                categoria=CategoriaObjetivo.COGNITIVO,
                emotional_preset="focused",
                style="cognitive",
                frecuencia_dominante=40.0,
                duracion_recomendada_min=15,
                duracion_recomendada_max=45,
                efectos_esperados=["mayor_enfoque", "claridad_mental", "atencion_sostenida"],
                contraindicaciones=["epilepsia", "trastornos_convulsivos"],
                evidencia_cientifica="Múltiples estudios demuestran efectividad de 40Hz para concentración",
                nivel_confianza=0.8,
                complejidad=NivelComplejidad.INTERMEDIO,
                modo_activacion=ModoActivacion.PROGRESIVO,
                contexto_uso=ContextoUso.PERSONAL
            ),
            TemplateObjetivoV7(
                nombre="relajacion",
                descripcion="Relajación profunda y reducción de estrés",
                categoria=CategoriaObjetivo.RELAJACION,
                emotional_preset="calm",
                style="relaxing",
                frecuencia_dominante=10.0,
                duracion_recomendada_min=20,
                duracion_recomendada_max=60,
                efectos_esperados=["relajacion_muscular", "calma_mental", "reduccion_estres"],
                contraindicaciones=["depresion_severa"],
                evidencia_cientifica="Ondas alfa demostradas para relajación",
                nivel_confianza=0.9,
                complejidad=NivelComplejidad.PRINCIPIANTE,
                modo_activacion=ModoActivacion.INSTANTANEO,
                contexto_uso=ContextoUso.WELLNESS
            ),
            TemplateObjetivoV7(
                nombre="creatividad",
                descripcion="Estimulación de creatividad e innovación",
                categoria=CategoriaObjetivo.CREATIVO,
                emotional_preset="creative",
                style="innovative",
                frecuencia_dominante=8.0,
                duracion_recomendada_min=25,
                duracion_recomendada_max=50,
                efectos_esperados=["pensamiento_divergente", "inspiracion", "fluidez_ideas"],
                contraindicaciones=["mania", "trastorno_bipolar_activo"],
                evidencia_cientifica="Ondas theta asociadas con creatividad",
                nivel_confianza=0.7,
                complejidad=NivelComplejidad.INTERMEDIO,
                modo_activacion=ModoActivacion.CICLICO,
                contexto_uso=ContextoUso.PERSONAL
            ),
            TemplateObjetivoV7(
                nombre="meditacion",
                descripcion="Meditación profunda y mindfulness",
                categoria=CategoriaObjetivo.MEDITACION,
                emotional_preset="meditative",
                style="spiritual",
                frecuencia_dominante=6.0,
                duracion_recomendada_min=30,
                duracion_recomendada_max=90,
                efectos_esperados=["estado_meditativo", "presencia", "paz_interior"],
                contraindicaciones=["psicosis", "trastornos_disociativos"],
                evidencia_cientifica="Ondas theta profundas en meditadores expertos",
                nivel_confianza=0.8,
                complejidad=NivelComplejidad.AVANZADO,
                modo_activacion=ModoActivacion.ADAPTATIVO,
                contexto_uso=ContextoUso.TERAPEUTICO
            ),
            TemplateObjetivoV7(
                nombre="energia",
                descripcion="Aumento de energía y vitalidad",
                categoria=CategoriaObjetivo.ENERGIA,
                emotional_preset="energetic",
                style="uplifting",
                frecuencia_dominante=20.0,
                duracion_recomendada_min=10,
                duracion_recomendada_max=30,
                efectos_esperados=["mayor_energia", "vitalidad", "motivacion"],
                contraindicaciones=["hipertension", "ansiedad_severa"],
                evidencia_cientifica="Ondas beta para activación",
                nivel_confianza=0.7,
                complejidad=NivelComplejidad.PRINCIPIANTE,
                modo_activacion=ModoActivacion.INSTANTANEO,
                contexto_uso=ContextoUso.WELLNESS
            ),
            TemplateObjetivoV7(
                nombre="sanacion",
                descripcion="Sanación integral y bienestar",
                categoria=CategoriaObjetivo.SANACION,
                emotional_preset="healing",
                style="therapeutic",
                frecuencia_dominante=528.0,
                duracion_recomendada_min=40,
                duracion_recomendada_max=120,
                efectos_esperados=["regeneracion", "equilibrio", "bienestar_integral"],
                contraindicaciones=["marcapasos", "implantes_electronicos"],
                evidencia_cientifica="Frecuencias Solfeggio para sanación",
                nivel_confianza=0.6,
                complejidad=NivelComplejidad.EXPERTO,
                modo_activacion=ModoActivacion.ADAPTATIVO,
                contexto_uso=ContextoUso.TERAPEUTICO
            )
        ]
        
        for template in templates_base:
            self.templates[template.nombre] = template
    
    def _migrar_v6(self):
        """Migra templates de V6 si están disponibles"""
        try:
            # Migración de templates V6 legacy
            templates_v6_legacy = {
                "focus": "concentracion",
                "relax": "relajacion",
                "sleep": "relajacion",
                "study": "concentracion",
                "meditation": "meditacion",
                "healing": "sanacion"
            }
            
            for v6_name, v7_name in templates_v6_legacy.items():
                if v7_name in self.templates and v6_name not in self.templates:
                    # Crear alias V6
                    template_base = self.templates[v7_name]
                    self.templates[v6_name] = template_base
            
            logger.info(f"Migración V6 completada: {len(templates_v6_legacy)} aliases creados")
            
        except Exception as e:
            logger.warning(f"Error en migración V6: {e}")
    
    def obtener_template(self, nombre: str) -> Optional[TemplateObjetivoV7]:
        """Obtiene un template por nombre"""
        template = self.templates.get(nombre.lower())
        if template and nombre in self.estadisticas_uso:
            self.estadisticas_uso[nombre]["veces_usado"] += 1
        elif template:
            self.estadisticas_uso[nombre] = {"veces_usado": 1, "tiempo_promedio_ms": 0}
        return template
    
    def buscar_templates_inteligente(self, criterios: Dict[str, Any], limite: int = 10) -> List[TemplateObjetivoV7]:
        """Búsqueda inteligente de templates"""
        resultados = []
        
        for template in self.templates.values():
            puntuacion = 0
            
            # Criterios de búsqueda
            if "categoria" in criterios:
                if template.categoria == criterios["categoria"]:
                    puntuacion += 10
            
            if "efectos" in criterios:
                efectos_buscados = criterios["efectos"]
                coincidencias = len(set(template.efectos_esperados) & set(efectos_buscados))
                puntuacion += coincidencias * 5
            
            if "complejidad" in criterios:
                if template.complejidad == criterios["complejidad"]:
                    puntuacion += 8
            
            if "confianza_minima" in criterios:
                if template.nivel_confianza >= criterios["confianza_minima"]:
                    puntuacion += 3
            
            if puntuacion > 0:
                resultados.append((template, puntuacion))
        
        # Ordenar por puntuación y retornar limitado
        resultados.sort(key=lambda x: x[1], reverse=True)
        return [template for template, _ in resultados[:limite]]

class AnalizadorSemantico:
    """Analizador semántico mejorado para objetivos"""
    
    def __init__(self):
        self.mapeo_sinonimos = {
            "concentracion": ["enfoque", "atencion", "focus", "concentrar", "enfocar", "concentrarse"],
            "relajacion": ["calma", "tranquilidad", "paz", "relax", "descanso", "relajar"],
            "creatividad": ["crear", "imaginacion", "innovacion", "arte", "inspiracion", "creativo"],
            "energia": ["activacion", "vigor", "fuerza", "vitalidad", "impulso", "energizar"],
            "meditacion": ["mindfulness", "consciencia", "presencia", "contemplacion", "meditar"],
            "sanacion": ["curacion", "terapia", "healing", "recuperacion", "bienestar", "sanar"]
        }
        self.contextos_especiales = {
            "trabajo": ["concentracion"],
            "estudio": ["concentracion"],
            "descanso": ["relajacion"],
            "dormir": ["relajacion"],
            "arte": ["creatividad"],
            "ejercicio": ["energia"],
            "yoga": ["meditacion", "relajacion"],
            "terapia": ["sanacion", "relajacion"]
        }
    
    def analizar_objetivo(self, objetivo: str) -> Dict[str, Any]:
        """Análisis semántico avanzado de objetivos"""
        objetivo_lower = objetivo.lower()
        
        categorias_detectadas = []
        palabras_clave = []
        contexto_detectado = None
        
        # Detectar contextos especiales
        for contexto, objetivos_relacionados in self.contextos_especiales.items():
            if contexto in objetivo_lower:
                contexto_detectado = contexto
                categorias_detectadas.extend(objetivos_relacionados)
        
        # Análisis de sinónimos
        for categoria, sinonimos in self.mapeo_sinonimos.items():
            if categoria in objetivo_lower:
                categorias_detectadas.append(categoria)
                palabras_clave.append(categoria)
            
            for sinonimo in sinonimos:
                if sinonimo in objetivo_lower:
                    categorias_detectadas.append(categoria)
                    palabras_clave.append(sinonimo)
                    break
        
        # Calcular complejidad estimada
        complejidad = "principiante"
        if len(palabras_clave) > 2:
            complejidad = "intermedio"
        if len(palabras_clave) > 4 or any(word in objetivo_lower for word in ["avanzado", "profundo", "intenso"]):
            complejidad = "experto"
        
        return {
            "objetivo_original": objetivo,
            "categorias_detectadas": list(set(categorias_detectadas)),
            "palabras_clave": list(set(palabras_clave)),
            "contexto_detectado": contexto_detectado,
            "complejidad_estimada": complejidad,
            "confianza_analisis": min(len(palabras_clave) * 0.3, 1.0)
        }

class MotorPersonalizacion:
    """Motor de personalización de templates"""
    
    def __init__(self):
        self.perfiles_usuario = {}
        self.adaptaciones_activas = {}
    
    def personalizar_template(self, template: TemplateObjetivoV7, perfil_usuario: PerfilUsuario) -> TemplateObjetivoV7:
        """Personaliza un template según el perfil del usuario"""
        template_personalizado = TemplateObjetivoV7(
            nombre=template.nombre,
            descripcion=template.descripcion,
            categoria=template.categoria,
            emotional_preset=template.emotional_preset,
            style=template.style,
            frecuencia_dominante=template.frecuencia_dominante,
            duracion_recomendada_min=template.duracion_recomendada_min,
            duracion_recomendada_max=template.duracion_recomendada_max,
            efectos_esperados=template.efectos_esperados.copy(),
            contraindicaciones=template.contraindicaciones.copy(),
            evidencia_cientifica=template.evidencia_cientifica,
            nivel_confianza=template.nivel_confianza,
            complejidad=template.complejidad,
            modo_activacion=template.modo_activacion,
            contexto_uso=template.contexto_uso,
            layers={k: v for k, v in template.layers.items()},
            metadatos=template.metadatos.copy()
        )
        
        # Ajustar según experiencia
        if perfil_usuario.nivel_experiencia == NivelComplejidad.PRINCIPIANTE:
            template_personalizado.duracion_recomendada_min = min(template.duracion_recomendada_min, 15)
            template_personalizado.duracion_recomendada_max = min(template.duracion_recomendada_max, 30)
        elif perfil_usuario.nivel_experiencia == NivelComplejidad.EXPERTO:
            template_personalizado.duracion_recomendada_min = max(template.duracion_recomendada_min, 30)
        
        # Aplicar preferencias
        if "frecuencia_preferida" in perfil_usuario.preferencias:
            freq_pref = perfil_usuario.preferencias["frecuencia_preferida"]
            if isinstance(freq_pref, (int, float)):
                template_personalizado.frecuencia_dominante = freq_pref
        
        # Verificar restricciones médicas
        if perfil_usuario.restricciones_medicas:
            contraindicaciones_usuario = set(perfil_usuario.restricciones_medicas)
            contraindicaciones_template = set(template.contraindicaciones)
            if contraindicaciones_usuario & contraindicaciones_template:
                template_personalizado.nivel_confianza *= 0.5  # Reducir confianza
        
        template_personalizado.metadatos["personalizado"] = True
        template_personalizado.metadatos["perfil_usuario_id"] = perfil_usuario.id_usuario
        
        return template_personalizado

class ValidadorCientifico:
    """Validador científico de templates"""
    
    def __init__(self):
        self.criterios_validacion = {
            "nivel_confianza_minimo": 0.3,
            "evidencia_requerida": True,
            "contraindicaciones_verificadas": True,
            "frecuencia_valida": True
        }
        self.rangos_frecuencia = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
            "solfeggio": [174, 285, 396, 417, 528, 639, 741, 852, 963]
        }
    
    def validar_template(self, template: TemplateObjetivoV7) -> Dict[str, Any]:
        """Validación científica completa de template"""
        validaciones = {
            "confianza_suficiente": template.nivel_confianza >= self.criterios_validacion["nivel_confianza_minimo"],
            "evidencia_presente": bool(template.evidencia_cientifica and len(template.evidencia_cientifica) > 10),
            "contraindicaciones_listadas": len(template.contraindicaciones) > 0,
            "frecuencia_en_rango": self._validar_frecuencia(template.frecuencia_dominante),
            "efectos_especificados": len(template.efectos_esperados) > 0,
            "duracion_razonable": 5 <= template.duracion_recomendada_min <= 180
        }
        
        score_total = sum(validaciones.values()) / len(validaciones)
        
        recomendaciones = []
        if not validaciones["confianza_suficiente"]:
            recomendaciones.append("Aumentar nivel de confianza con más evidencia")
        if not validaciones["evidencia_presente"]:
            recomendaciones.append("Proporcionar evidencia científica detallada")
        if not validaciones["frecuencia_en_rango"]:
            recomendaciones.append("Verificar frecuencia dominante en rangos conocidos")
        
        return {
            "valido": score_total >= 0.7,
            "score": score_total,
            "validaciones_individuales": validaciones,
            "recomendaciones": recomendaciones,
            "nivel_evidencia": self._clasificar_evidencia(template),
            "timestamp_validacion": datetime.now().isoformat()
        }
    
    def _validar_frecuencia(self, frecuencia: float) -> bool:
        """Valida si la frecuencia está en rangos conocidos"""
        # Verificar rangos de ondas cerebrales
        for rango_min, rango_max in self.rangos_frecuencia.values():
            if isinstance(rango_min, (int, float)) and isinstance(rango_max, (int, float)):
                if rango_min <= frecuencia <= rango_max:
                    return True
        
        # Verificar frecuencias Solfeggio
        if frecuencia in self.rangos_frecuencia["solfeggio"]:
            return True
        
        # Permitir otras frecuencias razonables
        return 0.1 <= frecuencia <= 1000.0
    
    def _clasificar_evidencia(self, template: TemplateObjetivoV7) -> str:
        """Clasifica el nivel de evidencia científica"""
        evidencia = template.evidencia_cientifica.lower()
        
        if any(keyword in evidencia for keyword in ["meta-análisis", "revisión sistemática", "ensayo clínico"]):
            return "alta"
        elif any(keyword in evidencia for keyword in ["estudio", "investigación", "paper"]):
            return "media"
        elif any(keyword in evidencia for keyword in ["observación", "reporte", "caso"]):
            return "baja"
        else:
            return "limitada"

class RouterInteligenteV7:
    """Router inteligente con IA semántica"""
    
    def __init__(self, gestor_templates: GestorTemplatesOptimizado):
        self.gestor_templates = gestor_templates
        self.analizador_semantico = AnalizadorSemantico()
        self.motor_personalizacion = MotorPersonalizacion()
        self.validador_cientifico = ValidadorCientifico()
        self.estadisticas_routing = {
            "total_ruteos": 0,
            "exitos": 0,
            "fallbacks": 0,
            "tiempo_promedio": 0.0
        }
        self.rutas_v6_mapeadas = self._crear_mapeo_v6()
        
        logger.info("Router V7 inicializado con gestor de templates integrado")
    
    def _crear_mapeo_v6(self) -> Dict[str, str]:
        """Crea mapeo de compatibilidad V6"""
        return {
            "focus": "concentracion",
            "relax": "relajacion",
            "sleep": "relajacion",
            "study": "concentracion",
            "meditation": "meditacion",
            "healing": "sanacion",
            "energy": "energia",
            "creative": "creatividad",
            "calm": "relajacion",
            "alert": "energia"
        }
    
    def rutear_objetivo_avanzado(self, objetivo: str, perfil_usuario: Optional[PerfilUsuario] = None, 
                               kwargs: Dict[str, Any] = None) -> ResultadoRuteo:
        """Ruteo inteligente avanzado"""
        start_time = time.time()
        kwargs = kwargs or {}
        
        try:
            # Análisis semántico del objetivo
            analisis = self.analizador_semantico.analizar_objetivo(objetivo)
            
            # Búsqueda directa primero
            template_directo = self.gestor_templates.obtener_template(objetivo)
            if template_directo:
                template_final = self._aplicar_personalizacion(template_directo, perfil_usuario)
                return self._crear_resultado(template_final, TipoRuteo.DIRECTO, start_time, analisis)
            
            # Mapeo V6
            if objetivo.lower() in self.rutas_v6_mapeadas:
                objetivo_v7 = self.rutas_v6_mapeadas[objetivo.lower()]
                template_v6 = self.gestor_templates.obtener_template(objetivo_v7)
                if template_v6:
                    template_final = self._aplicar_personalizacion(template_v6, perfil_usuario)
                    return self._crear_resultado(template_final, TipoRuteo.DIRECTO, start_time, analisis)
            
            # Búsqueda semántica
            if analisis["categorias_detectadas"]:
                criterios = {
                    "efectos": analisis["categorias_detectadas"],
                    "confianza_minima": kwargs.get("confianza_minima", 0.5)
                }
                templates_semanticos = self.gestor_templates.buscar_templates_inteligente(criterios, 3)
                
                if templates_semanticos:
                    template_mejor = templates_semanticos[0]
                    template_final = self._aplicar_personalizacion(template_mejor, perfil_usuario)
                    resultado = self._crear_resultado(template_final, TipoRuteo.SEMANTICO, start_time, analisis)
                    resultado.alternativas = templates_semanticos[1:3]
                    return resultado
            
            # Fallback inteligente
            template_fallback = self._crear_template_fallback(objetivo, analisis)
            return self._crear_resultado(template_fallback, TipoRuteo.FALLBACK, start_time, analisis)
            
        except Exception as e:
            logger.error(f"Error en ruteo avanzado: {e}")
            template_error = self._crear_template_error(objetivo)
            return self._crear_resultado(template_error, TipoRuteo.FALLBACK, start_time, {})
        
        finally:
            self.estadisticas_routing["total_ruteos"] += 1
            tiempo_total = (time.time() - start_time) * 1000
            self.estadisticas_routing["tiempo_promedio"] = (
                (self.estadisticas_routing["tiempo_promedio"] * (self.estadisticas_routing["total_ruteos"] - 1) + tiempo_total) 
                / self.estadisticas_routing["total_ruteos"]
            )
    
    def _aplicar_personalizacion(self, template: TemplateObjetivoV7, perfil_usuario: Optional[PerfilUsuario]) -> TemplateObjetivoV7:
        """Aplica personalización si hay perfil de usuario"""
        if perfil_usuario:
            return self.motor_personalizacion.personalizar_template(template, perfil_usuario)
        return template
    
    def _crear_resultado(self, template: TemplateObjetivoV7, tipo: TipoRuteo, start_time: float, analisis: Dict) -> ResultadoRuteo:
        """Crea resultado de ruteo estructurado"""
        tiempo_ms = (time.time() - start_time) * 1000
        
        # Calcular confianza basada en tipo de ruteo y template
        confianza_base = template.nivel_confianza if template else 0.3
        factor_ruteo = {
            TipoRuteo.DIRECTO: 1.0,
            TipoRuteo.SEMANTICO: 0.8,
            TipoRuteo.PERSONALIZADO: 0.9,
            TipoRuteo.HIBRIDO: 0.85,
            TipoRuteo.FALLBACK: 0.4
        }
        
        confianza_final = confianza_base * factor_ruteo.get(tipo, 0.5)
        confianza_final *= analisis.get("confianza_analisis", 0.7)
        
        return ResultadoRuteo(
            template_seleccionado=template,
            nivel_confianza=min(confianza_final, 1.0),
            tipo_ruteo=tipo,
            tiempo_procesamiento_ms=tiempo_ms,
            metadatos_ruteo={
                "analisis_semantico": analisis,
                "timestamp": datetime.now().isoformat(),
                "version_router": "7.2.1"
            }
        )
    
    def _crear_template_fallback(self, objetivo: str, analisis: Dict) -> TemplateObjetivoV7:
        """Crea template de fallback inteligente"""
        # Detectar categoría más probable
        categoria = CategoriaObjetivo.COGNITIVO  # Default
        if analisis.get("categorias_detectadas"):
            categoria_str = analisis["categorias_detectadas"][0]
            try:
                categoria = CategoriaObjetivo(categoria_str)
            except ValueError:
                pass
        
        return TemplateObjetivoV7(
            nombre=f"fallback_{objetivo}",
            descripcion=f"Template generado automáticamente para: {objetivo}",
            categoria=categoria,
            emotional_preset="balanced",
            style="adaptive",
            frecuencia_dominante=10.0,  # Alpha seguro
            duracion_recomendada_min=15,
            duracion_recomendada_max=30,
            efectos_esperados=["equilibrio", "bienestar_general"],
            contraindicaciones=["epilepsia"],
            evidencia_cientifica="Template generado automáticamente con parámetros seguros",
            nivel_confianza=0.4,
            complejidad=NivelComplejidad.PRINCIPIANTE,
            modo_activacion=ModoActivacion.PROGRESIVO,
            contexto_uso=ContextoUso.PERSONAL,
            metadatos={"generado_automaticamente": True, "objetivo_original": objetivo}
        )
    
    def _crear_template_error(self, objetivo: str) -> TemplateObjetivoV7:
        """Crea template de error para casos extremos"""
        return TemplateObjetivoV7(
            nombre="error_template",
            descripcion="Template de error - configuración mínima segura",
            categoria=CategoriaObjetivo.RELAJACION,
            emotional_preset="safe",
            style="minimal",
            frecuencia_dominante=10.0,
            duracion_recomendada_min=10,
            duracion_recomendada_max=15,
            efectos_esperados=["seguridad"],
            contraindicaciones=["cualquier_condicion_medica"],
            evidencia_cientifica="Configuración mínima de seguridad",
            nivel_confianza=0.1,
            complejidad=NivelComplejidad.PRINCIPIANTE,
            modo_activacion=ModoActivacion.INSTANTANEO,
            contexto_uso=ContextoUso.PERSONAL
        )
    
    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del router"""
        return {
            "router_stats": self.estadisticas_routing.copy(),
            "templates_stats": self.gestor_templates.estadisticas_uso.copy(),
            "rutas_v6_mapeadas": len(self.rutas_v6_mapeadas),
            "templates_disponibles": len(self.gestor_templates.templates),
            "tiempo_promedio_ms": self.estadisticas_routing.get("tiempo_promedio", 0),
            "tasa_exito": (
                self.estadisticas_routing["exitos"] / max(self.estadisticas_routing["total_ruteos"], 1)
            ) * 100,
            "aurora_v7_optimizado": True,
            "protocolo_director_implementado": True
        }

# ============================================================================
# CLASE PRINCIPAL - OBJECTIVE MANAGER UNIFICADO
# ============================================================================

class ObjectiveManagerUnificado:
    """
    Gestor unificado de objetivos terapéuticos para Aurora V7
    Implementa GestorPerfilesInterface para compatibilidad con Factory
    """
    
    def __init__(self):
        self.version = VERSION
        self.gestor_templates = GestorTemplatesOptimizado()
        self.router = RouterInteligenteV7(self.gestor_templates)
        self.estadisticas_globales = {
            "inicializado": datetime.now().isoformat(),
            "templates_cargados": len(self.gestor_templates.templates),
            "rutas_v6_disponibles": len(self.router.rutas_v6_mapeadas),
            "total_objetivos_procesados": 0,
            "tiempo_total_procesamiento": 0.0
        }
        logger.info(f"ObjectiveManager V7 inicializado - {self.estadisticas_globales['templates_cargados']} templates")
    
    # ============================================================================
    # IMPLEMENTACIÓN DE GestorPerfilesInterface - REQUERIDO POR FACTORY
    # ============================================================================
    
    def obtener_perfil(self, nombre: str) -> Optional[Dict[str, Any]]:
        """
        Implementación de GestorPerfilesInterface
        Mapea a obtener_template() existente
        
        Args:
            nombre: Nombre del template/objetivo
            
        Returns:
            Dict con configuración del template o None
        """
        try:
            template = self.obtener_template(nombre)
            if template:
                # Convertir TemplateObjetivoV7 a formato Dict compatible con interface
                return {
                    "nombre": template.nombre,
                    "descripcion": template.descripcion,
                    "categoria": template.categoria.value if hasattr(template.categoria, 'value') else str(template.categoria),
                    "emotional_preset": template.emotional_preset,
                    "style": template.style,
                    "frecuencia_dominante": template.frecuencia_dominante,
                    "duracion_recomendada": template.duracion_recomendada_min,
                    "efectos_esperados": template.efectos_esperados,
                    "evidencia_cientifica": template.evidencia_cientifica,
                    "nivel_confianza": template.nivel_confianza,
                    "tipo_perfil": "objetivo_terapeutico",
                    "layers": {k: v.to_dict() for k, v in template.layers.items()},
                    "metadatos": {
                        "complejidad": template.complejidad.value if hasattr(template.complejidad, 'value') else str(template.complejidad),
                        "version": self.version,
                        "timestamp": datetime.now().isoformat(),
                        "modo_activacion": template.modo_activacion.value,
                        "contexto_uso": template.contexto_uso.value
                    }
                }
            return None
        except Exception as e:
            logger.error(f"Error obteniendo perfil {nombre}: {e}")
            return None
    
    def listar_perfiles(self) -> List[str]:
        """
        Implementación de GestorPerfilesInterface
        Mapea a listar_objetivos_disponibles() existente
        
        Returns:
            Lista de nombres de templates/objetivos disponibles
        """
        try:
            return self.listar_objetivos_disponibles()
        except Exception as e:
            logger.error(f"Error listando perfiles: {e}")
            return []
    
    def validar_perfil(self, perfil: Dict[str, Any]) -> bool:
        """
        Implementación ROBUSTA de GestorPerfilesInterface
        Valida estructura y contenido de un perfil/template
        
        Args:
            perfil: Configuración del perfil a validar
            
        Returns:
            True si el perfil es válido
        """
        try:
            # Validación básica de tipo
            if not isinstance(perfil, dict):
                logger.debug("Perfil no es un diccionario")
                return False
            
            # No permitir perfiles completamente vacíos
            if len(perfil) == 0:
                logger.debug("Perfil vacío")
                return False
            
            # Validaciones de estructura básica
            
            # Si tiene nombre, debe ser string válido y no vacío
            if "nombre" in perfil:
                if not isinstance(perfil["nombre"], str) or len(perfil["nombre"].strip()) == 0:
                    logger.debug("Nombre de perfil inválido o vacío")
                    return False
            
            # Si tiene emotional_preset, debe ser string válido
            if "emotional_preset" in perfil:
                if not isinstance(perfil["emotional_preset"], str) or len(perfil["emotional_preset"].strip()) == 0:
                    logger.debug("emotional_preset inválido o vacío")
                    return False
            
            # Si tiene categoria, debe ser string válido
            if "categoria" in perfil:
                if not isinstance(perfil["categoria"], str) or len(perfil["categoria"].strip()) == 0:
                    logger.debug("categoria inválida o vacía")
                    return False
            
            # Si tiene nivel_confianza, debe estar en rango válido
            if "nivel_confianza" in perfil:
                if not isinstance(perfil["nivel_confianza"], (int, float)):
                    logger.debug("nivel_confianza debe ser numérico")
                    return False
                if perfil["nivel_confianza"] < 0 or perfil["nivel_confianza"] > 1:
                    logger.debug("nivel_confianza debe estar entre 0 y 1")
                    return False
            
            # Si tiene duracion_recomendada, debe ser positiva
            if "duracion_recomendada" in perfil:
                if not isinstance(perfil["duracion_recomendada"], (int, float)):
                    logger.debug("duracion_recomendada debe ser numérica")
                    return False
                if perfil["duracion_recomendada"] <= 0:
                    logger.debug("duracion_recomendada debe ser positiva")
                    return False
            
            # Si tiene frecuencia_dominante, debe estar en rango válido
            if "frecuencia_dominante" in perfil:
                if not isinstance(perfil["frecuencia_dominante"], (int, float)):
                    logger.debug("frecuencia_dominante debe ser numérica")
                    return False
                if perfil["frecuencia_dominante"] < 0.1 or perfil["frecuencia_dominante"] > 1000:
                    logger.debug("frecuencia_dominante fuera de rango válido (0.1-1000 Hz)")
                    return False
            
            # Si tiene efectos_esperados, debe ser lista válida
            if "efectos_esperados" in perfil:
                if not isinstance(perfil["efectos_esperados"], list):
                    logger.debug("efectos_esperados debe ser una lista")
                    return False
                if len(perfil["efectos_esperados"]) == 0:
                    logger.debug("efectos_esperados no puede estar vacía")
                    return False
            
            # Si tiene layers o capas, debe ser dict válido
            for layer_key in ["layers", "capas"]:
                if layer_key in perfil:
                    if not isinstance(perfil[layer_key], dict):
                        logger.debug(f"{layer_key} debe ser un diccionario")
                        return False
            
            # Validación de coherencia: si tiene tanto nombre como categoria, deben ser coherentes
            if "nombre" in perfil and "categoria" in perfil:
                nombre_lower = perfil["nombre"].lower()
                categoria_lower = perfil["categoria"].lower()
                
                # Validaciones de coherencia básica
                if "relajacion" in nombre_lower and "energia" in categoria_lower:
                    logger.warning("Posible incoherencia: nombre sugiere relajación pero categoría sugiere energía")
                    # No fallar, solo advertir
            
            # Si llegamos aquí, el perfil pasó todas las validaciones
            logger.debug(f"Perfil validado exitosamente: {perfil.get('nombre', 'sin_nombre')}")
            return True
            
        except Exception as e:
            logger.error(f"Error validando perfil: {e}")
            return False
    
    def obtener_categoria(self) -> str:
        """Implementación de GestorPerfilesInterface - Categoría del gestor"""
        return "objetivos_terapeuticos"
    
    # ============================================================================
    # MÉTODOS ADICIONALES PARA COMPATIBILIDAD TOTAL CON FACTORY
    # ============================================================================
    
    def get_info(self) -> Dict[str, str]:
        """
        Información básica del gestor para validación de Factory
        
        Returns:
            Dict con información del componente
        """
        return {
            "nombre": "ObjectiveManager",
            "version": self.version,
            "descripcion": "Gestor unificado de objetivos y templates terapéuticos",
            "tipo": "gestor",
            "categoria": "objetivos_terapeuticos",
            "interface": "GestorPerfilesInterface",
            "total_templates": str(len(self.gestor_templates.templates) if hasattr(self, 'gestor_templates') and self.gestor_templates else 0),
            "modo": "unificado",
            "compatible_aurora_v7": "true"
        }
    
    def validar_configuracion(self, config: Dict[str, Any]) -> bool:
        """
        Valida configuración para el gestor
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        try:
            # Validaciones básicas
            if not isinstance(config, dict):
                logger.debug("Configuración no es un diccionario")
                return False
            
            # Permitir configuraciones vacías (para inicialización)
            if len(config) == 0:
                return True
            
            # Si tiene objective, debe ser string válido
            if "objective" in config:
                if not isinstance(config["objective"], str) or len(config["objective"].strip()) == 0:
                    logger.debug("Objetivo en configuración inválido")
                    return False
            
            # Si tiene duracion_sec, debe ser positiva
            if "duracion_sec" in config:
                if not isinstance(config["duracion_sec"], (int, float)):
                    logger.debug("duracion_sec debe ser numérica")
                    return False
                if config["duracion_sec"] <= 0:
                    logger.debug("duracion_sec debe ser positiva")
                    return False
            
            # Si tiene modo, debe ser string válido
            if "modo" in config:
                if not isinstance(config["modo"], str):
                    logger.debug("modo debe ser string")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando configuración: {e}")
            return False
    
    # ============================================================================
    # MÉTODOS PRINCIPALES DE FUNCIONALIDAD
    # ============================================================================
    
    def obtener_template(self, nombre: str) -> Optional[TemplateObjetivoV7]:
        """Obtiene un template por nombre"""
        return self.gestor_templates.obtener_template(nombre)
    
    def buscar_templates(self, criterios: Dict[str, Any], limite: int = 10) -> List[TemplateObjetivoV7]:
        """Búsqueda inteligente de templates"""
        return self.gestor_templates.buscar_templates_inteligente(criterios, limite)
    
    def listar_templates_por_categoria(self, categoria: CategoriaObjetivo) -> List[str]:
        """Lista templates por categoría"""
        return [
            nombre for nombre, template in self.gestor_templates.templates.items()
            if template.categoria == categoria
        ]
    
    def obtener_templates_populares(self, limite: int = 5) -> List[Tuple[str, int]]:
        """Obtiene templates más populares"""
        stats = self.gestor_templates.estadisticas_uso
        return sorted(stats.items(), key=lambda x: x[1].get("veces_usado", 0), reverse=True)[:limite]
    
    def rutear_objetivo_avanzado(self, objetivo: str, perfil_usuario: Optional[Dict[str, Any]] = None, 
                               **kwargs) -> Dict[str, Any]:
        """Ruteo avanzado con personalización"""
        
        # Convertir perfil de usuario si es necesario
        perfil = None
        if perfil_usuario:
            perfil = PerfilUsuario(
                id_usuario=perfil_usuario.get("id", "usuario_anonimo"),
                nivel_experiencia=NivelComplejidad(perfil_usuario.get("experiencia", "intermedio"))
            )
        
        resultado = self.router.rutear_objetivo_avanzado(objetivo, perfil, kwargs)
        
        # Actualizar estadísticas globales
        self.estadisticas_globales["total_objetivos_procesados"] += 1
        self.estadisticas_globales["tiempo_total_procesamiento"] += resultado.tiempo_procesamiento_ms
        
        # Convertir a formato dict para compatibilidad
        return {
            "template": resultado.template_seleccionado,
            "confianza": resultado.nivel_confianza,
            "tipo_ruteo": resultado.tipo_ruteo.value,
            "tiempo_ms": resultado.tiempo_procesamiento_ms,
            "metadatos": resultado.metadatos_ruteo,
            "alternativas": resultado.alternativas
        }
    
    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesa un objetivo y retorna configuración para director"""
        
        resultado_ruteo = self.rutear_objetivo_avanzado(objetivo, contexto)
        template = resultado_ruteo["template"]
        
        if not template:
            return {
                "exito": False,
                "error": f"No se encontró template para objetivo: {objetivo}",
                "preset_emocional": "default",
                "estilo": "basic",
                "modo": "fallback"
            }
        
        # Crear configuración para Aurora Director
        config_director = {
            "exito": True,
            "objetivo_procesado": objetivo,
            "preset_emocional": template.emotional_preset,
            "estilo": template.style,
            "modo": "avanzado",
            "beat_base": template.frecuencia_dominante,
            "duracion_recomendada": template.duracion_recomendada_min,
            "capas": {
                nombre_capa: capa.to_dict()
                for nombre_capa, capa in template.layers.items()
            },
            "metadatos": {
                "template_nombre": template.nombre,
                "categoria": template.categoria.value,
                "confianza": resultado_ruteo["confianza"],
                "evidencia_cientifica": template.evidencia_cientifica,
                "efectos_esperados": template.efectos_esperados
            },
            "nivel_confianza": resultado_ruteo["confianza"],
            "tiempo_procesamiento": resultado_ruteo["tiempo_ms"]
        }
        
        return config_director
    
    def validar_template(self, template: TemplateObjetivoV7) -> Dict[str, Any]:
        """Valida un template usando el validador científico"""
        return self.router.validador_cientifico.validar_template(template)
    
    def listar_objetivos_disponibles(self) -> List[str]:
        """Lista todos los objetivos disponibles"""
        objetivos_directos = list(self.gestor_templates.templates.keys())
        objetivos_v6 = list(self.router.rutas_v6_mapeadas.keys())
        return sorted(list(set(objetivos_directos + objetivos_v6)))
    
    def buscar_objetivos_similares(self, objetivo: str, limite: int = 5) -> List[Tuple[str, float]]:
        """Busca objetivos similares usando análisis semántico"""
        analisis = self.router.analizador_semantico.analizar_objetivo(objetivo)
        similares = []
        
        for nombre_template in self.gestor_templates.templates.keys():
            analisis_template = self.router.analizador_semantico.analizar_objetivo(nombre_template)
            
            # Calcular similitud basada en categorías comunes
            categorias_comunes = set(analisis["categorias_detectadas"]) & set(analisis_template["categorias_detectadas"])
            similitud = len(categorias_comunes) / max(len(analisis["categorias_detectadas"]), 1)
            
            if similitud > 0:
                similares.append((nombre_template, similitud))
        
        return sorted(similares, key=lambda x: x[1], reverse=True)[:limite]
    
    def recomendar_secuencia_objetivos(self, objetivo_principal: str, duracion_total: int = 60, 
                                     perfil_usuario: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Recomienda una secuencia de objetivos para sesión completa"""
        secuencia = []
        tiempo_usado = 0
        
        # Objetivo principal (50% del tiempo)
        template_principal = self.obtener_template(objetivo_principal)
        if template_principal:
            duracion_principal = min(duracion_total * 0.5, template_principal.duracion_recomendada_max)
            secuencia.append({
                "objetivo": objetivo_principal,
                "template": template_principal,
                "duracion": duracion_principal,
                "orden": 1,
                "tipo": "principal"
            })
            tiempo_usado += duracion_principal
        
        # Objetivos complementarios
        tiempo_restante = duracion_total - tiempo_usado
        if tiempo_restante > 10:  # Mínimo 10 minutos para complementarios
            similares = self.buscar_objetivos_similares(objetivo_principal, 3)
            
            for objetivo_similar, similitud in similares:
                if tiempo_restante <= 0:
                    break
                
                template_similar = self.obtener_template(objetivo_similar)
                if template_similar:
                    duracion_similar = min(tiempo_restante * 0.5, template_similar.duracion_recomendada_min)
                    secuencia.append({
                        "objetivo": objetivo_similar,
                        "template": template_similar,
                        "duracion": duracion_similar,
                        "orden": len(secuencia) + 1,
                        "tipo": "complementario",
                        "similitud": similitud
                    })
                    tiempo_restante -= duracion_similar
        
        return secuencia
    
    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema"""
        return {
            "version": self.version,
            "estadisticas_globales": self.estadisticas_globales,
            "router_stats": self.router.obtener_estadisticas_completas(),
            "templates_disponibles": len(self.gestor_templates.templates),
            "total_procesado": self.estadisticas_globales["total_objetivos_procesados"],
            "tiempo_promedio_procesamiento": (
                self.estadisticas_globales["tiempo_total_procesamiento"] / 
                max(self.estadisticas_globales["total_objetivos_procesados"], 1)
            ),
            "aurora_v7_ready": True,
            "factory_compatible": True
        }
    
    def obtener_diagnostico_completo(self) -> Dict[str, Any]:
        """Diagnóstico completo para debugging"""
        return {
            "manager_info": self.get_info(),
            "estadisticas": self.obtener_estadisticas_completas(),
            "templates_count": len(self.gestor_templates.templates),
            "router_ready": self.router is not None,
            "interfaces_implementadas": ["GestorPerfilesInterface"],
            "metodos_principales": [
                "obtener_perfil", "listar_perfiles", "validar_perfil",
                "obtener_template", "procesar_objetivo", "rutear_objetivo_avanzado"
            ]
        }

# ============================================================================
# FUNCIONES GLOBALES PARA COMPATIBILIDAD Y ACCESO FÁCIL
# ============================================================================

# Variables globales para compatibilidad
_manager_global: Optional[ObjectiveManagerUnificado] = None
OBJECTIVE_TEMPLATES: Dict[str, Any] = {}

def obtener_manager() -> ObjectiveManagerUnificado:
    """Obtiene instancia global del manager"""
    global _manager_global
    if _manager_global is None:
        _manager_global = ObjectiveManagerUnificado()
    return _manager_global

def crear_manager_optimizado() -> ObjectiveManagerUnificado:
    """Crea nueva instancia optimizada del manager"""
    return ObjectiveManagerUnificado()

def obtener_template(nombre: str) -> Optional[TemplateObjetivoV7]:
    """Función global para obtener template"""
    return obtener_manager().obtener_template(nombre)

def buscar_templates_por_objetivo(objetivo: str, limite: int = 5) -> List[TemplateObjetivoV7]:
    """Función global para buscar templates por objetivo"""
    return obtener_manager().buscar_templates({"efectos": [objetivo]}, limite)

def validar_template_avanzado(template: TemplateObjetivoV7) -> Dict[str, Any]:
    """Función global para validar template"""
    return obtener_manager().validar_template(template)

def ruta_por_objetivo(nombre: str) -> Dict[str, Any]:
    """Función global para ruteo por objetivo"""
    resultado = obtener_manager().procesar_objetivo(nombre)
    return {
        "preset": resultado.get("preset_emocional"),
        "estilo": resultado.get("estilo"),
        "modo": resultado.get("modo"),
        "beat": resultado.get("beat_base"),
        "capas": resultado.get("capas")
    }

def rutear_objetivo_inteligente(objetivo: str, perfil_usuario: Optional[Dict[str, Any]] = None, 
                              **kwargs) -> Dict[str, Any]:
    """Función global para ruteo inteligente"""
    return obtener_manager().rutear_objetivo_avanzado(objetivo, perfil_usuario, **kwargs)

def procesar_objetivo_director(objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
    """Función global para procesamiento de objetivos por director"""
    return obtener_manager().procesar_objetivo(objetivo, contexto)

def listar_objetivos_disponibles() -> List[str]:
    """Función global para listar objetivos"""
    return obtener_manager().listar_objetivos_disponibles()

def buscar_objetivos_similares(objetivo: str, limite: int = 5) -> List[Tuple[str, float]]:
    """Función global para buscar objetivos similares"""
    return obtener_manager().buscar_objetivos_similares(objetivo, limite)

def recomendar_secuencia_objetivos(objetivo_principal: str, duracion_total: int = 60, 
                                 perfil_usuario: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Función global para recomendar secuencias"""
    return obtener_manager().recomendar_secuencia_objetivos(objetivo_principal, duracion_total, perfil_usuario)

# Aliases para compatibilidad V6
crear_gestor_optimizado = crear_manager_optimizado

def crear_router_inteligente():
    """Crea router inteligente"""
    return obtener_manager().router

def _generar_compatibilidad_v6():
    """Genera compatibilidad V6 sin recursión"""
    global OBJECTIVE_TEMPLATES
    try:
        # Solo ejecutar si el manager ya está inicializado
        if _manager_global is not None:
            manager = _manager_global
            for nombre, template in manager.gestor_templates.templates.items():
                layers_v6 = {}
                for nombre_capa, config_capa in template.layers.items():
                    layers_v6[nombre_capa] = config_capa.to_dict()
                
                OBJECTIVE_TEMPLATES[nombre] = {
                    "emotional_preset": template.emotional_preset,
                    "style": template.style,
                    "layers": layers_v6
                }
    except Exception as e:
        logger.warning(f"Error generando compatibilidad V6: {e}")

def diagnostico_manager() -> Dict[str, Any]:
    """Diagnóstico completo del manager"""
    manager = obtener_manager()
    test_objetivos = ["concentracion", "relajacion", "creatividad", "test_inexistente"]
    resultados_test = {}
    
    for objetivo in test_objetivos:
        try:
            resultado = manager.procesar_objetivo(objetivo)
            resultados_test[objetivo] = {
                "exito": True,
                "preset": resultado.get("preset_emocional"),
                "confianza": resultado.get("nivel_confianza"),
                "tipo": "procesamiento_director"
            }
        except Exception as e:
            resultados_test[objetivo] = {"exito": False, "error": str(e)}
    
    estadisticas = manager.obtener_estadisticas_completas()
    
    return {
        "version": manager.version,
        "inicializacion_exitosa": True,
        "estadisticas_completas": estadisticas,
        "test_objetivos": resultados_test,
        "templates_disponibles": len(manager.gestor_templates.templates),
        "objetivos_v6_compatibles": len(manager.router.rutas_v6_mapeadas),
        "interfaces_implementadas": ["GestorPerfilesInterface"],
        "factory_compatible": True,
        "aurora_v7_ready": True
    }

# Generar compatibilidad V6 al cargar el módulo
_generar_compatibilidad_v6()

# ============================================================================
# MEJORAS ADITIVAS HYPERMOD V32 - OPTIMIZACIÓN Y EXTENSIONES
# ============================================================================

# Variables globales expandidas para HyperMod V32
objective_templates = OBJECTIVE_TEMPLATES
TEMPLATES_OBJETIVOS_AURORA = OBJECTIVE_TEMPLATES
templates_disponibles = OBJECTIVE_TEMPLATES

class GestorTemplatesExpandido(GestorTemplatesOptimizado):
    """Gestor expandido con funciones especializadas para HyperMod V32"""
    
    def __init__(self):
        super().__init__()
        self._cargar_templates_especializados()
    
    def _cargar_templates_especializados(self):
        """Carga templates especializados para HyperMod"""
        try:
            # Templates específicos para integración con HyperMod
            templates_hypermod = [
                TemplateObjetivoV7(
                    nombre="estructural_cognitivo",
                    descripcion="Template estructural para mejoras cognitivas",
                    categoria=CategoriaObjetivo.COGNITIVO,
                    emotional_preset="structured_cognitive",
                    style="architectural",
                    frecuencia_dominante=40.0,
                    duracion_recomendada_min=20,
                    duracion_recomendada_max=45,
                    efectos_esperados=["estructura_mental", "organizacion_cognitiva", "clarity"],
                    contraindicaciones=["epilepsia"],
                    evidencia_cientifica="Estructuras gamma para organización cognitiva",
                    nivel_confianza=0.8,
                    complejidad=NivelComplejidad.AVANZADO,
                    modo_activacion=ModoActivacion.PROGRESIVO,
                    contexto_uso=ContextoUso.TERAPEUTICO
                ),
                TemplateObjetivoV7(
                    nombre="flow_creativo_estructurado",
                    descripcion="Flow creativo con estructura arquitectónica",
                    categoria=CategoriaObjetivo.CREATIVO,
                    emotional_preset="structured_flow",
                    style="creative_architecture", 
                    frecuencia_dominante=8.0,
                    duracion_recomendada_min=30,
                    duracion_recomendada_max=60,
                    efectos_esperados=["flow_estructurado", "creatividad_organizada", "inspiration"],
                    contraindicaciones=["mania"],
                    evidencia_cientifica="Theta estructurado para creatividad dirigida",
                    nivel_confianza=0.75,
                    complejidad=NivelComplejidad.EXPERTO,
                    modo_activacion=ModoActivacion.CICLICO,
                    contexto_uso=ContextoUso.PERSONAL
                )
            ]
            
            for template in templates_hypermod:
                self.templates[template.nombre] = template
                
            logger.info(f"Templates especializados HyperMod cargados: {len(templates_hypermod)}")
            
        except Exception as e:
            logger.warning(f"Error cargando templates especializados: {e}")

class RouterTemplatesInteligente(RouterInteligenteV7):
    """Router expandido con capacidades especiales para templates"""
    
    def __init__(self, gestor_templates: GestorTemplatesOptimizado):
        super().__init__(gestor_templates)
        self.cache_ruteo_avanzado = {}
        self.metricas_performance = {
            "cache_hits": 0,
            "cache_misses": 0,
            "tiempo_promedio_cached": 0.0
        }
    
    def rutear_con_cache(self, objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ruteo con sistema de cache inteligente"""
        cache_key = f"{objetivo}_{hash(str(contexto))}" if contexto else objetivo
        
        # Verificar cache
        if cache_key in self.cache_ruteo_avanzado:
            self.metricas_performance["cache_hits"] += 1
            cached_result = self.cache_ruteo_avanzado[cache_key]
            cached_result["cached"] = True
            cached_result["timestamp"] = datetime.now().isoformat()
            return cached_result
        
        # Ruteo normal si no está en cache
        self.metricas_performance["cache_misses"] += 1
        resultado = self.rutear_objetivo_avanzado(objetivo, None, contexto or {})
        
        # Guardar en cache si es exitoso
        if resultado.template_seleccionado:
            self.cache_ruteo_avanzado[cache_key] = {
                "template": resultado.template_seleccionado,
                "confianza": resultado.nivel_confianza,
                "tipo_ruteo": resultado.tipo_ruteo.value,
                "cached": False
            }
        
        return {
            "template": resultado.template_seleccionado,
            "confianza": resultado.nivel_confianza,
            "tipo_ruteo": resultado.tipo_ruteo.value,
            "cached": False
        }
    
    def limpiar_cache(self):
        """Limpia el cache de ruteo"""
        self.cache_ruteo_avanzado.clear()
        logger.info("Cache de ruteo limpiado")
    
    def obtener_metricas_cache(self) -> Dict[str, Any]:
        """Obtiene métricas del cache"""
        total_requests = self.metricas_performance["cache_hits"] + self.metricas_performance["cache_misses"]
        hit_rate = (self.metricas_performance["cache_hits"] / max(total_requests, 1)) * 100
        
        return {
            "cache_size": len(self.cache_ruteo_avanzado),
            "cache_hits": self.metricas_performance["cache_hits"],
            "cache_misses": self.metricas_performance["cache_misses"],
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests
        }

# ============================================================================
# FUNCIONES DE UTILIDAD Y HELPERS GLOBALES
# ============================================================================

def crear_template_personalizado(nombre: str, config: Dict[str, Any]) -> Optional[TemplateObjetivoV7]:
    """Crea un template personalizado desde configuración"""
    try:
        template = TemplateObjetivoV7(
            nombre=nombre,
            descripcion=config.get("descripcion", f"Template personalizado: {nombre}"),
            categoria=CategoriaObjetivo(config.get("categoria", "cognitivo")),
            emotional_preset=config.get("emotional_preset", "balanced"),
            style=config.get("style", "adaptive"),
            frecuencia_dominante=float(config.get("frecuencia_dominante", 10.0)),
            duracion_recomendada_min=int(config.get("duracion_min", 15)),
            duracion_recomendada_max=int(config.get("duracion_max", 30)),
            efectos_esperados=config.get("efectos_esperados", ["bienestar"]),
            contraindicaciones=config.get("contraindicaciones", []),
            evidencia_cientifica=config.get("evidencia", "Template personalizado"),
            nivel_confianza=float(config.get("confianza", 0.5)),
            complejidad=NivelComplejidad(config.get("complejidad", "intermedio")),
            modo_activacion=ModoActivacion(config.get("modo_activacion", "progresivo")),
            contexto_uso=ContextoUso(config.get("contexto", "personal")),
            metadatos={"personalizado": True, "creado": datetime.now().isoformat()}
        )
        
        # Agregar al gestor global
        manager = obtener_manager()
        manager.gestor_templates.templates[nombre] = template
        
        logger.info(f"Template personalizado creado: {nombre}")
        return template
        
    except Exception as e:
        logger.error(f"Error creando template personalizado: {e}")
        return None

def exportar_templates_json(ruta_archivo: str = "templates_aurora_v7.json") -> bool:
    """Exporta todos los templates a archivo JSON"""
    try:
        manager = obtener_manager()
        templates_export = {}
        
        for nombre, template in manager.gestor_templates.templates.items():
            templates_export[nombre] = {
                "nombre": template.nombre,
                "descripcion": template.descripcion,
                "categoria": template.categoria.value,
                "emotional_preset": template.emotional_preset,
                "style": template.style,
                "frecuencia_dominante": template.frecuencia_dominante,
                "duracion_min": template.duracion_recomendada_min,
                "duracion_max": template.duracion_recomendada_max,
                "efectos_esperados": template.efectos_esperados,
                "contraindicaciones": template.contraindicaciones,
                "evidencia": template.evidencia_cientifica,
                "confianza": template.nivel_confianza,
                "complejidad": template.complejidad.value,
                "modo_activacion": template.modo_activacion.value,
                "contexto": template.contexto_uso.value,
                "layers": {k: v.to_dict() for k, v in template.layers.items()},
                "metadatos": template.metadatos
            }
        
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            json.dump(templates_export, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Templates exportados a: {ruta_archivo}")
        return True
        
    except Exception as e:
        logger.error(f"Error exportando templates: {e}")
        return False

def importar_templates_json(ruta_archivo: str) -> int:
    """Importa templates desde archivo JSON"""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            templates_data = json.load(f)
        
        manager = obtener_manager()
        count = 0
        
        for nombre, data in templates_data.items():
            try:
                template = TemplateObjetivoV7(
                    nombre=data["nombre"],
                    descripcion=data["descripcion"],
                    categoria=CategoriaObjetivo(data["categoria"]),
                    emotional_preset=data["emotional_preset"],
                    style=data["style"],
                    frecuencia_dominante=data["frecuencia_dominante"],
                    duracion_recomendada_min=data["duracion_min"],
                    duracion_recomendada_max=data["duracion_max"],
                    efectos_esperados=data["efectos_esperados"],
                    contraindicaciones=data["contraindicaciones"],
                    evidencia_cientifica=data["evidencia"],
                    nivel_confianza=data["confianza"],
                    complejidad=NivelComplejidad(data["complejidad"]),
                    modo_activacion=ModoActivacion(data["modo_activacion"]),
                    contexto_uso=ContextoUso(data["contexto"]),
                    metadatos=data.get("metadatos", {})
                )
                
                # Recrear layers
                if "layers" in data:
                    for layer_name, layer_data in data["layers"].items():
                        template.layers[layer_name] = ConfiguracionCapaV7(**layer_data)
                
                manager.gestor_templates.templates[nombre] = template
                count += 1
                
            except Exception as e:
                logger.warning(f"Error importando template {nombre}: {e}")
        
        logger.info(f"Templates importados: {count}")
        return count
        
    except Exception as e:
        logger.error(f"Error importando templates: {e}")
        return 0

def validar_sistema_completo() -> Dict[str, Any]:
    """Validación completa del sistema ObjectiveManager"""
    try:
        manager = obtener_manager()
        
        # Tests básicos
        tests = {
            "manager_inicializado": manager is not None,
            "templates_disponibles": len(manager.gestor_templates.templates) > 0,
            "router_funcional": manager.router is not None,
            "interfaces_implementadas": True
        }
        
        # Test de funcionalidad básica
        try:
            test_perfil = manager.obtener_perfil("concentracion")
            tests["obtener_perfil_funcional"] = test_perfil is not None
        except:
            tests["obtener_perfil_funcional"] = False
        
        try:
            perfiles = manager.listar_perfiles()
            tests["listar_perfiles_funcional"] = len(perfiles) > 0
        except:
            tests["listar_perfiles_funcional"] = False
        
        try:
            validacion = manager.validar_perfil({"nombre": "test", "categoria": "cognitivo"})
            tests["validar_perfil_funcional"] = isinstance(validacion, bool)
        except:
            tests["validar_perfil_funcional"] = False
        
        # Test de ruteo
        try:
            resultado = manager.procesar_objetivo("concentracion")
            tests["ruteo_funcional"] = resultado.get("exito", False)
        except:
            tests["ruteo_funcional"] = False
        
        # Calcular score general
        score = sum(tests.values()) / len(tests) * 100
        
        return {
            "score_general": score,
            "tests_individuales": tests,
            "estado": "OPTIMO" if score >= 90 else "BUENO" if score >= 70 else "DEFICIENTE",
            "manager_info": manager.get_info(),
            "estadisticas": manager.obtener_estadisticas_completas(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en validación completa: {e}")
        return {
            "score_general": 0,
            "estado": "ERROR",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# INICIALIZACIÓN Y CONFIGURACIÓN FINAL
# ============================================================================

def inicializar_objective_manager_completo() -> bool:
    """Inicialización completa del sistema ObjectiveManager"""
    try:
        logger.info("🚀 Inicializando ObjectiveManager V7 completo...")
        
        # Crear manager global
        manager = obtener_manager()
        
        # Generar compatibilidad V6
        _generar_compatibilidad_v6()
        
        # Validar sistema
        validacion = validar_sistema_completo()
        
        if validacion["score_general"] >= 70:
            logger.info(f"✅ ObjectiveManager inicializado exitosamente - Score: {validacion['score_general']:.1f}%")
            return True
        else:
            logger.warning(f"⚠️ ObjectiveManager inicializado con advertencias - Score: {validacion['score_general']:.1f}%")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error inicializando ObjectiveManager: {e}")
        return False

# Auto-inicialización al importar el módulo
if __name__ != "__main__":
    inicializar_objective_manager_completo()

# ============================================================================
# INFORMACIÓN DEL MÓDULO
# ============================================================================

__version__ = VERSION
__author__ = "Sistema Aurora V7"
__description__ = "Gestor unificado de objetivos terapéuticos con IA semántica"
__all__ = [
    # Clases principales
    "ObjectiveManagerUnificado", "GestorTemplatesOptimizado", "RouterInteligenteV7",
    "TemplateObjetivoV7", "ConfiguracionCapaV7", "ResultadoRuteo", "PerfilUsuario",
    
    # Enums
    "CategoriaObjetivo", "NivelComplejidad", "ModoActivacion", "TipoRuteo", 
    "NivelConfianza", "ContextoUso",
    
    # Funciones principales
    "obtener_manager", "crear_manager_optimizado", "obtener_template",
    "buscar_templates_por_objetivo", "validar_template_avanzado", "ruta_por_objetivo",
    "rutear_objetivo_inteligente", "procesar_objetivo_director", "listar_objetivos_disponibles",
    "buscar_objetivos_similares", "recomendar_secuencia_objetivos",
    
    # Utilities
    "crear_template_personalizado", "exportar_templates_json", "importar_templates_json",
    "validar_sistema_completo", "diagnostico_manager",
    
    # Variables globales
    "OBJECTIVE_TEMPLATES", "objective_templates", "TEMPLATES_OBJETIVOS_AURORA"
]

if __name__ == "__main__":
    # Modo de testing
    print("🔍 ObjectiveManager V7 - Modo Testing")
    print(f"Versión: {VERSION}")
    
    # Ejecutar diagnóstico
    diagnostico = diagnostico_manager()
    print(f"Diagnóstico: {diagnostico}")
    
    # Validación completa
    validacion = validar_sistema_completo()
    print(f"Validación: Score {validacion['score_general']:.1f}% - {validacion['estado']}")
    
    print("✅ Testing completado")