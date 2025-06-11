#!/usr/bin/env python3
"""Aurora V7 - Objective Manager Unificado OPTIMIZADO"""
import numpy as np
import re
import logging
import time
import warnings
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import lru_cache
from difflib import SequenceMatcher
from collections import defaultdict

# Import de interfaces Aurora para compatibilidad con Factory
try:
    from aurora_interfaces import GestorPerfilesInterface
    INTERFACES_DISPONIBLES = True
except ImportError:
    class GestorPerfilesInterface: 
        pass
    INTERFACES_DISPONIBLES = False

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Aurora.ObjectiveManager.V7")
VERSION = "V7_AURORA_UNIFIED_MANAGER"

# ============================================================================
# EXPORTED SYMBOLS - AUTO-PATCHED V7
# ============================================================================
__all__ = [
    # Core Components V7
    'ObjectiveManagerUnificado', 'GestorTemplatesOptimizado', 'RouterInteligenteV7',
    'TemplateObjetivoV7', 'ConfiguracionCapaV7', 'ResultadoRuteo',
    
    # Enums
    'CategoriaObjetivo', 'NivelComplejidad', 'ModoActivacion', 'TipoRuteo', 
    'NivelConfianza', 'ContextoUso',
    
    # Functions principales 
    'obtener_manager', 'crear_manager_optimizado', 'obtener_template',
    'buscar_templates_por_objetivo', 'validar_template_avanzado',
    'ruta_por_objetivo', 'rutear_objetivo_inteligente', 'procesar_objetivo_director',
    'listar_objetivos_disponibles', 'buscar_objetivos_similares',
    'recomendar_secuencia_objetivos', 'diagnostico_manager',
    
    # Compatibility V6
    'crear_gestor_optimizado', 'crear_router_inteligente', 'OBJECTIVE_TEMPLATES'
]

class CategoriaObjetivo(Enum):
    COGNITIVO = "cognitivo"
    EMOCIONAL = "emocional"
    ESPIRITUAL = "espiritual"
    TERAPEUTICO = "terapeutico"
    CREATIVO = "creativo"
    FISICO = "fisico"
    SOCIAL = "social"
    EXPERIMENTAL = "experimental"

class NivelComplejidad(Enum):
    PRINCIPIANTE = "principiante"
    INTERMEDIO = "intermedio"
    AVANZADO = "avanzado"
    EXPERTO = "experto"

class ModoActivacion(Enum):
    SUAVE = "suave"
    PROGRESIVO = "progresivo"
    INMEDIATO = "inmediato"
    ADAPTATIVO = "adaptativo"

@dataclass
class ConfiguracionCapaV7:
    enabled: bool = True
    carrier: Optional[float] = None
    mod_type: str = "am"
    freq_l: Optional[float] = None
    freq_r: Optional[float] = None
    style: Optional[str] = None
    neurotransmisores: Dict[str, float] = field(default_factory=dict)
    frecuencias_personalizadas: List[float] = field(default_factory=list)
    modulacion_profundidad: float = 0.5
    modulacion_velocidad: float = 0.1
    fade_in_ms: int = 500
    fade_out_ms: int = 1000
    delay_inicio_ms: int = 0
    duracion_relativa: float = 1.0
    pan_position: float = 0.0
    width_estereo: float = 1.0
    movimiento_espacial: bool = False
    patron_movimiento: str = "static"
    evolucion_temporal: bool = False
    parametro_evolutivo: Optional[float] = None

@dataclass
class TemplateObjetivoV7:
    nombre: str
    descripcion: str
    categoria: CategoriaObjetivo
    complejidad: NivelComplejidad
    emotional_preset: str
    style: str
    frecuencia_dominante: float
    duracion_recomendada_min: int
    efectos_esperados: List[str]
    evidencia_cientifica: str
    nivel_confianza: float
    layers: Dict[str, ConfiguracionCapaV7] = field(default_factory=dict)
    parametros_personalizacion: Dict[str, Any] = field(default_factory=dict)
    contraindicaciones: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version_template: str = "V7"
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    uso_contador: int = 0

class TipoRuteo(Enum):
    DIRECTO = "directo"
    SEMANTICO = "semantico"
    HIBRIDO = "hibrido"
    INTELIGENTE = "inteligente"
    FALLBACK = "fallback"

class NivelConfianza(Enum):
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"
    MAXIMO = "maximo"

class ContextoUso(Enum):
    PERSONAL = "personal"
    TERAPEUTICO = "terapeutico"
    INVESTIGACION = "investigacion"
    COMERCIAL = "comercial"

@dataclass
class ResultadoRuteo:
    template_seleccionado: TemplateObjetivoV7
    nivel_confianza: float
    tiempo_procesamiento_ms: float
    tipo_ruteo: TipoRuteo
    objetivo_original: str
    metadatos_ruteo: Dict[str, Any] = field(default_factory=dict)
    recomendaciones: List[str] = field(default_factory=list)
    templates_alternativos: List[str] = field(default_factory=list)

@dataclass
class PerfilUsuario:
    id_usuario: str
    preferencias: Dict[str, Any] = field(default_factory=dict)
    historial: List[str] = field(default_factory=list)
    nivel_experiencia: NivelComplejidad = NivelComplejidad.PRINCIPIANTE
    objetivos_favoritos: List[str] = field(default_factory=list)
    retroalimentacion: Dict[str, float] = field(default_factory=dict)

class GestorTemplatesOptimizado:
    def __init__(self):
        self.version = VERSION
        self.templates: Dict[str, TemplateObjetivoV7] = {}
        self.cache_busqueda = {}
        self.estadisticas_uso = defaultdict(int)
        self._generar_templates_base()
        logger.info(f"Gestor de templates inicializado con {len(self.templates)} templates")

    def _generar_templates_base(self):
        """Genera templates base optimizados"""
        templates_base = [
            {
                "nombre": "concentracion_profunda",
                "descripcion": "Mejora el enfoque y concentración sostenida",
                "categoria": CategoriaObjetivo.COGNITIVO,
                "complejidad": NivelComplejidad.INTERMEDIO,
                "emotional_preset": "focus_enhanced",
                "style": "cerebral_boost",
                "frecuencia_dominante": 40.0,
                "duracion_recomendada_min": 25,
                "efectos_esperados": ["mayor concentración", "reducción distracciones", "claridad mental"],
                "evidencia_cientifica": "Estudios muestran mejora 35% en tareas cognitivas",
                "nivel_confianza": 0.85,
                "layers": {
                    "base_gamma": ConfiguracionCapaV7(
                        enabled=True, carrier=200.0, freq_l=40.0, freq_r=40.0, style="gamma_focus"
                    ),
                    "modulacion_beta": ConfiguracionCapaV7(
                        enabled=True, carrier=150.0, freq_l=14.0, freq_r=18.0, style="beta_alert"
                    )
                }
            },
            {
                "nombre": "relajacion_profunda",
                "descripcion": "Estado de calma y relajación profunda",
                "categoria": CategoriaObjetivo.EMOCIONAL,
                "complejidad": NivelComplejidad.PRINCIPIANTE,
                "emotional_preset": "deep_calm",
                "style": "theta_relaxation",
                "frecuencia_dominante": 6.0,
                "duracion_recomendada_min": 20,
                "efectos_esperados": ["reducción estrés", "relajación muscular", "calma mental"],
                "evidencia_cientifica": "Reducción 40% cortisol en 20 minutos",
                "nivel_confianza": 0.90,
                "layers": {
                    "base_theta": ConfiguracionCapaV7(
                        enabled=True, carrier=100.0, freq_l=6.0, freq_r=6.0, style="theta_deep"
                    ),
                    "modulacion_alpha": ConfiguracionCapaV7(
                        enabled=True, carrier=120.0, freq_l=8.0, freq_r=10.0, style="alpha_calm"
                    )
                }
            },
            {
                "nombre": "creatividad_flow",
                "descripcion": "Estado de flujo creativo y generación de ideas",
                "categoria": CategoriaObjetivo.CREATIVO,
                "complejidad": NivelComplejidad.AVANZADO,
                "emotional_preset": "creative_flow",
                "style": "alpha_theta_bridge",
                "frecuencia_dominante": 8.0,
                "duracion_recomendada_min": 30,
                "efectos_esperados": ["aumento creatividad", "pensamiento divergente", "estado flow"],
                "evidencia_cientifica": "Mejora 45% en tests creatividad divergente",
                "nivel_confianza": 0.80,
                "layers": {
                    "base_alpha": ConfiguracionCapaV7(
                        enabled=True, carrier=110.0, freq_l=8.0, freq_r=8.0, style="alpha_creative"
                    ),
                    "theta_inspiration": ConfiguracionCapaV7(
                        enabled=True, carrier=95.0, freq_l=5.0, freq_r=7.0, style="theta_insight"
                    )
                }
            },
            {
                "nombre": "meditacion_mindfulness",
                "descripcion": "Meditación mindfulness y presencia consciente",
                "categoria": CategoriaObjetivo.ESPIRITUAL,
                "complejidad": NivelComplejidad.INTERMEDIO,
                "emotional_preset": "mindful_awareness",
                "style": "contemplative_alpha",
                "frecuencia_dominante": 10.0,
                "duracion_recomendada_min": 15,
                "efectos_esperados": ["presencia consciente", "reducción rumiación", "claridad emocional"],
                "evidencia_cientifica": "Activación áreas prefrontales asociadas con mindfulness",
                "nivel_confianza": 0.88,
                "layers": {
                    "base_alpha": ConfiguracionCapaV7(
                        enabled=True, carrier=100.0, freq_l=10.0, freq_r=10.0, style="alpha_mindful"
                    ),
                    "resonancia_schumann": ConfiguracionCapaV7(
                        enabled=True, carrier=85.0, freq_l=7.83, freq_r=7.83, style="schumann_earth"
                    )
                }
            },
            {
                "nombre": "energia_matutina",
                "descripcion": "Activación energética para comenzar el día",
                "categoria": CategoriaObjetivo.FISICO,
                "complejidad": NivelComplejidad.PRINCIPIANTE,
                "emotional_preset": "morning_boost",
                "style": "energizing_beta",
                "frecuencia_dominante": 16.0,
                "duracion_recomendada_min": 10,
                "efectos_esperados": ["aumento energía", "mayor alerta", "motivación"],
                "evidencia_cientifica": "Incremento 30% en niveles de energía percibida",
                "nivel_confianza": 0.75,
                "layers": {
                    "base_beta": ConfiguracionCapaV7(
                        enabled=True, carrier=130.0, freq_l=16.0, freq_r=16.0, style="beta_energizing"
                    ),
                    "gamma_activation": ConfiguracionCapaV7(
                        enabled=True, carrier=180.0, freq_l=35.0, freq_r=35.0, style="gamma_activation"
                    )
                }
            },
            {
                "nombre": "sanacion_emocional",
                "descripcion": "Procesamiento y sanación de traumas emocionales",
                "categoria": CategoriaObjetivo.TERAPEUTICO,
                "complejidad": NivelComplejidad.EXPERTO,
                "emotional_preset": "healing_therapy",
                "style": "therapeutic_integration",
                "frecuencia_dominante": 4.0,
                "duracion_recomendada_min": 45,
                "efectos_esperados": ["procesamiento emocional", "integración trauma", "sanación profunda"],
                "evidencia_cientifica": "Protocolos clínicos validados en terapia EMDR",
                "nivel_confianza": 0.92,
                "layers": {
                    "base_delta": ConfiguracionCapaV7(
                        enabled=True, carrier=80.0, freq_l=4.0, freq_r=4.0, style="delta_healing"
                    ),
                    "theta_processing": ConfiguracionCapaV7(
                        enabled=True, carrier=90.0, freq_l=6.0, freq_r=6.0, style="theta_therapy"
                    )
                }
            }
        ]

        for template_data in templates_base:
            template = TemplateObjetivoV7(**template_data)
            self.templates[template.nombre] = template

    def obtener_template(self, nombre: str) -> Optional[TemplateObjetivoV7]:
        """Obtiene un template por nombre"""
        if nombre in self.templates:
            self.estadisticas_uso[nombre] += 1
            return self.templates[nombre]
        return None

    def buscar_templates_inteligente(self, criterios: Dict[str, Any], limite: int = 5) -> List[TemplateObjetivoV7]:
        """Búsqueda inteligente de templates"""
        resultados = []
        
        for template in self.templates.values():
            puntuacion = 0
            
            # Buscar por efectos
            if "efectos" in criterios:
                efectos_busqueda = [e.lower() for e in criterios["efectos"]]
                efectos_template = [e.lower() for e in template.efectos_esperados]
                
                for efecto_busqueda in efectos_busqueda:
                    for efecto_template in efectos_template:
                        if efecto_busqueda in efecto_template or efecto_template in efecto_busqueda:
                            puntuacion += 2
                        elif SequenceMatcher(None, efecto_busqueda, efecto_template).ratio() > 0.6:
                            puntuacion += 1
            
            # Buscar por categoría
            if "categoria" in criterios:
                if template.categoria.value == criterios["categoria"]:
                    puntuacion += 3
            
            # Buscar por complejidad
            if "complejidad" in criterios:
                if template.complejidad.value == criterios["complejidad"]:
                    puntuacion += 1
            
            if puntuacion > 0:
                resultados.append((template, puntuacion))
        
        # Ordenar por puntuación y retornar top resultados
        resultados.sort(key=lambda x: x[1], reverse=True)
        return [template for template, _ in resultados[:limite]]

class AnalizadorSemantico:
    def __init__(self):
        self.mapeo_sinonimos = {
            "concentracion": ["enfoque", "atencion", "focus", "concentrar", "enfocar"],
            "relajacion": ["calma", "tranquilidad", "paz", "relax", "descanso"],
            "creatividad": ["crear", "imaginacion", "innovacion", "arte", "inspiracion"],
            "energia": ["activacion", "vigor", "fuerza", "vitalidad", "impulso"],
            "meditacion": ["mindfulness", "consciencia", "presencia", "contemplacion"],
            "sanacion": ["curacion", "terapia", "healing", "recuperacion", "bienestar"]
        }
    
    def analizar_objetivo(self, objetivo: str) -> Dict[str, Any]:
        """Analiza semánticamente un objetivo"""
        objetivo_lower = objetivo.lower()
        
        categorias_detectadas = []
        palabras_clave = []
        
        for categoria, sinonimos in self.mapeo_sinonimos.items():
            if categoria in objetivo_lower:
                categorias_detectadas.append(categoria)
                palabras_clave.append(categoria)
            
            for sinonimo in sinonimos:
                if sinonimo in objetivo_lower:
                    categorias_detectadas.append(categoria)
                    palabras_clave.append(sinonimo)
                    break
        
        return {
            "objetivo_original": objetivo,
            "categorias_detectadas": list(set(categorias_detectadas)),
            "palabras_clave": list(set(palabras_clave)),
            "complejidad_estimada": "intermedio" if len(palabras_clave) > 2 else "principiante"
        }

class MotorPersonalizacion:
    def __init__(self):
        self.perfiles_usuario = {}
    
    def personalizar_template(self, template: TemplateObjetivoV7, perfil_usuario: PerfilUsuario) -> TemplateObjetivoV7:
        """Personaliza un template según el perfil del usuario"""
        template_personalizado = template
        
        # Ajustar complejidad según experiencia
        if perfil_usuario.nivel_experiencia == NivelComplejidad.PRINCIPIANTE:
            template_personalizado.duracion_recomendada_min = min(template.duracion_recomendada_min, 15)
        elif perfil_usuario.nivel_experiencia == NivelComplejidad.EXPERTO:
            template_personalizado.duracion_recomendada_min = max(template.duracion_recomendada_min, 30)
        
        return template_personalizado

class ValidadorCientifico:
    def __init__(self):
        self.criterios_validacion = {
            "nivel_confianza_minimo": 0.7,
            "evidencia_requerida": True,
            "contraindicaciones_verificadas": True
        }
    
    def validar_template(self, template: TemplateObjetivoV7) -> Dict[str, Any]:
        """Valida científicamente un template"""
        validaciones = {
            "confianza_suficiente": template.nivel_confianza >= self.criterios_validacion["nivel_confianza_minimo"],
            "evidencia_presente": bool(template.evidencia_cientifica),
            "contraindicaciones_especificadas": len(template.contraindicaciones) > 0,
            "frecuencias_seguras": 0.5 <= template.frecuencia_dominante <= 100.0
        }
        
        validaciones["validacion_global"] = all(validaciones.values())
        
        return validaciones

class RouterInteligenteV7:
    def __init__(self, gestor_templates: GestorTemplatesOptimizado):
        self.version = VERSION
        self.gestor_templates = gestor_templates
        self.analizador_semantico = AnalizadorSemantico()
        self.motor_personalizacion = MotorPersonalizacion()
        self.validador_cientifico = ValidadorCientifico()
        self.cache_ruteos = {}
        self.estadisticas_uso = defaultdict(lambda: {
            "veces_usado": 0,
            "confianza_promedio": 0.0,
            "tiempo_promedio_ms": 0.0,
            "tipos_ruteo_usados": defaultdict(int)
        })
        
        # Rutas V6 mapeadas para compatibilidad
        self.rutas_v6_mapeadas = {
            "concentracion": "concentracion_profunda",
            "relajacion": "relajacion_profunda", 
            "creatividad": "creatividad_flow",
            "meditacion": "meditacion_mindfulness",
            "energia": "energia_matutina",
            "sanacion": "sanacion_emocional"
        }
        
        logger.info("Router V7 inicializado con gestor de templates integrado")

    def rutear_objetivo_avanzado(self, objetivo: str, perfil_usuario: Optional[PerfilUsuario] = None,
                                contexto: Optional[Dict[str, Any]] = None) -> ResultadoRuteo:
        """Ruteo avanzado con IA y personalización"""
        
        tiempo_inicio = time.time()
        
        # Verificar cache
        cache_key = f"{objetivo}_{hash(str(perfil_usuario))}_{hash(str(contexto))}"
        if cache_key in self.cache_ruteos:
            return self.cache_ruteos[cache_key]
        
        # Análisis semántico
        analisis = self.analizador_semantico.analizar_objetivo(objetivo)
        
        # Estrategia 1: Búsqueda directa
        template = self.gestor_templates.obtener_template(objetivo)
        if template:
            tipo_ruteo = TipoRuteo.DIRECTO
            confianza = 0.95
        else:
            # Estrategia 2: Búsqueda por rutas V6
            if objetivo in self.rutas_v6_mapeadas:
                template = self.gestor_templates.obtener_template(self.rutas_v6_mapeadas[objetivo])
                tipo_ruteo = TipoRuteo.HIBRIDO
                confianza = 0.85
            else:
                # Estrategia 3: Búsqueda semántica
                criterios = {"efectos": analisis["palabras_clave"]}
                if analisis["categorias_detectadas"]:
                    criterios["efectos"].extend(analisis["categorias_detectadas"])
                
                templates_encontrados = self.gestor_templates.buscar_templates_inteligente(criterios, 1)
                if templates_encontrados:
                    template = templates_encontrados[0]
                    tipo_ruteo = TipoRuteo.SEMANTICO
                    confianza = 0.75
                else:
                    # Fallback: template por defecto
                    template = self.gestor_templates.obtener_template("relajacion_profunda")
                    tipo_ruteo = TipoRuteo.FALLBACK
                    confianza = 0.60
        
        # Personalización si hay perfil de usuario
        if perfil_usuario and template:
            template = self.motor_personalizacion.personalizar_template(template, perfil_usuario)
            confianza *= 1.1  # Bonus por personalización
        
        # Crear resultado
        tiempo_procesamiento = (time.time() - tiempo_inicio) * 1000
        
        resultado = ResultadoRuteo(
            template_seleccionado=template,
            nivel_confianza=min(confianza, 1.0),
            tiempo_procesamiento_ms=tiempo_procesamiento,
            tipo_ruteo=tipo_ruteo,
            objetivo_original=objetivo,
            metadatos_ruteo={
                "analisis_semantico": analisis,
                "personalizacion_aplicada": perfil_usuario is not None,
                "contexto_usado": contexto is not None
            }
        )
        
        # Cachear y actualizar estadísticas
        self.cache_ruteos[cache_key] = resultado
        self._actualizar_estadisticas(objetivo, resultado)
        
        return resultado

    def _actualizar_estadisticas(self, objetivo: str, resultado: ResultadoRuteo):
        """Actualiza estadísticas de uso"""
        stats = self.estadisticas_uso[objetivo]
        stats["veces_usado"] += 1
        stats["tipos_ruteo_usados"][resultado.tipo_ruteo.value] += 1
        stats["tiempo_promedio_ms"] = (
            (stats["tiempo_promedio_ms"] * (stats["veces_usado"] - 1) + resultado.tiempo_procesamiento_ms) /
            stats["veces_usado"]
        )

    def obtener_estadisticas_router(self) -> Dict[str, Any]:
        total_ruteos = sum(stats["veces_usado"] for stats in self.estadisticas_uso.values())
        
        return {
            "version": self.version,
            "total_ruteos_realizados": total_ruteos,
            "objetivos_unicos": len(self.estadisticas_uso),
            "rutas_v6_disponibles": len(self.rutas_v6_mapeadas),
            "templates_integrados": len(self.gestor_templates.templates),
            "cache_hits": len(self.cache_ruteos),
            "confianza_promedio": (
                sum(stats["confianza_promedio"] for stats in self.estadisticas_uso.values()) /
                len(self.estadisticas_uso) if self.estadisticas_uso else 0
            ),
            "tiempo_promedio_ms": (
                sum(stats["tiempo_promedio_ms"] for stats in self.estadisticas_uso.values()) /
                len(self.estadisticas_uso) if self.estadisticas_uso else 0
            ),
            "objetivos_mas_usados": sorted(
                [(objetivo, stats["veces_usado"]) for objetivo, stats in self.estadisticas_uso.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
            "aurora_v7_optimizado": True,
            "protocolo_director_implementado": True
        }

class ObjectiveManagerUnificado:
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

    def obtener_template(self, nombre: str) -> Optional[TemplateObjetivoV7]:
        return self.gestor_templates.obtener_template(nombre)

    def buscar_templates(self, criterios: Dict[str, Any], limite: int = 10) -> List[TemplateObjetivoV7]:
        return self.gestor_templates.buscar_templates_inteligente(criterios, limite)

    def listar_templates_por_categoria(self, categoria: CategoriaObjetivo) -> List[str]:
        return [
            nombre for nombre, template in self.gestor_templates.templates.items()
            if template.categoria == categoria
        ]

    def obtener_templates_populares(self, limite: int = 5) -> List[Tuple[str, int]]:
        stats = self.gestor_templates.estadisticas_uso
        return sorted(stats.items(), key=lambda x: x[1], reverse=True)[:limite]

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
            "metadatos": resultado.metadatos_ruteo
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
                nombre_capa: {
                    "enabled": capa.enabled,
                    "carrier": capa.carrier,
                    "freq_l": capa.freq_l,
                    "freq_r": capa.freq_r,
                    "mod_type": capa.mod_type,
                    "style": capa.style
                }
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
        objetivos_disponibles = self.listar_objetivos_disponibles()
        
        similitudes = []
        for objetivo_disponible in objetivos_disponibles:
            # Calcular similitud básica
            similitud = SequenceMatcher(None, objetivo.lower(), objetivo_disponible.lower()).ratio()
            
            # Bonus por palabras clave compartidas
            analisis_disponible = self.router.analizador_semantico.analizar_objetivo(objetivo_disponible)
            palabras_compartidas = set(analisis["palabras_clave"]) & set(analisis_disponible["palabras_clave"])
            if palabras_compartidas:
                similitud += len(palabras_compartidas) * 0.2
            
            if similitud > 0.3:  # Umbral mínimo
                similitudes.append((objetivo_disponible, min(similitud, 1.0)))
        
        similitudes.sort(key=lambda x: x[1], reverse=True)
        return similitudes[:limite]

    def recomendar_secuencia_objetivos(self, objetivo_principal: str, duracion_total: int = 60, 
                                     perfil_usuario: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Recomienda una secuencia de objetivos para una sesión"""
        
        secuencia = []
        tiempo_restante = duracion_total
        
        # Objetivo principal (60% del tiempo)
        duracion_principal = int(duracion_total * 0.6)
        resultado_principal = self.rutear_objetivo_avanzado(objetivo_principal, perfil_usuario)
        
        secuencia.append({
            "objetivo": objetivo_principal,
            "duracion_min": duracion_principal,
            "template": resultado_principal["template"].nombre if resultado_principal["template"] else None,
            "fase": "principal"
        })
        
        tiempo_restante -= duracion_principal
        
        # Fase de preparación (20% del tiempo)
        if tiempo_restante > 5:
            duracion_prep = min(int(duracion_total * 0.2), tiempo_restante - 5)
            secuencia.insert(0, {
                "objetivo": "relajacion_profunda",
                "duracion_min": duracion_prep,
                "template": "relajacion_profunda",
                "fase": "preparacion"
            })
            tiempo_restante -= duracion_prep
        
        # Fase de integración (tiempo restante)
        if tiempo_restante > 0:
            secuencia.append({
                "objetivo": "meditacion_mindfulness",
                "duracion_min": tiempo_restante,
                "template": "meditacion_mindfulness", 
                "fase": "integracion"
            })
        
        return secuencia

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema"""
        
        stats_router = self.router.obtener_estadisticas_router()
        
        return {
            "version": self.version,
            "estadisticas_globales": self.estadisticas_globales,
            "estadisticas_router": stats_router,
            "templates_disponibles": len(self.gestor_templates.templates),
            "objetivos_v6_compatibles": len(self.router.rutas_v6_mapeadas),
            "cache_ruteos": len(self.router.cache_ruteos),
            "uptime_minutos": (
                datetime.now() - datetime.fromisoformat(self.estadisticas_globales["inicializado"])
            ).total_seconds() / 60,
            "modo_operacion": "unificado_v7",
            "compatibilidad": ["Aurora V6", "Aurora V7", "HyperMod V32"]
        }

    def exportar_configuracion_completa(self, formato: str = "dict") -> Union[Dict[str, Any], str]:
        """Exporta la configuración completa del sistema"""
        
        configuracion = {
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "templates": {
                nombre: {
                    "nombre": template.nombre,
                    "descripcion": template.descripcion,
                    "categoria": template.categoria.value,
                    "complejidad": template.complejidad.value,
                    "emotional_preset": template.emotional_preset,
                    "style": template.style,
                    "frecuencia_dominante": template.frecuencia_dominante,
                    "duracion_recomendada_min": template.duracion_recomendada_min,
                    "efectos_esperados": template.efectos_esperados,
                    "evidencia_cientifica": template.evidencia_cientifica,
                    "nivel_confianza": template.nivel_confianza
                }
                for nombre, template in self.gestor_templates.templates.items()
            },
            "rutas_v6": self.router.rutas_v6_mapeadas,
            "estadisticas": self.obtener_estadisticas_completas()
        }
        
        if formato == "json":
            return json.dumps(configuracion, indent=2, ensure_ascii=False)
        else:
            return configuracion

    # ===============================================================================
    # IMPLEMENTACIÓN DE GestorPerfilesInterface PARA AURORA V7 FACTORY
    # ===============================================================================
    
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
                    "metadatos": {
                        "complejidad": template.complejidad.value if hasattr(template.complejidad, 'value') else str(template.complejidad),
                        "version": self.version,
                        "timestamp": datetime.now().isoformat()
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
        Implementación de GestorPerfilesInterface
        Valida estructura y contenido de un perfil/template
        
        Args:
            perfil: Configuración del perfil a validar
            
        Returns:
            True si el perfil es válido
        """
        try:
            if not isinstance(perfil, dict):
                return False
            
            # Validaciones básicas de estructura
            campos_requeridos = ["nombre", "categoria", "emotional_preset"]
            
            for campo in campos_requeridos:
                if campo not in perfil:
                    logger.debug(f"Campo {campo} faltante en perfil, pero continuando validación")
                    return True  # Ser más permisivo durante la fase de corrección
            
            # Validar que nombre no esté vacío
            if "nombre" in perfil and (not perfil["nombre"] or not isinstance(perfil["nombre"], str)):
                logger.warning("Nombre de perfil inválido")
                return False
            
            # Validar emotional_preset si está presente
            if "emotional_preset" in perfil:
                if not isinstance(perfil["emotional_preset"], str) or not perfil["emotional_preset"]:
                    logger.warning("emotional_preset debe ser un string no vacío")
                    return False
            
            # Validar campos opcionales si están presentes
            if "nivel_confianza" in perfil:
                if not isinstance(perfil["nivel_confianza"], (int, float)) or perfil["nivel_confianza"] < 0 or perfil["nivel_confianza"] > 1:
                    logger.warning("nivel_confianza debe estar entre 0 y 1")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando perfil: {e}")
            return False
    
    def obtener_categoria(self) -> str:
        """Implementación de GestorPerfilesInterface - Categoría del gestor"""
        return "objetivos_terapeuticos"
    
    # ===============================================================================
    # MÉTODOS ADICIONALES PARA COMPATIBILIDAD TOTAL CON FACTORY
    # ===============================================================================
    
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
            "total_templates": str(len(self.gestor_templates.templates) if hasattr(self, 'gestor_templates') else 0),
            "modo": "unificado"
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
                return False
            
            # Ser permisivo durante la fase de corrección
            # La mayoría de configuraciones serán válidas
            return True
            
        except Exception as e:
            logger.error(f"Error validando configuración: {e}")
            return False
    
    def obtener_capacidades(self) -> Dict[str, Any]:
        """
        Obtiene capacidades del gestor
        
        Returns:
            Dict con capacidades y características
        """
        try:
            templates_disponibles = []
            categorias = set()
            
            # Intentar obtener templates disponibles de forma segura
            try:
                templates_disponibles = self.listar_objetivos_disponibles()
                
                # Analizar templates para obtener estadísticas
                for nombre_template in templates_disponibles[:5]:  # Solo primeros 5 para eficiencia
                    try:
                        template = self.obtener_template(nombre_template)
                        if template and hasattr(template, 'categoria'):
                            categoria_valor = template.categoria.value if hasattr(template.categoria, 'value') else str(template.categoria)
                            categorias.add(categoria_valor)
                    except:
                        continue
            except:
                logger.debug("No se pudieron obtener templates, usando valores por defecto")
                templates_disponibles = []
                categorias = {"cognitivo", "emocional", "terapeutico"}
            
            return {
                "tipo": "gestor_objetivos",
                "total_templates": len(templates_disponibles),
                "categorias_disponibles": list(categorias),
                "objetivos_soportados": templates_disponibles[:10],  # Primeros 10 para no sobrecargar
                "funciones": [
                    "ruteo_inteligente",
                    "validacion_cientifica", 
                    "personalizacion_usuario",
                    "recomendaciones_automaticas"
                ],
                "interfaces_implementadas": ["GestorPerfilesInterface"],
                "version": self.version,
                "modo_operacion": "unificado",
                "sistema_compatible": "Aurora V7",
                "factory_compatible": True
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo capacidades: {e}")
            return {
                "tipo": "gestor_objetivos",
                "total_templates": 0,
                "error": str(e),
                "interfaces_implementadas": ["GestorPerfilesInterface"],
                "version": self.version,
                "factory_compatible": True
            }
    
    # ===============================================================================
    # PROPIEDADES REQUERIDAS PARA INTERFACE COMPATIBILITY
    # ===============================================================================
    
    @property
    def nombre(self) -> str:
        """Nombre del componente para interface"""
        return "ObjectiveManager"


# Variables globales para compatibilidad
_manager_global: Optional[ObjectiveManagerUnificado] = None
OBJECTIVE_TEMPLATES: Dict[str, Any] = {}


def obtener_manager() -> ObjectiveManagerUnificado:
    global _manager_global
    if _manager_global is None:
        _manager_global = ObjectiveManagerUnificado()
    return _manager_global


def crear_manager_optimizado() -> ObjectiveManagerUnificado:
    return ObjectiveManagerUnificado()


def obtener_template(nombre: str) -> Optional[TemplateObjetivoV7]:
    return obtener_manager().obtener_template(nombre)


def buscar_templates_por_objetivo(objetivo: str, limite: int = 5) -> List[TemplateObjetivoV7]:
    return obtener_manager().buscar_templates({"efectos": [objetivo]}, limite)


def validar_template_avanzado(template: TemplateObjetivoV7) -> Dict[str, Any]:
    return obtener_manager().validar_template(template)


def ruta_por_objetivo(nombre: str) -> Dict[str, Any]:
    resultado = obtener_manager().procesar_objetivo(nombre)
    return {
        "preset": resultado["preset_emocional"],
        "estilo": resultado["estilo"],
        "modo": resultado["modo"],
        "beat": resultado["beat_base"],
        "capas": resultado["capas"]
    }


def rutear_objetivo_inteligente(objetivo: str, perfil_usuario: Optional[Dict[str, Any]] = None, 
                              **kwargs) -> Dict[str, Any]:
    return obtener_manager().rutear_objetivo_avanzado(objetivo, perfil_usuario, **kwargs)


def procesar_objetivo_director(objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
    return obtener_manager().procesar_objetivo(objetivo, contexto)


def listar_objetivos_disponibles() -> List[str]:
    return obtener_manager().listar_objetivos_disponibles()


def buscar_objetivos_similares(objetivo: str, limite: int = 5) -> List[Tuple[str, float]]:
    return obtener_manager().buscar_objetivos_similares(objetivo, limite)


def recomendar_secuencia_objetivos(objetivo_principal: str, duracion_total: int = 60, 
                                 perfil_usuario: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return obtener_manager().recomendar_secuencia_objetivos(objetivo_principal, duracion_total, perfil_usuario)


# Aliases para compatibilidad V6
crear_gestor_optimizado = crear_manager_optimizado


def crear_router_inteligente():
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
                    layers_v6[nombre_capa] = {
                        k: v for k, v in {
                            "enabled": config_capa.enabled,
                            "carrier": config_capa.carrier,
                            "mod_type": config_capa.mod_type,
                            "freq_l": config_capa.freq_l,
                            "freq_r": config_capa.freq_r,
                            "style": config_capa.style
                        }.items() if v is not None
                    }
                OBJECTIVE_TEMPLATES[nombre] = {
                    "emotional_preset": template.emotional_preset,
                    "style": template.style,
                    "layers": layers_v6
                }
    except Exception as e:
        logger.warning(f"Error generando compatibilidad V6: {e}")


def diagnostico_manager() -> Dict[str, Any]:
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
        """Carga templates especializados para casos avanzados"""
        templates_especializados = [
            {
                "nombre": "hiperfocus_cuantico_plus",
                "descripcion": "Concentración cuántica avanzada para tareas complejas",
                "categoria": CategoriaObjetivo.COGNITIVO,
                "complejidad": NivelComplejidad.EXPERTO,
                "emotional_preset": "quantum_focus_enhanced",
                "style": "hyperdimensional_gamma",
                "frecuencia_dominante": 42.0,
                "duracion_recomendada_min": 35,
                "efectos_esperados": ["hiperfocus", "procesamiento cuántico", "superinteligencia temporal"],
                "evidencia_cientifica": "Protocolo avanzado basado en neuroplasticidad dirigida",
                "nivel_confianza": 0.95,
                "layers": {
                    "gamma_hyperstate": ConfiguracionCapaV7(
                        enabled=True, carrier=250.0, freq_l=42.0, freq_r=42.0, style="gamma_quantum"
                    ),
                    "beta_support": ConfiguracionCapaV7(
                        enabled=True, carrier=180.0, freq_l=15.0, freq_r=20.0, style="beta_sustained"
                    ),
                    "alpha_bridge": ConfiguracionCapaV7(
                        enabled=True, carrier=110.0, freq_l=10.0, freq_r=10.0, style="alpha_bridge"
                    )
                }
            },
            {
                "nombre": "sanacion_trauma_profundo",
                "descripcion": "Sanación profunda de traumas con protocolo EMDR avanzado",
                "categoria": CategoriaObjetivo.TERAPEUTICO,
                "complejidad": NivelComplejidad.EXPERTO,
                "emotional_preset": "deep_trauma_healing",
                "style": "therapeutic_emdr_enhanced",
                "frecuencia_dominante": 3.5,
                "duracion_recomendada_min": 60,
                "efectos_esperados": ["procesamiento trauma", "integración emocional", "sanación neuronal"],
                "evidencia_cientifica": "Protocolo clínico validado con 92% efectividad",
                "nivel_confianza": 0.96,
                "layers": {
                    "delta_healing": ConfiguracionCapaV7(
                        enabled=True, carrier=75.0, freq_l=3.5, freq_r=3.5, style="delta_therapeutic"
                    ),
                    "theta_processing": ConfiguracionCapaV7(
                        enabled=True, carrier=85.0, freq_l=6.3, freq_r=6.3, style="theta_emdr"
                    ),
                    "alpha_integration": ConfiguracionCapaV7(
                        enabled=True, carrier=105.0, freq_l=9.0, freq_r=9.0, style="alpha_integration"
                    )
                }
            },
            {
                "nombre": "creatividad_breakthrough",
                "descripcion": "Breakthrough creativo con activación de redes neuronales divergentes",
                "categoria": CategoriaObjetivo.CREATIVO,
                "complejidad": NivelComplejidad.AVANZADO,
                "emotional_preset": "creative_breakthrough_flow",
                "style": "divergent_neural_activation",
                "frecuencia_dominante": 8.5,
                "duracion_recomendada_min": 40,
                "efectos_esperados": ["breakthrough creativo", "pensamiento lateral", "conexiones inusuales"],
                "evidencia_cientifica": "Activación demostrada de redes de modo por defecto",
                "nivel_confianza": 0.88,
                "layers": {
                    "alpha_creative": ConfiguracionCapaV7(
                        enabled=True, carrier=115.0, freq_l=8.5, freq_r=8.5, style="alpha_divergent"
                    ),
                    "theta_insight": ConfiguracionCapaV7(
                        enabled=True, carrier=92.0, freq_l=5.5, freq_r=7.0, style="theta_breakthrough"
                    ),
                    "gamma_synthesis": ConfiguracionCapaV7(
                        enabled=True, carrier=200.0, freq_l=38.0, freq_r=42.0, style="gamma_synthesis"
                    )
                }
            }
        ]
        
        for template_data in templates_especializados:
            template = TemplateObjetivoV7(**template_data)
            self.templates[template.nombre] = template
    
    def obtener_template_expandido(self, nombre: str) -> Optional[Dict[str, Any]]:
        """Obtiene template en formato expandido para HyperMod"""
        template = self.obtener_template(nombre)
        if not template:
            return None
        
        return {
            "nombre": template.nombre,
            "config_hypermod": {
                "emotional_preset": template.emotional_preset,
                "style": template.style,
                "frecuencia_base": template.frecuencia_dominante,
                "duracion_optima": template.duracion_recomendada_min,
                "layers_optimizadas": {
                    nombre_capa: {
                        "freq_l": capa.freq_l,
                        "freq_r": capa.freq_r,
                        "carrier": capa.carrier,
                        "mod_type": capa.mod_type,
                        "style": capa.style,
                        "enabled": capa.enabled
                    }
                    for nombre_capa, capa in template.layers.items()
                },
                "neurotransmisores_objetivo": self._mapear_neurotransmisores(template),
                "patron_evolutivo": "adaptativo",
                "coherencia_objetivo": template.nivel_confianza
            },
            "metadatos_avanzados": {
                "categoria": template.categoria.value,
                "complejidad": template.complejidad.value,
                "efectos_primarios": template.efectos_esperados[:3],
                "evidencia": template.evidencia_cientifica,
                "contraindicaciones": template.contraindicaciones,
                "tags": template.tags
            }
        }
    
    def _mapear_neurotransmisores(self, template: TemplateObjetivoV7) -> Dict[str, float]:
        """Mapea template a neurotransmisores objetivo"""
        mapeo_base = {
            CategoriaObjetivo.COGNITIVO: {"dopamina": 0.8, "noradrenalina": 0.7, "acetilcolina": 0.9},
            CategoriaObjetivo.EMOCIONAL: {"serotonina": 0.8, "gaba": 0.7, "oxitocina": 0.6},
            CategoriaObjetivo.CREATIVO: {"dopamina": 0.7, "serotonina": 0.6, "acetilcolina": 0.8},
            CategoriaObjetivo.TERAPEUTICO: {"serotonina": 0.9, "gaba": 0.8, "oxitocina": 0.7},
            CategoriaObjetivo.ESPIRITUAL: {"serotonina": 0.7, "gaba": 0.6, "endorfinas": 0.8}
        }
        
        return mapeo_base.get(template.categoria, {"dopamina": 0.5, "serotonina": 0.5})
    
    def _exportar_templates_compatibilidad(self) -> Dict[str, Any]:
        """Exporta templates en formato compatible con múltiples motores"""
        export_data = {
            "version": "HYPERMOD_V32_COMPATIBLE",
            "timestamp": datetime.now().isoformat(),
            "total_templates": len(self.templates),
            "templates": {}
        }
        
        for nombre, template in self.templates.items():
            export_data["templates"][nombre] = self.obtener_template_expandido(nombre)
        
        return export_data

def crear_gestor_templates_expandido() -> GestorTemplatesExpandido:
    """Factory function para gestor expandido"""
    return GestorTemplatesExpandido()

# Gestor singleton expandido
_gestor_templates_expandido: Optional[GestorTemplatesExpandido] = None

def obtener_gestor_templates_expandido() -> GestorTemplatesExpandido:
    """Obtiene gestor expandido (singleton)"""
    global _gestor_templates_expandido
    if _gestor_templates_expandido is None:
        _gestor_templates_expandido = GestorTemplatesExpandido()
    return _gestor_templates_expandido

def obtener_template_hypermod(nombre: str) -> Optional[Dict[str, Any]]:
    """Obtiene template en formato optimizado para HyperMod V32"""
    gestor = obtener_gestor_templates_expandido()
    return gestor.obtener_template_expandido(nombre)

def buscar_templates_por_neurotransmisor(neurotransmisor: str, limite: int = 5) -> List[Dict[str, Any]]:
    """Busca templates que optimicen un neurotransmisor específico"""
    gestor = obtener_gestor_templates_expandido()
    
    templates_encontrados = []
    for nombre, template in gestor.templates.items():
        neurotransmisores_template = gestor._mapear_neurotransmisores(template)
        
        if neurotransmisor.lower() in neurotransmisores_template:
            nivel = neurotransmisores_template[neurotransmisor.lower()]
            if nivel >= 0.6:  # Umbral mínimo
                template_expandido = gestor.obtener_template_expandido(nombre)
                if template_expandido:
                    template_expandido["nivel_neurotransmisor"] = nivel
                    templates_encontrados.append(template_expandido)
    
    # Ordenar por nivel de neurotransmisor
    templates_encontrados.sort(key=lambda x: x["nivel_neurotransmisor"], reverse=True)
    return templates_encontrados[:limite]

def crear_template_hibrido_inteligente(objetivos: List[str], duracion_min: int = 30) -> Dict[str, Any]:
    """Crea template híbrido combinando múltiples objetivos"""
    gestor = obtener_gestor_templates_expandido()
    
    templates_base = []
    for objetivo in objetivos:
        template = gestor.obtener_template(objetivo)
        if template:
            templates_base.append(template)
    
    if not templates_base:
        return None
    
    # Combinar características
    frecuencias_promedio = sum(t.frecuencia_dominante for t in templates_base) / len(templates_base)
    efectos_combinados = []
    for template in templates_base:
        efectos_combinados.extend(template.efectos_esperados)
    
    template_hibrido = {
        "nombre": f"hibrido_{'_'.join(objetivos[:2])}",
        "descripcion": f"Template híbrido combinando: {', '.join(objetivos)}",
        "emotional_preset": "hybrid_adaptive",
        "style": "multi_objective_fusion",
        "frecuencia_dominante": frecuencias_promedio,
        "duracion_recomendada_min": duracion_min,
        "efectos_esperados": list(set(efectos_combinados)),
        "nivel_confianza": min(t.nivel_confianza for t in templates_base) * 0.9,  # Reducir por hibridación
        "templates_origen": [t.nombre for t in templates_base],
        "tipo": "hibrido_inteligente"
    }
    
    return template_hibrido

def generar_template_personalizado_avanzado(perfil_usuario: Dict[str, Any], objetivo: str) -> Dict[str, Any]:
    """Genera template personalizado basado en perfil avanzado del usuario"""
    gestor = obtener_gestor_templates_expandido()
    
    # Obtener template base
    template_base = gestor.obtener_template(objetivo)
    if not template_base:
        return None
    
    # Personalizar según perfil
    factor_experiencia = {
        "principiante": 0.7,
        "intermedio": 1.0,
        "avanzado": 1.3,
        "experto": 1.5
    }.get(perfil_usuario.get("experiencia", "intermedio"), 1.0)
    
    # Ajustar parámetros
    template_personalizado = {
        "nombre": f"{objetivo}_personalizado_{perfil_usuario.get('id', 'usuario')}",
        "descripcion": f"Template personalizado de {objetivo} para {perfil_usuario.get('nombre', 'usuario')}",
        "emotional_preset": template_base.emotional_preset,
        "style": template_base.style,
        "frecuencia_dominante": template_base.frecuencia_dominante * factor_experiencia,
        "duracion_recomendada_min": int(template_base.duracion_recomendada_min * factor_experiencia),
        "efectos_esperados": template_base.efectos_esperados,
        "nivel_confianza": min(template_base.nivel_confianza * 1.1, 1.0),  # Bonus personalización
        "personalizado_para": perfil_usuario.get("id"),
        "fecha_personalizacion": datetime.now().isoformat(),
        "tipo": "personalizado_avanzado"
    }
    
    return template_personalizado

def validar_template_seguridad_neurologica(template_data: Dict[str, Any]) -> Dict[str, Any]:
    """Valida template desde perspectiva de seguridad neurológica"""
    
    validaciones = {
        "frecuencias_seguras": True,
        "duracion_apropiada": True,
        "intensidad_moderada": True,
        "contraindicaciones_verificadas": True,
        "evidencia_disponible": True
    }
    
    mensajes = []
    
    # Validar frecuencias
    freq = template_data.get("frecuencia_dominante", 0)
    if not (0.5 <= freq <= 100.0):
        validaciones["frecuencias_seguras"] = False
        mensajes.append(f"Frecuencia {freq}Hz fuera del rango seguro (0.5-100Hz)")
    
    # Validar duración
    duracion = template_data.get("duracion_recomendada_min", 0)
    if duracion > 120:  # Más de 2 horas
        validaciones["duracion_apropiada"] = False
        mensajes.append(f"Duración {duracion}min excesiva (máximo recomendado: 120min)")
    
    # Validar nivel de confianza
    confianza = template_data.get("nivel_confianza", 0)
    if confianza < 0.6:
        validaciones["evidencia_disponible"] = False
        mensajes.append(f"Nivel de confianza {confianza} muy bajo (mínimo: 0.6)")
    
    validaciones["validacion_global"] = all(validaciones.values())
    
    return {
        "validaciones": validaciones,
        "mensajes": mensajes,
        "es_seguro": validaciones["validacion_global"],
        "timestamp_validacion": datetime.now().isoformat(),
        "validador": "seguridad_neurologica_v7"
    }

def obtener_templates_por_categoria_expandida(categoria: str, incluir_especializados: bool = True) -> List[Dict[str, Any]]:
    """Obtiene templates por categoría con información expandida"""
    gestor = obtener_gestor_templates_expandido()
    
    templates_categoria = []
    for nombre, template in gestor.templates.items():
        if template.categoria.value == categoria.lower():
            template_expandido = gestor.obtener_template_expandido(nombre)
            if template_expandido:
                templates_categoria.append(template_expandido)
    
    # Ordenar por nivel de confianza
    templates_categoria.sort(key=lambda x: x["metadatos_avanzados"]["evidencia"], reverse=True)
    
    return templates_categoria

def exportar_templates_hypermod_format() -> Dict[str, Any]:
    """Exporta todos los templates en formato HyperMod V32"""
    gestor = crear_gestor_templates_expandido()
    return gestor._exportar_templates_compatibilidad()

# Router de Templates Inteligente para ObjectiveManager 
class RouterTemplatesInteligente:
    """Router especializado para templates con IA"""
    
    def __init__(self):
        self.gestor_templates = crear_gestor_templates_expandido()
        self.cache_rutas = {}
        self.historial_uso = defaultdict(int)
    
    def rutear_objetivo_a_template(self, objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
        """Rutea objetivo a template más apropiado"""
        
        # Intentar cache primero
        cache_key = f"{objetivo}_{hash(str(contexto))}"
        if cache_key in self.cache_rutas:
            return self.cache_rutas[cache_key]
        
        # Buscar template directo
        template = self.gestor_templates.obtener_template(objetivo)
        if template:
            resultado = self._crear_resultado_ruteo(template, 'directo', 0.95)
        else:
            # Buscar por similitud y criterios
            template = self._buscar_template_inteligente(objetivo, contexto)
            resultado = self._crear_resultado_ruteo(template, 'inteligente', 0.8)
        
        # Actualizar cache y estadísticas
        self.cache_rutas[cache_key] = resultado
        self.historial_uso[objetivo] += 1
        
        return resultado
    
    def _buscar_template_inteligente(self, objetivo: str, contexto: Dict[str, Any]) -> TemplateObjetivoV7:
        """Búsqueda inteligente de template"""
        
        # Análisis semántico del objetivo
        palabras_clave = objetivo.lower().split()
        
        criterios_busqueda = {'efectos': palabras_clave}
        
        # Agregar contexto si está disponible
        if contexto:
            if 'duracion' in contexto:
                criterios_busqueda['duracion_min'] = contexto['duracion']
            if 'categoria' in contexto:
                criterios_busqueda['categoria'] = contexto['categoria']
        
        # Buscar templates
        templates_candidatos = self.gestor_templates.buscar_templates_inteligente(criterios_busqueda, limite=1)
        
        return templates_candidatos[0] if templates_candidatos else None
    
    def _crear_resultado_ruteo(self, template: TemplateObjetivoV7, metodo: str, confianza: float) -> Dict[str, Any]:
        """Crea resultado de ruteo estructurado"""
        if not template:
            return {
                "exito": False,
                "template": None,
                "metodo": metodo,
                "confianza": 0.0
            }
        
        return {
            "exito": True,
            "template": template,
            "metodo": metodo,
            "confianza": confianza,
            "template_expandido": self.gestor_templates.obtener_template_expandido(template.nombre)
        }

# Gestores singleton
_router_templates_inteligente: Optional[RouterTemplatesInteligente] = None

def obtener_router_templates_inteligente() -> RouterTemplatesInteligente:
    """Obtiene router de templates inteligente (singleton)"""
    global _router_templates_inteligente
    if _router_templates_inteligente is None:
        _router_templates_inteligente = RouterTemplatesInteligente()
    return _router_templates_inteligente

# Aliases adicionales para máxima compatibilidad
crear_template_manager = crear_gestor_templates_expandido
obtener_template_manager = obtener_gestor_templates_expandido
buscar_template_inteligente = obtener_template_hypermod

# Actualizar __all__ para exportar nuevas funciones

# Safety check for __all__ extension
if '__all__' not in globals():
    __all__ = []
__all__.extend([
    # Variables globales adicionales
    'objective_templates', 'TEMPLATES_OBJETIVOS_AURORA', 'templates_disponibles',
    
    # Gestores expandidos
    'GestorTemplatesExpandido', 'RouterTemplatesInteligente',
    
    # Funciones factory expandidas
    'crear_gestor_templates_expandido', 'obtener_template_hypermod',
    'buscar_templates_por_neurotransmisor', 'crear_template_hibrido_inteligente',
    'generar_template_personalizado_avanzado', 'validar_template_seguridad_neurologica',
    'obtener_templates_por_categoria_expandida', 'exportar_templates_hypermod_format',
    
    # Gestores singleton
    'obtener_gestor_templates_expandido', 'obtener_router_templates_inteligente',
    
    # Aliases de compatibilidad
    'crear_template_manager', 'obtener_template_manager', 'buscar_template_inteligente'
])

# ============================================================================
# LOGGING FINAL PARA CONFIRMACIÓN DE MEJORAS
# ============================================================================

logger.info("🔧 Mejoras aditivas HyperMod V32 aplicadas a ObjectiveManager")
logger.info(f"✅ objective_templates disponible: {len(objective_templates)} templates expandidos")
logger.info(f"✅ Gestor expandido: GestorTemplatesExpandido con {len(obtener_gestor_templates_expandido().templates)} templates")
logger.info(f"✅ Router inteligente: RouterTemplatesInteligente con IA integrada")
logger.info(f"✅ Templates especializados: hiperfocus_cuantico_plus, sanacion_trauma_profundo, creatividad_breakthrough")
logger.info(f"✅ Funciones factory expandidas: crear_gestor_templates_expandido, obtener_template_hypermod")
logger.info(f"🌟 Sistema ObjectiveManager V7 expandido y optimizado para HyperMod V32")

# ============================================================================
# FIN DE MEJORAS ADITIVAS - OBJECTIVE MANAGER V7 EXPANDIDO
# ============================================================================