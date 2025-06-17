"""Aurora V7 Field Profiles - Optimized with Surgical Filtering"""
import numpy as np
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from functools import lru_cache
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Aurora.FieldProfiles.V7")
VERSION = "V7_AURORA_DIRECTOR_CONNECTED_FILTERED"

class GestorInteligencia(Protocol):
    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]: ...
    def obtener_alternativas(self, objetivo: str) -> List[str]: ...

def _safe_import_templates():
    try:
        from objective_templates_optimized import ConfiguracionCapaV7, TemplateObjetivoV7, CategoriaObjetivo, NivelComplejidad, ModoActivacion
        return ConfiguracionCapaV7, TemplateObjetivoV7, CategoriaObjetivo, True
    except ImportError:
        class ConfiguracionCapaV7:
            def __init__(self, **kwargs):
                for k, v in kwargs.items(): setattr(self, k, v)
                self.enabled = True
        class TemplateObjetivoV7:
            def __init__(self, **kwargs):
                for k, v in kwargs.items(): setattr(self, k, v)
        class CategoriaObjetivo(Enum):
            COGNITIVO = "cognitivo"
            EMOCIONAL = "emocional"
            ESPIRITUAL = "espiritual"
            CREATIVO = "creativo"
            TERAPEUTICO = "terapeutico"
            FISICO = "fisico"
            SOCIAL = "social"
            EXPERIMENTAL = "experimental"
        return ConfiguracionCapaV7, TemplateObjetivoV7, CategoriaObjetivo, False

ConfiguracionCapaV7, TemplateObjetivoV7, CategoriaObjetivo, TEMPLATES_AVAILABLE = _safe_import_templates()

class CampoCosciencia(Enum):
    COGNITIVO = "cognitivo"
    EMOCIONAL = "emocional"
    ESPIRITUAL = "espiritual"
    FISICO = "fisico"
    ENERGETICO = "energetico"
    SOCIAL = "social"
    CREATIVO = "creativo"
    SANACION = "sanacion"

class NivelActivacion(Enum):
    SUTIL = "sutil"
    MODERADO = "moderado"
    INTENSO = "intenso"
    PROFUNDO = "profundo"
    TRASCENDENTE = "trascendente"

class TipoRespuesta(Enum):
    INMEDIATA = "inmediata"
    PROGRESIVA = "progresiva"
    PROFUNDA = "profunda"
    INTEGRATIVA = "integrativa"

class CalidadEvidencia(Enum):
    EXPERIMENTAL = "experimental"
    VALIDADO = "validado"
    CLINICO = "clinico"
    INVESTIGACION = "investigacion"

@dataclass
class ConfiguracionNeuroacustica:
    beat_primario: float = 10.0
    beat_secundario: Optional[float] = None
    armonicos: List[float] = field(default_factory=list)
    modulacion_amplitude: float = 0.5
    modulacion_frecuencia: float = 0.1
    modulacion_fase: float = 0.0
    lateralizacion: float = 0.0
    profundidad_espacial: float = 1.0
    movimiento_3d: bool = False
    patron_movimiento: str = "estatico"
    evolucion_activada: bool = False
    curva_evolucion: str = "lineal"
    tiempo_evolucion_min: float = 5.0
    frecuencias_resonancia: List[float] = field(default_factory=list)
    Q_factor: float = 1.0
    coherencia_neuroacustica: float = 0.8
    
    def __post_init__(self):
        if not self.beat_secundario: 
            self.beat_secundario = self.beat_primario * 1.618
        if not self.armonicos: 
            self.armonicos = [self.beat_primario * i for i in [2, 3, 4]]
        if not 0.1 <= self.beat_primario <= 100: 
            logger.warning(f"Beat fuera de rango: {self.beat_primario}")
        if not 0 <= self.modulacion_amplitude <= 1: 
            logger.warning(f"Modulación fuera de rango: {self.modulacion_amplitude}")

@dataclass
class MetricasEfectividad:
    veces_usado: int = 0
    efectividad_promedio: float = 0.0
    satisfaccion_usuarios: float = 0.0
    tiempo_respuesta_promedio: float = 0.0
    reportes_efectos_positivos: int = 0
    reportes_efectos_negativos: int = 0
    ultima_actualizacion: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def actualizar_uso(self, efectividad: float, satisfaccion: float):
        self.veces_usado += 1
        self.efectividad_promedio = ((self.efectividad_promedio * (self.veces_usado - 1) + efectividad) / self.veces_usado)
        self.satisfaccion_usuarios = ((self.satisfaccion_usuarios * (self.veces_usado - 1) + satisfaccion) / self.veces_usado)
        self.ultima_actualizacion = datetime.now().isoformat()

@dataclass
class PerfilCampoV7:
    nombre: str
    descripcion: str
    campo_consciencia: CampoCosciencia
    version: str = "v7.0"
    style: str = "sereno"
    configuracion_neuroacustica: ConfiguracionNeuroacustica = field(default_factory=ConfiguracionNeuroacustica)
    neurotransmisores_simples: List[str] = field(default_factory=list)
    neurotransmisores_principales: Dict[str, float] = field(default_factory=dict)
    neurotransmisores_moduladores: Dict[str, float] = field(default_factory=dict)
    beat_base: float = 10.0
    ondas_primarias: List[str] = field(default_factory=list)
    ondas_secundarias: List[str] = field(default_factory=list)
    patron_ondas: str = "estable"
    nivel_activacion: NivelActivacion = NivelActivacion.MODERADO
    tipo_respuesta: TipoRespuesta = TipoRespuesta.PROGRESIVA
    duracion_efecto_min: int = 15
    duracion_optima_min: int = 30
    duracion_maxima_min: int = 60
    efectos_cognitivos: List[str] = field(default_factory=list)
    efectos_emocionales: List[str] = field(default_factory=list)
    efectos_fisicos: List[str] = field(default_factory=list)
    efectos_energeticos: List[str] = field(default_factory=list)
    mejores_momentos: List[str] = field(default_factory=list)
    ambientes_optimos: List[str] = field(default_factory=list)
    posturas_recomendadas: List[str] = field(default_factory=list)
    perfiles_sinergicos: List[str] = field(default_factory=list)
    perfiles_antagonicos: List[str] = field(default_factory=list)
    secuencia_recomendada: List[str] = field(default_factory=list)
    parametros_ajustables: Dict[str, Any] = field(default_factory=dict)
    adaptable_intensidad: bool = True
    adaptable_duracion: bool = True
    base_cientifica: CalidadEvidencia = CalidadEvidencia.VALIDADO
    estudios_referencia: List[str] = field(default_factory=list)
    nivel_evidencia: float = 0.8
    mecanismo_accion: str = "neuroacústico"
    contraindicaciones: List[str] = field(default_factory=list)
    precauciones: List[str] = field(default_factory=list)
    poblacion_objetivo: List[str] = field(default_factory=list)
    complejidad_tecnica: str = "moderado"
    recursos_requeridos: Dict[str, Any] = field(default_factory=dict)
    compatibilidad_v6: bool = True
    metricas: MetricasEfectividad = field(default_factory=MetricasEfectividad)
    aurora_director_compatible: bool = True
    protocolo_inteligencia: bool = True
    # ✅ NUEVA PROPIEDAD PARA METADATA EXTRA
    metadata_extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.neurotransmisores_principales and self.neurotransmisores_simples:
            self.neurotransmisores_principales = {nt: 1.0 for nt in self.neurotransmisores_simples}
        
        if not self.configuracion_neuroacustica.beat_primario:
            self.configuracion_neuroacustica.beat_primario = self.beat_base
        
        self.configuracion_neuroacustica.beat_primario = self.beat_base
        
        if self.campo_consciencia in [CampoCosciencia.EMOCIONAL, CampoCosciencia.SANACION]:
            if not self.efectos_emocionales:
                self.efectos_emocionales = ["Equilibrio emocional", "Bienestar"]
        
        if self.campo_consciencia == CampoCosciencia.COGNITIVO:
            if not self.efectos_cognitivos:
                self.efectos_cognitivos = ["Claridad mental", "Concentración"]

    def migrar_desde_v6(self, datos_v6: Dict[str, Any]) -> 'PerfilCampoV7':
        """Migra un perfil V6 a V7 con validaciones"""
        self.style = datos_v6.get("style", "sereno")
        self.neurotransmisores_simples = datos_v6.get("nt", [])
        self.beat_base = datos_v6.get("beat", 10.0)
        self.version = "v7.0_migrado_v6"
        self.compatibilidad_v6 = True
        return self

    def validar_coherencia(self) -> Tuple[bool, List[str]]:
        """Valida la coherencia del perfil"""
        errores = []
        
        if not self.nombre or len(self.nombre) < 3:
            errores.append("Nombre debe tener al menos 3 caracteres")
        
        if not self.descripcion or len(self.descripcion) < 10:
            errores.append("Descripción debe tener al menos 10 caracteres")
        
        if not 0.1 <= self.beat_base <= 100:
            errores.append(f"Beat base fuera de rango válido: {self.beat_base}")
        
        if not 0.0 <= self.nivel_evidencia <= 1.0:
            errores.append(f"Nivel de evidencia fuera de rango: {self.nivel_evidencia}")
        
        if not self.efectos_cognitivos and not self.efectos_emocionales and not self.efectos_fisicos:
            errores.append("Debe tener al menos un tipo de efecto definido")
        
        return len(errores) == 0, errores

    def calcular_coherencia_neuroacustica(self) -> float:
        """Calcula la coherencia neuroacústica del perfil"""
        try:
            base_coherencia = 0.5
            
            # Factor por neurotransmisores
            if self.neurotransmisores_principales:
                nt_factor = min(len(self.neurotransmisores_principales) * 0.1, 0.3)
                base_coherencia += nt_factor
            
            # Factor por configuración
            if self.configuracion_neuroacustica:
                config_factor = self.configuracion_neuroacustica.coherencia_neuroacustica * 0.3
                base_coherencia += config_factor
            
            # Factor por evidencia científica
            evidencia_factor = self.nivel_evidencia * 0.2
            base_coherencia += evidencia_factor
            
            return min(base_coherencia, 1.0)
        except Exception as e:
            logger.warning(f"Error calculando coherencia para {self.nombre}: {e}")
            return 0.5

    def configurar_para_aurora_director(self) -> Dict[str, Any]:
        """Configura el perfil para integración con Aurora Director V7"""
        return {
            "perfil_id": self.nombre,
            "estilo": self.style,
            "modo": "perfil_campo",
            "beat_base": self.configuracion_neuroacustica.beat_primario,
            "capas": {
                "neuro_wave": len(self.neurotransmisores_principales) > 0,
                "binaural": self.configuracion_neuroacustica.beat_primario > 0,
                "wave_pad": len(self.configuracion_neuroacustica.armonicos) > 0,
                "textured_noise": True,
                "heartbeat": self.campo_consciencia in [CampoCosciencia.EMOCIONAL, CampoCosciencia.ENERGETICO, CampoCosciencia.SANACION]
            },
            "neurotransmisores": self.neurotransmisores_principales,
            "duracion_recomendada": self.duracion_optima_min,
            "coherencia": self.calcular_coherencia_neuroacustica(),
            "validacion_cientifica": self.base_cientifica.value,
            "aurora_v7_optimizado": True
        }

    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa un objetivo específico con este perfil"""
        relevancia = 0.5
        
        # Calcular relevancia basada en efectos
        efectos_totales = self.efectos_cognitivos + self.efectos_emocionales + self.efectos_fisicos + self.efectos_energeticos
        if any(objetivo.lower() in efecto.lower() for efecto in efectos_totales):
            relevancia += 0.3
        
        # Factor por campo de consciencia
        if objetivo.lower() in self.campo_consciencia.value.lower():
            relevancia += 0.2
        
        return {
            "perfil": self.nombre,
            "relevancia": min(relevancia, 1.0),
            "configuracion_optimizada": self.configurar_para_aurora_director(),
            "duracion_sugerida": self.duracion_optima_min,
            "precauciones": self.precauciones
        }

    def exportar_para_motores(self) -> Dict[str, Any]:
        """Exporta configuración optimizada para motores Aurora"""
        return {
            "nombre": self.nombre,
            "style": self.style,
            "configuracion_neuroacustica": asdict(self.configuracion_neuroacustica),
            "neurotransmisores": self.neurotransmisores_principales,
            "beat_base": self.beat_base,
            "nivel_activacion": self.nivel_activacion.value,
            "tipo_respuesta": self.tipo_respuesta.value,
            "campo_consciencia": self.campo_consciencia.value,
            "efectos": {
                "cognitivos": self.efectos_cognitivos,
                "emocionales": self.efectos_emocionales,
                "fisicos": self.efectos_fisicos,
                "energeticos": self.efectos_energeticos
            },
            "coherencia": self.calcular_coherencia_neuroacustica(),
            "metadata_extra": self.metadata_extra  # ✅ INCLUIR METADATA EXTRA
        }

class GestorPerfilesCampo:
    def __init__(self):
        self.version = VERSION
        self.perfiles: Dict[str, PerfilCampoV7] = {}
        self.categorias: Dict[CampoCosciencia, List[str]] = {}
        self.sinergias_mapeadas: Dict[str, Dict[str, float]] = {}
        self.cache_recomendaciones: Dict[str, Any] = {}
        self.estadisticas_uso: Dict[str, Any] = {
            "total_consultas": 0,
            "perfiles_mas_usados": {},
            "coherencia_promedio": 0.0
        }
        # ✅ NUEVO: Repositorio de metadata extra
        self.metadata_repository: Dict[str, Dict[str, Any]] = {}
        
        self._migrar_perfiles_v6()
        self._crear_perfiles_v7_exclusivos()
        self._calcular_sinergias_automaticas()
        self._organizar_por_categorias()
        self._configurar_integracion_aurora_v7()

    # ✅ NUEVA FUNCIÓN: FILTRACIÓN QUIRÚRGICA
    def _crear_perfil_filtrado(self, nombre: str, datos: Dict[str, Any]) -> PerfilCampoV7:
        """
        🎯 FILTRACIÓN QUIRÚRGICA: Crea perfil filtrando parámetros inválidos
        
        Esta función implementa la solución de raíz para el ERROR #3:
        - Filtra parámetros que no están en el constructor de PerfilCampoV7
        - Preserva metadata extra (como efectos_espirituales) para uso futuro
        - Mantiene la arquitectura limpia sin modificar el constructor
        """
        
        # Parámetros válidos para PerfilCampoV7 (basado en @dataclass)
        parametros_validos = {
            'nombre', 'descripcion', 'campo_consciencia', 'version', 'style',
            'configuracion_neuroacustica', 'neurotransmisores_simples', 
            'neurotransmisores_principales', 'neurotransmisores_moduladores',
            'beat_base', 'ondas_primarias', 'ondas_secundarias', 'patron_ondas',
            'nivel_activacion', 'tipo_respuesta', 'duracion_efecto_min',
            'duracion_optima_min', 'duracion_maxima_min', 'efectos_cognitivos',
            'efectos_emocionales', 'efectos_fisicos', 'efectos_energeticos',
            'mejores_momentos', 'ambientes_optimos', 'posturas_recomendadas',
            'perfiles_sinergicos', 'perfiles_antagonicos', 'secuencia_recomendada',
            'parametros_ajustables', 'adaptable_intensidad', 'adaptable_duracion',
            'base_cientifica', 'estudios_referencia', 'nivel_evidencia',
            'mecanismo_accion', 'contraindicaciones', 'precauciones',
            'poblacion_objetivo', 'complejidad_tecnica', 'recursos_requeridos',
            'compatibilidad_v6', 'metricas', 'aurora_director_compatible',
            'protocolo_inteligencia', 'metadata_extra'
        }
        
        # 1. Filtrar solo parámetros válidos
        datos_filtrados = {k: v for k, v in datos.items() if k in parametros_validos}
        
        # 2. Extraer metadata extra (parámetros no válidos pero valiosos)
        metadata_extra = {k: v for k, v in datos.items() if k not in parametros_validos}
        
        # 3. Crear perfil sin errores usando solo parámetros válidos
        try:
            perfil = PerfilCampoV7(nombre=nombre, **datos_filtrados)
            
            # 4. Preservar metadata extra si existe
            if metadata_extra:
                perfil.metadata_extra.update(metadata_extra)
                # Guardar en repositorio para acceso futuro
                self.metadata_repository[nombre] = metadata_extra
                logger.info(f"✅ Metadata extra preservada para {nombre}: {list(metadata_extra.keys())}")
            
            logger.info(f"✅ Perfil {nombre} creado exitosamente con filtración")
            return perfil
            
        except Exception as e:
            logger.error(f"❌ Error creando perfil filtrado {nombre}: {e}")
            # Fallback: crear perfil mínimo válido
            return PerfilCampoV7(
                nombre=nombre,
                descripcion=datos.get('descripcion', f'Perfil {nombre}'),
                campo_consciencia=datos.get('campo_consciencia', CampoCosciencia.COGNITIVO)
            )

    def _migrar_perfiles_v6(self):
        """✅ ACTUALIZADO: Migración V6 con filtración quirúrgica"""
        logger.info("🔄 Iniciando migración de perfiles V6 con filtración...")
        
        perfiles_data = {
            "autoestima": {
                "v6": {"style": "luminoso", "nt": ["Dopamina", "Serotonina"], "beat": 10},
                "v7": {
                    "descripcion": "Fortalecimiento del sentido de valor personal y confianza interior",
                    "campo_consciencia": CampoCosciencia.EMOCIONAL,
                    "nivel_activacion": NivelActivacion.MODERADO,
                    "efectos_emocionales": ["Confianza elevada", "Autoestima saludable", "Amor propio"],
                    "efectos_cognitivos": ["Pensamiento positivo", "Autoconcepto claro"],
                    "mejores_momentos": ["mañana", "tarde"],
                    "ambientes_optimos": ["privado", "tranquilo"],
                    "nivel_evidencia": 0.88,
                    "base_cientifica": CalidadEvidencia.VALIDADO,
                    # ✅ PARÁMETRO PROBLEMÁTICO QUE SE FILTRARÁ
                    "efectos_espirituales": ["Conexión con el ser superior", "Autotrascendencia"]
                }
            },
            "memoria": {
                "v6": {"style": "etereo", "nt": ["BDNF"], "beat": 12},
                "v7": {
                    "descripcion": "Optimización de la consolidación y recuperación de memoria",
                    "campo_consciencia": CampoCosciencia.COGNITIVO,
                    "nivel_activacion": NivelActivacion.MODERADO,
                    "efectos_cognitivos": ["Memoria mejorada", "Consolidación acelerada", "Recall optimizado"],
                    "mejores_momentos": ["mañana", "pre-estudio"],
                    "ambientes_optimos": ["silencioso", "organizado"],
                    "nivel_evidencia": 0.92,
                    "base_cientifica": CalidadEvidencia.CLINICO
                }
            },
            "sueño": {
                "v6": {"style": "nocturno", "nt": ["Melatonina", "GABA"], "beat": 2},
                "v7": {
                    "descripcion": "Inducción natural del sueño reparador y profundo",
                    "campo_consciencia": CampoCosciencia.FISICO,
                    "nivel_activacion": NivelActivacion.SUTIL,
                    "tipo_respuesta": TipoRespuesta.PROGRESIVA,
                    "efectos_fisicos": ["Relajación profunda", "Sueño reparador", "Descanso"],
                    "efectos_emocionales": ["Tranquilidad", "Paz mental"],
                    "mejores_momentos": ["noche", "pre-dormir"],
                    "ambientes_optimos": ["oscuro", "silencioso", "fresco"],
                    "nivel_evidencia": 0.95,
                    "base_cientifica": CalidadEvidencia.CLINICO
                }
            },
            "flow_creativo": {
                "v6": {"style": "inspirador", "nt": ["Dopamina", "Norepinefrina"], "beat": 8},
                "v7": {
                    "descripcion": "Estado de flujo óptimo para creatividad e inspiración",
                    "campo_consciencia": CampoCosciencia.CREATIVO,
                    "nivel_activacion": NivelActivacion.MODERADO,
                    "efectos_cognitivos": ["Creatividad expandida", "Flujo mental", "Inspiración"],
                    "efectos_emocionales": ["Entusiasmo", "Motivación", "Gozo creativo"],
                    "mejores_momentos": ["mañana", "tarde"],
                    "ambientes_optimos": ["inspirador", "luminoso"],
                    "nivel_evidencia": 0.85,
                    "base_cientifica": CalidadEvidencia.VALIDADO
                }
            },
            "meditacion": {
                "v6": {"style": "zen", "nt": ["GABA", "Serotonina"], "beat": 4},
                "v7": {
                    "descripcion": "Estado meditativo profundo y concentración plena",
                    "campo_consciencia": CampoCosciencia.ESPIRITUAL,
                    "nivel_activacion": NivelActivacion.PROFUNDO,
                    "tipo_respuesta": TipoRespuesta.PROFUNDA,
                    "efectos_cognitivos": ["Concentración profunda", "Mindfulness", "Claridad mental"],
                    "efectos_emocionales": ["Serenidad", "Equilibrio", "Paz interior"],
                    "mejores_momentos": ["mañana", "tarde", "noche"],
                    "ambientes_optimos": ["silencioso", "natural", "sagrado"],
                    "nivel_evidencia": 0.90,
                    "base_cientifica": CalidadEvidencia.CLINICO,
                    # ✅ OTRO PARÁMETRO PROBLEMÁTICO QUE SE FILTRARÁ
                    "efectos_espirituales": ["Conexión trascendental", "Expansión consciencia"]
                }
            },
            "claridad_mental": {
                "v6": {"style": "cristalino", "nt": ["Acetilcolina", "BDNF"], "beat": 15},
                "v7": {
                    "descripcion": "Optimización del rendimiento cognitivo y claridad mental",
                    "campo_consciencia": CampoCosciencia.COGNITIVO,
                    "nivel_activacion": NivelActivacion.INTENSO,
                    "efectos_cognitivos": ["Claridad mental", "Concentración aguda", "Pensamiento claro"],
                    "mejores_momentos": ["mañana", "trabajo"],
                    "ambientes_optimos": ["organizado", "luminoso"],
                    "nivel_evidencia": 0.87,
                    "base_cientifica": CalidadEvidencia.VALIDADO
                }
            }
        }
        
        # ✅ USAR FILTRACIÓN QUIRÚRGICA en lugar de construcción directa
        for nombre, data in perfiles_data.items():
            try:
                # Combinar datos V6 y V7
                datos_completos = {**data["v6"], **data["v7"]}
                
                # Convertir algunos valores a tipos correctos
                if "campo_consciencia" in datos_completos and isinstance(datos_completos["campo_consciencia"], str):
                    try:
                        datos_completos["campo_consciencia"] = CampoCosciencia(datos_completos["campo_consciencia"])
                    except ValueError:
                        datos_completos["campo_consciencia"] = CampoCosciencia.COGNITIVO
                
                # ✅ APLICAR FILTRACIÓN QUIRÚRGICA
                perfil = self._crear_perfil_filtrado(nombre, datos_completos)
                
                # Migrar datos V6 específicos
                perfil.migrar_desde_v6(data["v6"])
                
                self.perfiles[nombre] = perfil
                logger.info(f"✅ Perfil V6 migrado: {nombre}")
                
            except Exception as e:
                logger.error(f"❌ Error migrando perfil {nombre}: {e}")

    def _crear_perfiles_v7_exclusivos(self):
        """✅ ACTUALIZADO: Creación de perfiles V7 con filtración quirúrgica"""
        logger.info("🔄 Creando perfiles V7 exclusivos con filtración...")
        
        perfiles_v7_exclusivos = {
            "expansion_consciente": {
                "descripcion": "Expansión de la consciencia y percepción ampliada",
                "campo_consciencia": CampoCosciencia.ESPIRITUAL,
                "nivel_activacion": NivelActivacion.TRASCENDENTE,
                "tipo_respuesta": TipoRespuesta.INTEGRATIVA,
                "style": "cosmico",
                "beat_base": 7.0,
                "efectos_emocionales": ["Expansión consciencia", "Apertura perceptual"],
                "efectos_cognitivos": ["Pensamiento holístico", "Intuición expandida"],
                "mejores_momentos": ["noche", "luna_llena"],
                "ambientes_optimos": ["natural", "sagrado"],
                "nivel_evidencia": 0.75,
                "base_cientifica": CalidadEvidencia.EXPERIMENTAL,
                # ✅ PARÁMETROS PROBLEMÁTICOS QUE SE FILTRARÁN
                "efectos_espirituales": ["Conexión cósmica", "Expansión dimensional"],
                "efectos_chamánicos": ["Viaje interior", "Sabiduría ancestral"]
            },
            "enraizamiento": {
                "descripcion": "Conexión profunda con la tierra y activación de energía vital",
                "campo_consciencia": CampoCosciencia.ENERGETICO,
                "nivel_activacion": NivelActivacion.INTENSO,
                "style": "tribal",
                "beat_base": 10.0,
                "efectos_fisicos": ["Vitalidad", "Energía terrestre", "Fuerza física"],
                "efectos_emocionales": ["Estabilidad", "Confianza corporal"],
                "mejores_momentos": ["mañana", "pre-ejercicio"],
                "ambientes_optimos": ["natural", "al_aire_libre"],
                "nivel_evidencia": 0.82,
                "base_cientifica": CalidadEvidencia.VALIDADO
            },
            "autocuidado": {
                "descripcion": "Activación del sistema de autocuidado y amor propio",
                "campo_consciencia": CampoCosciencia.SANACION,
                "nivel_activacion": NivelActivacion.SUTIL,
                "tipo_respuesta": TipoRespuesta.INTEGRATIVA,
                "style": "organico",
                "beat_base": 6.5,
                "efectos_emocionales": ["Amor propio", "Compasión", "Cuidado interno"],
                "efectos_fisicos": ["Relajación", "Sanación"],
                "mejores_momentos": ["tarde", "cuando_necesario"],
                "ambientes_optimos": ["cálido", "acogedor"],
                "nivel_evidencia": 0.87,
                "base_cientifica": CalidadEvidencia.VALIDADO
            }
        }
        
        # ✅ USAR FILTRACIÓN QUIRÚRGICA
        for nombre, datos in perfiles_v7_exclusivos.items():
            try:
                perfil = self._crear_perfil_filtrado(nombre, datos)
                self.perfiles[nombre] = perfil
                logger.info(f"✅ Perfil V7 exclusivo creado: {nombre}")
            except Exception as e:
                logger.error(f"❌ Error creando perfil V7 {nombre}: {e}")

    def _calcular_sinergias_automaticas(self):
        """Calcula sinergias entre perfiles automáticamente"""
        logger.info("🔄 Calculando sinergias entre perfiles...")
        
        for nombre1, perfil1 in self.perfiles.items():
            sinergias = {}
            for nombre2, perfil2 in self.perfiles.items():
                if nombre1 != nombre2:
                    sinergia = self._calcular_sinergia_entre_perfiles(perfil1, perfil2)
                    if sinergia > 0.3:  # Solo sinergias significativas
                        sinergias[nombre2] = sinergia
            
            if sinergias:
                self.sinergias_mapeadas[nombre1] = sinergias

    def _calcular_sinergia_entre_perfiles(self, perfil1: PerfilCampoV7, perfil2: PerfilCampoV7) -> float:
        """Calcula sinergia entre dos perfiles"""
        try:
            sinergia = 0.0
            
            # Factor por campo de consciencia
            if perfil1.campo_consciencia == perfil2.campo_consciencia:
                sinergia += 0.3
            elif perfil1.campo_consciencia in [CampoCosciencia.EMOCIONAL, CampoCosciencia.COGNITIVO] and \
                 perfil2.campo_consciencia in [CampoCosciencia.EMOCIONAL, CampoCosciencia.COGNITIVO]:
                sinergia += 0.2
            
            # Factor por neurotransmisores compartidos
            nt1 = set(perfil1.neurotransmisores_principales.keys())
            nt2 = set(perfil2.neurotransmisores_principales.keys())
            interseccion = len(nt1.intersection(nt2))
            if interseccion > 0:
                sinergia += interseccion * 0.1
            
            # Factor por beats compatibles
            ratio_beats = abs(perfil1.beat_base - perfil2.beat_base) / max(perfil1.beat_base, perfil2.beat_base)
            if ratio_beats < 0.3:  # Beats similares
                sinergia += 0.2
            
            return min(sinergia, 1.0)
        except Exception as e:
            logger.warning(f"Error calculando sinergia entre {perfil1.nombre} y {perfil2.nombre}: {e}")
            return 0.0

    def _organizar_por_categorias(self):
        """Organiza perfiles por categorías de campo de consciencia"""
        for nombre, perfil in self.perfiles.items():
            campo = perfil.campo_consciencia
            if campo not in self.categorias:
                self.categorias[campo] = []
            self.categorias[campo].append(nombre)

    def _configurar_integracion_aurora_v7(self):
        """Configura integración con Aurora Director V7"""
        for perfil in self.perfiles.values():
            perfil.aurora_director_compatible = True
            perfil.protocolo_inteligencia = True

    # ✅ NUEVA FUNCIÓN: Acceso a metadata extra
    def obtener_metadata_extra(self, nombre_perfil: str) -> Dict[str, Any]:
        """Obtiene metadata extra de un perfil (como efectos_espirituales)"""
        return self.metadata_repository.get(nombre_perfil, {})

    def buscar_perfiles(self, criterios: Dict[str, Any]) -> List[PerfilCampoV7]:
        """Busca perfiles según criterios específicos"""
        resultados = []
        
        for perfil in self.perfiles.values():
            coincide = True
            
            if "campo" in criterios:
                if perfil.campo_consciencia.value != criterios["campo"]:
                    coincide = False
            
            if "efectos" in criterios and coincide:
                efectos_buscados = criterios["efectos"]
                efectos_perfil = (perfil.efectos_cognitivos + perfil.efectos_emocionales + 
                                perfil.efectos_fisicos + perfil.efectos_energeticos)
                if not any(efecto in " ".join(efectos_perfil).lower() for efecto in efectos_buscados):
                    coincide = False
            
            if "nivel_activacion" in criterios and coincide:
                if perfil.nivel_activacion.value != criterios["nivel_activacion"]:
                    coincide = False
            
            if coincide:
                resultados.append(perfil)
        
        # Ordenar por coherencia neuroacústica
        resultados.sort(key=lambda p: p.calcular_coherencia_neuroacustica(), reverse=True)
        return resultados

    def recomendar_secuencia_perfiles(self, objetivo: str, duracion_total_min: int = 60) -> List[str]:
        """Recomienda una secuencia de perfiles para un objetivo"""
        perfiles_relevantes = self.buscar_perfiles({"efectos": [objetivo.lower()]})
        
        if not perfiles_relevantes:
            return list(self.perfiles.keys())[:3]  # Fallback
        
        secuencia = []
        tiempo_acumulado = 0
        
        for perfil in perfiles_relevantes:
            if tiempo_acumulado + perfil.duracion_optima_min <= duracion_total_min:
                secuencia.append(perfil.nombre)
                tiempo_acumulado += perfil.duracion_optima_min
            
            if len(secuencia) >= 3:  # Máximo 3 perfiles en secuencia
                break
        
        return secuencia

    def obtener_perfil(self, nombre: str) -> Optional[PerfilCampoV7]:
        """Obtiene un perfil por nombre"""
        return self.perfiles.get(nombre)

    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesa un objetivo y retorna recomendaciones"""
        if contexto is None:
            contexto = {}
        
        perfiles_relevantes = self.buscar_perfiles({"efectos": [objetivo.lower()]})
        
        if not perfiles_relevantes:
            return {"error": f"No se encontraron perfiles para objetivo: {objetivo}"}
        
        perfil_principal = perfiles_relevantes[0]
        
        return {
            "objetivo": objetivo,
            "perfil_principal": perfil_principal.nombre,
            "configuracion_optimizada": perfil_principal.configurar_para_aurora_director(),
            "perfiles_alternativos": [p.nombre for p in perfiles_relevantes[1:4]],
            "secuencia_recomendada": self.recomendar_secuencia_perfiles(objetivo, contexto.get("duracion_total", 60)),
            "coherencia_neuroacustica": perfil_principal.calcular_coherencia_neuroacustica(),
            "evidencia_cientifica": perfil_principal.base_cientifica.value,
            "aurora_v7_optimizado": True
        }

    def obtener_alternativas(self, objetivo: str) -> List[str]:
        """Obtiene perfiles alternativos para un objetivo"""
        perfiles_relevantes = self.buscar_perfiles({"efectos": [objetivo]})
        return [p.nombre for p in perfiles_relevantes[:5]]

    def exportar_estadisticas(self) -> Dict[str, Any]:
        """Exporta estadísticas completas del gestor"""
        coherencia_total = sum(p.calcular_coherencia_neuroacustica() for p in self.perfiles.values())
        
        return {
            "version": self.version,
            "total_perfiles": len(self.perfiles),
            "perfiles_v6_migrados": len([p for p in self.perfiles.values() if p.compatibilidad_v6]),
            "perfiles_v7_exclusivos": len(self.perfiles) - len([p for p in self.perfiles.values() if p.compatibilidad_v6]),
            "perfiles_por_campo": {campo.value: len(perfiles) for campo, perfiles in self.categorias.items()},
            "neurotransmisores_utilizados": len(set().union(*[list(p.neurotransmisores_principales.keys()) for p in self.perfiles.values()])),
            "promedio_nivel_evidencia": sum(p.nivel_evidencia for p in self.perfiles.values()) / len(self.perfiles),
            "promedio_coherencia_neuroacustica": coherencia_total / len(self.perfiles),
            "sinergias_calculadas": sum(len(sinergias) for sinergias in self.sinergias_mapeadas.values()),
            "perfiles_alta_complejidad": len([p for p in self.perfiles.values() if p.complejidad_tecnica == "alto"]),
            "perfiles_validados_clinicamente": len([p for p in self.perfiles.values() if p.base_cientifica == CalidadEvidencia.CLINICO]),
            "aurora_director_v7_compatible": all(p.aurora_director_compatible for p in self.perfiles.values()),
            "protocolo_inteligencia_implementado": all(p.protocolo_inteligencia for p in self.perfiles.values()),
            "estadisticas_uso": self.estadisticas_uso,
            # ✅ NUEVA ESTADÍSTICA: Metadata preservada
            "metadata_extra_preservada": len(self.metadata_repository),
            "tipos_metadata_preservada": list(set().union(*[list(metadata.keys()) for metadata in self.metadata_repository.values()]))
        }
    
    # ===============================================================================
    # IMPLEMENTACIÓN DE GestorPerfilesInterface
    # ===============================================================================

    def listar_perfiles(self) -> List[str]:
        """
        Lista todos los perfiles disponibles (implementa GestorPerfilesInterface)
        
        Returns:
            Lista de nombres de perfiles
        """
        return list(self.perfiles.keys())

    def validar_perfil(self, perfil: Dict[str, Any]) -> bool:
        """
        Valida si un perfil de campo es correcto (implementa GestorPerfilesInterface)
        
        Args:
            perfil: Configuración del perfil a validar
            
        Returns:
            True si el perfil es válido
        """
        try:
            # Validación básica de tipo
            if not isinstance(perfil, dict):
                return False
            
            # Debe tener al menos un campo básico
            if len(perfil) == 0:
                return False
            
            # Si tiene nombre, validar que sea string válido
            if "nombre" in perfil:
                if not isinstance(perfil["nombre"], str) or len(perfil["nombre"].strip()) == 0:
                    return False
            
            # Si tiene style, validar que sea string válido
            if "style" in perfil:
                if not isinstance(perfil["style"], str):
                    return False
            
            # Si tiene neurotransmisores, validar estructura
            if "nt" in perfil or "neurotransmisores_simples" in perfil:
                nt_field = perfil.get("nt", perfil.get("neurotransmisores_simples"))
                if not isinstance(nt_field, list):
                    return False
            
            # Si tiene beat, validar que sea numérico válido
            if "beat" in perfil or "beat_base" in perfil:
                beat_field = perfil.get("beat", perfil.get("beat_base"))
                if not isinstance(beat_field, (int, float)) or beat_field <= 0:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando perfil de campo: {e}")
            return False

    def obtener_categoria(self) -> str:
        """Categoría del gestor de perfiles"""
        return "field_profiles"

# ✅ FUNCIONES DE COMPATIBILIDAD V6 (sin cambios)
def _generar_field_profiles_v6() -> Dict[str, Dict[str, Any]]:
    """Genera perfiles V6 para compatibilidad"""
    gestor = GestorPerfilesCampo()
    field_profiles = {}
    
    perfiles_v6_originales = [
        "autoestima", "memoria", "sueño", "flow_creativo", "meditacion", "claridad_mental",
        "expansion_consciente", "enraizamiento", "autocuidado", "liberacion_emocional",
        "conexion_espiritual", "gozo_vital"
    ]
    
    for nombre in perfiles_v6_originales:
        perfil = gestor.obtener_perfil(nombre)
        if perfil:
            field_profiles[nombre] = {
                "style": perfil.style,
                "nt": perfil.neurotransmisores_simples,
                "beat": perfil.beat_base
            }
    
    return field_profiles

# ✅ VARIABLES GLOBALES (sin cambios)
FIELD_PROFILES = _generar_field_profiles_v6()

# ✅ FUNCIONES DE INTERFAZ (sin cambios)
def crear_gestor_perfiles() -> GestorPerfilesCampo:
    """Crea y retorna una instancia del gestor de perfiles"""
    return GestorPerfilesCampo()

def obtener_perfil_campo(nombre: str) -> Optional[PerfilCampoV7]:
    """Obtiene un perfil de campo por nombre"""
    gestor = crear_gestor_perfiles()
    return gestor.obtener_perfil(nombre)

def listar_perfiles_disponibles() -> List[str]:
    """Lista todos los perfiles disponibles"""
    gestor = crear_gestor_perfiles()
    return list(gestor.perfiles.keys())

def buscar_perfiles_por_campo(campo: str) -> List[PerfilCampoV7]:
    """Busca perfiles por campo de consciencia"""
    gestor = crear_gestor_perfiles()
    try:
        campo_enum = CampoCosciencia(campo.lower())
        return gestor.buscar_perfiles({"campo": campo_enum.value})
    except ValueError:
        return []

def procesar_objetivo_usuario(objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
    """Procesa un objetivo del usuario y retorna recomendaciones"""
    gestor = crear_gestor_perfiles()
    return gestor.procesar_objetivo(objetivo, contexto or {})

# ✅ LOGGING FINAL
logger.info(f"✅ Field Profiles V7 inicializado - Versión: {VERSION}")
logger.info("🎯 Filtración quirúrgica implementada exitosamente")
logger.info("✅ ERROR #3 solucionado - Constructor PerfilCampoV7 protegido")
logger.info("📦 Metadata extra preservada para uso futuro")
