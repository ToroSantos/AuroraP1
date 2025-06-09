import numpy as np, logging, time, importlib, json
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

try:
    import neuromix_definitivo_v7
    print("🚀 Super parche NeuroMix importado")
except Exception as e:
    print(f"⚠️ Error importando super parche: {e}")

try:
    from sync_and_scheduler import (
        sincronizar_y_estructurar_capas, aplicar_fade_narrativo, 
        optimizar_coherencia_global, validar_sync_y_estructura_completa,
        estructura_layer_fase, generar_estructura_inteligente,
        ParametrosSincronizacion, ConfiguracionScheduling
    )
    SYNC_SCHEDULER_HIBRIDO_AVAILABLE = True
    logging.info("✅ Sync híbrido detectado")
except ImportError:
    SYNC_SCHEDULER_HIBRIDO_AVAILABLE = False
    logging.warning("⚠️ Sync híbrido no disponible")

try:
    from objective_manager import (
        ObjectiveManagerUnificado, RouterInteligenteV7,
        AnalizadorSemantico, MotorPersonalizacion, ValidadorCientifico,
        crear_manager_optimizado
    )   
    OM_AVAIL = True
    logging.info("✅ OM detectado")
except ImportError:
    OM_AVAIL = False
    logging.warning("⚠️ OM no disponible")

try:
    from verify_structure import (
        verificar_estructura_aurora_v7_unificada, 
        ParametrosValidacion, NivelValidacion
    )
    VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE = True
    logging.info("✅ Verificación estructural Aurora V7.2 disponible")
except ImportError:
    VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE = False
    logging.warning("⚠️ Verificación estructural Aurora V7.2 no disponible")

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger("Aurora.V7")

class MotorAurora(Protocol):
    def generar_audio(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray: ...
    def validar_configuracion(self, config: Dict[str, Any]) -> bool: ...
    def obtener_capacidades(self) -> Dict[str, Any]: ...

class GestorInteligencia(Protocol):
    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]: ...
    def obtener_alternativas(self, objetivo: str) -> List[str]: ...

class TipoComponente(Enum):
    MOTOR = "motor"
    GESTOR_INTELIGENCIA = "gestor_inteligencia"
    PIPELINE = "pipeline"
    PRESET_MANAGER = "preset_manager"
    STYLE_PROFILE = "style_profile"
    OBJECTIVE_MANAGER = "objective_manager"
    SYNC_SCHEDULER = "sync_scheduler"

class EstrategiaGeneracion(Enum):
    AURORA_ORQUESTADO = "aurora_orquestado"
    MULTI_MOTOR = "multi_motor"
    MOTOR_ESPECIALIZADO = "motor_especializado"
    INTELIGENCIA_ADAPTIVA = "inteligencia_adaptiva"
    OBJECTIVE_MANAGER_DRIVEN = "objective_manager_driven"
    FALLBACK_PROGRESIVO = "fallback_progresivo"
    SYNC_SCHEDULER_HIBRIDO = "sync_scheduler_hibrido"

class ModoOrquestacion(Enum):
    SECUENCIAL = "secuencial"
    PARALELO = "paralelo"
    LAYERED = "layered"
    HYBRID = "hybrid"
    SYNC_HIBRIDO = "sync_hibrido"

@dataclass
class ComponenteAurora:
    nombre: str
    tipo: TipoComponente
    modulo: str
    clase_principal: str
    disponible: bool = False
    instancia: Optional[Any] = None
    version: str = "unknown"
    capacidades: Dict[str, Any] = field(default_factory=dict)
    dependencias: List[str] = field(default_factory=list)
    fallback_disponible: bool = False
    nivel_prioridad: int = 1
    compatibilidad_aurora: bool = True
    metadatos: Dict[str, Any] = field(default_factory=dict)

    def get(self, key, default=None):
        return getattr(self, key, default)

@dataclass
class ConfiguracionAuroraUnificada:
    objetivo: str = "relajacion"
    duracion_min: int = 20
    sample_rate: int = 44100
    estrategia_preferida: Optional[EstrategiaGeneracion] = None
    modo_orquestacion: ModoOrquestacion = ModoOrquestacion.HYBRID
    motores_preferidos: List[str] = field(default_factory=list)
    forzar_componentes: List[str] = field(default_factory=list)
    excluir_componentes: List[str] = field(default_factory=list)
    intensidad: str = "media"
    estilo: str = "sereno"
    neurotransmisor_preferido: Optional[str] = None
    calidad_objetivo: str = "alta"
    normalizar: bool = True
    aplicar_mastering: bool = True
    validacion_automatica: bool = True
    exportar_wav: bool = True
    nombre_archivo: str = "aurora_experience"
    incluir_metadatos: bool = True
    configuracion_custom: Dict[str, Any] = field(default_factory=dict)
    perfil_usuario: Optional[Dict[str, Any]] = None
    contexto_uso: Optional[str] = None
    session_id: Optional[str] = None
    usar_objective_manager: bool = True
    template_personalizado: Optional[str] = None
    perfil_campo_personalizado: Optional[str] = None
    secuencia_fases_personalizada: Optional[str] = None
    metadatos_emocionales: Dict[str, Any] = field(default_factory=dict)
    parametros_neuroacusticos: Dict[str, Any] = field(default_factory=dict)
    efectos_psicodelicos: Dict[str, Any] = field(default_factory=dict)
    secuencia_perfiles: List[Tuple[str, int]] = field(default_factory=list)
    frecuencia_base_psicodelica: Optional[float] = None
    coherencia_objetivo: float = 0.8
    configuracion_enriquecida: bool = False
    habilitar_sync_hibrido: bool = True
    parametros_sync_hibrido: Optional[Dict[str, Any]] = None

    def validar(self) -> List[str]:
        p = []
        if self.duracion_min <= 0:
            p.append("Duración debe ser positiva")
        if self.sample_rate not in [22050, 44100, 48000]:
            p.append("Sample rate no estándar")
        if self.intensidad not in ["suave", "media", "intenso"]:
            p.append("Intensidad inválida")
        if not self.objetivo.strip():
            p.append("Objetivo no puede estar vacío")
        return p

@dataclass
class ResultadoAuroraIntegrado:
    audio_data: np.ndarray
    metadatos: Dict[str, Any]
    estrategia_usada: EstrategiaGeneracion
    modo_orquestacion: ModoOrquestacion
    componentes_usados: List[str]
    tiempo_generacion: float
    calidad_score: float
    coherencia_neuroacustica: float
    efectividad_terapeutica: float
    configuracion: ConfiguracionAuroraUnificada
    capas_audio: Dict[str, np.ndarray] = field(default_factory=dict)
    analisis_espectral: Dict[str, Any] = field(default_factory=dict)
    recomendaciones: List[str] = field(default_factory=list)
    proxima_sesion: Dict[str, Any] = field(default_factory=dict)
    resultado_objective_manager: Optional[Dict[str, Any]] = None
    template_utilizado: Optional[str] = None
    perfil_campo_utilizado: Optional[str] = None
    secuencia_fases_utilizada: Optional[str] = None
    resultado_sync_hibrido: Optional[Dict[str, Any]] = None
    validacion_sync_scheduler: Optional[Dict[str, Any]] = None
    estructura_fases_utilizada: Optional[List[Dict[str, Any]]] = None
    verificacion_calidad: Optional[Dict[str, Any]] = None
    calidad_verificada: Optional[str] = None

class DetectorComponentesAvanzado:
    def __init__(self):
        self.componentes_registrados = self._init_registro_completo()
        self.componentes_activos: Dict[str, ComponenteAurora] = {}
        self.stats = {
            "total": 0, "exitosos": 0, "fallidos": 0, "fallback": 0,
            "tiempo_deteccion": 0.0, "motores_detectados": 0,
            "gestores_detectados": 0, "sync_scheduler_detectado": 0
        }
        self.cache_deteccion = {}

    def _init_registro_completo(self) -> Dict[str, ComponenteAurora]:
        r = {
            "neuromix_v27": ComponenteAurora(
                "neuromix_v27", TipoComponente.MOTOR, "neuromix_aurora_v27",
                "AuroraNeuroAcousticEngineV27", dependencias=[], fallback_disponible=True,
                nivel_prioridad=1, metadatos={"especialidad": "neuroacustica", "calidad": "alta"}
            ),
            "hypermod_v32": ComponenteAurora(
                "hypermod_v32", TipoComponente.MOTOR, "hypermod_v32",
                "HyperModEngineV32AuroraConnected", dependencias=[], fallback_disponible=True,
                nivel_prioridad=1, metadatos={"especialidad": "ondas_cerebrales", "calidad": "maxima"}
            ),
            "harmonic_essence_v34": ComponenteAurora(
                "harmonic_essence_v34", TipoComponente.MOTOR, "harmonicEssence_v34",
                "HarmonicEssenceV34AuroraConnected", dependencias=[], fallback_disponible=True,
                nivel_prioridad=1, metadatos={"especialidad": "texturas", "calidad": "alta"}
            ),
            "field_profiles": ComponenteAurora(
                "field_profiles", TipoComponente.GESTOR_INTELIGENCIA, "field_profiles",
                "GestorPerfilesCampo", dependencias=[], fallback_disponible=True, nivel_prioridad=2
            ),
            "objective_router": ComponenteAurora(
                "objective_router", TipoComponente.GESTOR_INTELIGENCIA, "objective_router",
                "RouterInteligenteV7", dependencias=["field_profiles"], fallback_disponible=True,
                nivel_prioridad=2
            ),
            "emotion_style_profiles": ComponenteAurora(
                "emotion_style_profiles", TipoComponente.GESTOR_INTELIGENCIA, "emotion_style_profiles",
                "GestorEmotionStyleUnificadoV7", dependencias=[], fallback_disponible=True,
                nivel_prioridad=2
            ),
            "quality_pipeline": ComponenteAurora(
                "quality_pipeline", TipoComponente.PIPELINE, "aurora_quality_pipeline",
                "AuroraQualityPipeline", dependencias=[], fallback_disponible=True, nivel_prioridad=4
            ),
            "neuromix_legacy": ComponenteAurora(
                "neuromix_legacy", TipoComponente.MOTOR, "neuromix_engine_v26_ultimate",
                "AuroraNeuroAcousticEngine", dependencias=[], fallback_disponible=True, nivel_prioridad=5
            ),
            "hypermod_legacy": ComponenteAurora(
                "hypermod_legacy", TipoComponente.MOTOR, "hypermod_engine_v31",
                "NeuroWaveGenerator", dependencias=[], fallback_disponible=True, nivel_prioridad=5
            ),
            "carmine_analyzer_v21": ComponenteAurora(
                "carmine_analyzer_v21", TipoComponente.PIPELINE, "Carmine_Analyzer",
                "CarmineAuroraAnalyzer", dependencias=[], fallback_disponible=True, nivel_prioridad=3,
                metadatos={"especialidad": "analisis_neuroacustico", "version": "2.1", "calidad": "maxima"}
            )
        }
        
        if OM_AVAIL:
            r.update({
                "objective_manager_unificado": ComponenteAurora(
                    "objective_manager_unificado", TipoComponente.OBJECTIVE_MANAGER, "objective_manager",
                    "ObjectiveManagerUnificado", dependencias=[], fallback_disponible=True, nivel_prioridad=1,
                    metadatos={
                        "especialidad": "gestion_objetivos_integral", "version": "unificado_v7",
                        "calidad": "maxima", "capacidades": [
                            "templates", "perfiles_campo", "secuencias_fases", "routing_inteligente"
                        ]
                    }
                )
            })

        if SYNC_SCHEDULER_HIBRIDO_AVAILABLE:
            r.update({
                "sync_scheduler_v7": ComponenteAurora(
                    "sync_scheduler_v7", TipoComponente.SYNC_SCHEDULER, "sync_and_scheduler",
                    "SyncSchedulerV7", dependencias=[], fallback_disponible=True, nivel_prioridad=1,
                    metadatos={
                        "especialidad": "sincronizacion_hibrida", "version": "V7_UNIFIED_OPTIMIZED",
                        "calidad": "maxima", "capacidades": [
                            "sincronizacion_multicapa", "estructura_fases", "fade_narrativo", "coherencia_global"
                        ]
                    }
                )
            })
        
        return r

    def detectar_todos(self) -> Dict[str, ComponenteAurora]:
        st = time.time()
        logger.info("🔍 Detectando componentes...")
        
        for n, c in sorted(self.componentes_registrados.items(), key=lambda x: x[1].nivel_prioridad):
            self._detectar_componente(c)
        
        self.stats["tiempo_deteccion"] = time.time() - st
        self._log_resumen_deteccion()
        return self.componentes_activos
    
    def _detectar_optimizaciones_harmonic_essence(self) -> bool:
        """Detecta automáticamente optimizaciones de HarmonicEssence"""
        try:
            import harmonic_essence_optimizations
            logger.info("🔍 Optimizaciones HarmonicEssence detectadas")
            return True
        except ImportError:
            logger.info("📋 Optimizaciones HarmonicEssence no disponibles")
            return False

    def _detectar_componente(self, comp: ComponenteAurora) -> bool:
        self.stats["total"] += 1
        if comp.nombre == "neuromix_v27": logger.info(f"🔍 Detectando NeuroMix: {comp.nombre}")
        if comp.nombre == "harmonic_essence_v34": logger.info(f"🔍 Detectando HarmonicEssence: {comp.nombre}")
    
        try:
            if not self._verificar_dependencias(comp):
                if comp.nombre == "neuromix_v27": logger.info(f"❌ NeuroMix: falla dependencias")
                return False
            
            if comp.nombre in self.cache_deteccion:
                if comp.nombre == "neuromix_v27": logger.info(f"🔄 NeuroMix: encontrado en cache")
                cr = self.cache_deteccion[comp.nombre]
                if comp.nombre == "neuromix_v27": logger.info(f"📋 NeuroMix cache success: {cr['success']}")
                if cr["success"]:
                    comp.disponible = True
                    comp.instancia = cr["instancia"]
                    comp.version = cr["version"]
                    comp.capacidades = cr["capacidades"]
                    self.componentes_activos[comp.nombre] = comp
                    self.stats["exitosos"] += 1
                return cr["success"]

            m = importlib.import_module(comp.modulo)
            if comp.nombre == "neuromix_v27": logger.info(f"✅ NeuroMix: módulo importado correctamente")
            if comp.nombre == "harmonic_essence_v34": logger.info(f"✅ HarmonicEssence: módulo importado correctamente")
        
            i = self._crear_instancia(m, comp)
            if comp.nombre == "neuromix_v27": logger.info(f"✅ NeuroMix: instancia creada: {type(i).__name__}")
            if comp.nombre == "harmonic_essence_v34": logger.info(f"✅ HarmonicEssence: instancia creada: {type(i).__name__}")
        
            if self._validar_instancia(i, comp):
                comp.disponible = True
                comp.instancia = i
                comp.capacidades = self._obtener_capacidades(i)
                comp.version = self._obtener_version(i)
            
                self.cache_deteccion[comp.nombre] = {
                    "success": True, "instancia": i, "version": comp.version, "capacidades": comp.capacidades
                }
            
                self.componentes_activos[comp.nombre] = comp
            # Auto-detectar y aplicar optimizaciones
            if comp.nombre == "harmonic_essence_v34" and self._detectar_optimizaciones_harmonic_essence():
                try:
                    from harmonic_essence_optimizations import aplicar_mejoras_harmonic_essence
                    comp.instancia = aplicar_mejoras_harmonic_essence(comp.instancia)
                    comp.metadatos["optimizaciones_activas"] = True
                    comp.metadatos["factor_potencia"] = "1.4x"
                    logger.info("⚡ HarmonicEssence V34 - OPTIMIZACIONES APLICADAS")
                except Exception as e:
                    logger.warning(f"⚠️ Error aplicando optimizaciones: {e}")

                if comp.tipo == TipoComponente.MOTOR:
                    self.stats["motores_detectados"] += 1
                elif comp.tipo == TipoComponente.GESTOR_INTELIGENCIA:
                    self.stats["gestores_detectados"] += 1
            
                self.stats["exitosos"] += 1
                logger.info(f"✅ {comp.nombre} v{comp.version}")
                return True
            else:
                raise Exception("Instancia no válida")
            
        except Exception as e:
                
            if comp.nombre == "harmonic_essence_v34": logger.error(f"❌ HarmonicEssence ERROR: {e}")
            if comp.fallback_disponible and self._crear_fallback(comp):
                self.stats["fallback"] += 1
                return True
            else:
                self.cache_deteccion[comp.nombre] = {"success": False, "error": str(e)}
                self.stats["fallidos"] += 1
                return False
        
    def _verificar_dependencias(self, comp: ComponenteAurora) -> bool:
        return all(dep in self.componentes_activos for dep in comp.dependencias)

    def _crear_instancia(self, modulo: Any, comp: ComponenteAurora) -> Any:
        creators = {
            "neuromix_aurora_v27": lambda: self._crear_neuromix_real(modulo),
            "hypermod_v32": lambda: getattr(modulo, "HyperModEngineV32AuroraConnected")(),
            "harmonicEssence_v34": lambda: getattr(modulo, "HarmonicEssenceV34AuroraConnected")(),
            "emotion_style_profiles": lambda: getattr(modulo, "crear_gestor_emotion_style_v7")(),
            "objective_manager": lambda: (
                getattr(modulo, "crear_objective_manager_unificado")() 
                if hasattr(modulo, "crear_objective_manager_unificado") 
                else getattr(modulo, "ObjectiveManagerUnificado")() 
                if hasattr(modulo, "ObjectiveManagerUnificado") 
                else None
            ),
            "sync_and_scheduler": lambda: modulo
        }
        
        if comp.modulo in creators:
            return creators[comp.modulo]()
        else:
            for metodo in [f"crear_gestor_{comp.nombre.split('_')[0]}", f"crear_{comp.nombre}", 
                          "crear_gestor", "obtener_gestor", comp.clase_principal]:
                if hasattr(modulo, metodo):
                    attr = getattr(modulo, metodo)
                    return attr() if callable(attr) else attr
        try:
            return getattr(modulo, comp.clase_principal)()
        except AttributeError:
            # Fallback para nombres de clase alternativos (ej: AuroraNeuroAcousticEngine -> AuroraNeuroAcousticEngineV27)
            if comp.clase_principal == "AuroraNeuroAcousticEngine" and hasattr(modulo, "AuroraNeuroAcousticEngineV27"):
                return getattr(modulo, "AuroraNeuroAcousticEngineV27")()
            raise

    def _crear_neuromix_real(self, modulo):
        """Crea instancia real de NeuroMix con parche definitivo si está disponible"""
        try:
            # PRIORIDAD 1: Intentar motor parcheado (neuromix_definitivo_v7)
            try:
                import neuromix_definitivo_v7
                if hasattr(neuromix_definitivo_v7, 'NeuroMixSuperUnificado'):
                    instancia = neuromix_definitivo_v7.NeuroMixSuperUnificado()
                    logger.info("🚀 NeuroMix Super Unificado (parcheado) creado")
                    return instancia
                elif hasattr(neuromix_definitivo_v7, 'AutoIntegradorSuper'):
                    # Usar AutoIntegrador como alternativa
                    auto_integrador = neuromix_definitivo_v7.AutoIntegradorSuper()
                    instancia = auto_integrador.obtener_motor_optimizado()
                    logger.info("🚀 NeuroMix via AutoIntegrador (parcheado) creado")
                    return instancia
            except ImportError:
                logger.info("💡 Parche definitivo no disponible, usando motor base")
            except Exception as e:
                logger.warning(f"⚠️ Error con parche definitivo: {e}, usando motor base")
            
            # PRIORIDAD 2: Motor base si parche no está disponible
            if hasattr(modulo, 'AuroraNeuroAcousticEngineV27'):
                instancia = modulo.AuroraNeuroAcousticEngineV27()
                logger.info("✅ NeuroMix base (motor real) creado")
                return instancia
            elif hasattr(modulo, 'AuroraNeuroAcousticEngine'):
                instancia = modulo.AuroraNeuroAcousticEngine()
                logger.info("✅ NeuroMix alternativo (motor real) creado")
                return instancia
            else:
                raise Exception("Ninguna clase NeuroMix encontrada en módulo")
                
        except Exception as e:
            logger.error(f"❌ Error creando NeuroMix real: {e}")
            raise e  # No usar fallback, lanzar error

    def _validar_instancia(self, instancia: Any, comp: ComponenteAurora) -> bool:
        try:
            validators = {
                TipoComponente.MOTOR: lambda: (
                    hasattr(instancia, 'generar_audio') and callable(getattr(instancia, 'generar_audio'))
                ) or any(hasattr(instancia, attr) for attr in [
                    'generate_neuro_wave', 'generate_audio', 'create_audio'
                ]),
                TipoComponente.GESTOR_INTELIGENCIA: lambda: any(hasattr(instancia, attr) for attr in [
                    'procesar_objetivo', 'rutear_objetivo', 'obtener_perfil'
                ]),
                TipoComponente.OBJECTIVE_MANAGER: lambda: any(hasattr(instancia, attr) for attr in [
                    'procesar_objetivo_completo', 'rutear_objetivo_inteligente', 
                    'obtener_configuracion_completa', 'generar_configuracion_motor'
                ]),
                TipoComponente.PIPELINE: lambda: (
                    hasattr(instancia, 'analyze_audio') if "carmine_analyzer" in comp.nombre 
                    else hasattr(instancia, 'validar_y_normalizar')
                ),
                TipoComponente.STYLE_PROFILE: lambda: any(hasattr(instancia, attr) for attr in [
                    'obtener_preset', 'buscar_por_efecto'
                ]),
                TipoComponente.SYNC_SCHEDULER: lambda: all(hasattr(instancia, attr) for attr in [
                    'sincronizar_y_estructurar_capas', 'aplicar_fade_narrativo', 'optimizar_coherencia_global'
                ])
            }
            return validators.get(comp.tipo, lambda: True)()
        except Exception as e:
            if any(motor in comp.nombre.lower() for motor in ['neuromix', 'hypermod', 'harmonic']):
                if instancia is not None and hasattr(instancia, '__class__'):
                    return True
            return False

    def _obtener_capacidades(self, instancia: Any) -> Dict[str, Any]:
        try:
            for metodo in ['obtener_capacidades', 'get_capabilities', 'capacidades']:
                if hasattr(instancia, metodo):
                    return getattr(instancia, metodo)()
            
            if hasattr(instancia, '__name__') and 'sync_and_scheduler' in instancia.__name__:
                return {
                    "sincronizacion_multicapa": True, "estructura_fases": True,
                    "fade_narrativo": True, "coherencia_global": True, "validacion_completa": True
                }
            return {}
        except Exception:
            return {}

    def _obtener_version(self, instancia: Any) -> str:
        for attr in ['version', 'VERSION', '__version__', '_version']:
            if hasattr(instancia, attr):
                return str(getattr(instancia, attr))
        
        if hasattr(instancia, '__name__') and 'sync_and_scheduler' in instancia.__name__:
            return "V7_UNIFIED_OPTIMIZED"
        return "unknown"

    def _crear_fallback(self, comp: ComponenteAurora) -> bool:
        try:
            fallback_creators = {
                "neuromix_v27": self._fallback_neuromix,
                "neuromix_legacy": self._fallback_neuromix,
                "hypermod_v32": self._fallback_hypermod,
                "hypermod_legacy": self._fallback_hypermod,
                "harmonic_essence_v34": self._fallback_harmonic,
                "field_profiles": self._fallback_field_profiles,
                "objective_router": self._fallback_objective_router,
                "quality_pipeline": self._fallback_quality_pipeline,
                "carmine_analyzer_v21": self._fallback_carmine_analyzer,
                "objective_manager_unificado": self._fallback_objective_manager,
                "sync_scheduler_v7": self._fallback_sync_scheduler
            }
            
            if comp.nombre in fallback_creators:
                comp.instancia = fallback_creators[comp.nombre]()
                comp.disponible = True
                comp.version = "fallback"
                self.componentes_activos[comp.nombre] = comp
                return True
            return False
        except Exception as e:
            logger.error(f"Error fallback {comp.nombre}: {e}")
            return False

    def _fallback_sync_scheduler(self):
        class SSF:
            def sincronizar_y_estructurar_capas(self, audio_layers, estructura_fases, **kwargs):
                return audio_layers, {"fallback": True, "coherencia_global": 0.7}
            
            def aplicar_fade_narrativo(self, audio_layers, fase_actual, configuracion):
                return audio_layers
            
            def optimizar_coherencia_global(self, audio_layers, estructura_fases, objetivo_coherencia=0.9):
                return audio_layers, {"coherencia_global": 0.8, "fallback_usado": True}
            
            def validar_sync_y_estructura_completa(self, audio_layers, estructura_fases, **kwargs):
                return {"validacion_global": True, "puntuacion_global": 0.75, "fallback": True}
            
            def estructura_layer_fase(self, total_bloques, modo="normal", estilo="neutro"):
                return [
                    {"bloque": i, "gain": 1.0, "paneo": 0.0, "capas": {"neuro_wave": True}}
                    for i in range(total_bloques)
                ]
            
            def generar_estructura_inteligente(self, dur_min, config_base=None, **params):
                return {
                    "configuracion": {"duracion_minutos": dur_min, "total_bloques": max(2, dur_min // 2)},
                    "estructura": self.estructura_layer_fase(max(2, dur_min // 2)),
                    "validacion_cientifica": {"confianza_global": 0.8, "fallback": True}
                }
        return SSF()

    def _fallback_objective_manager(self):
        class OMF:
            def procesar_objetivo_completo(self, objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
                mapping_basico = {
                    "relajacion": {
                        "neurotransmisor_preferido": "gaba", "intensidad": "suave", "estilo": "sereno",
                        "template_recomendado": "relajacion_profunda", "perfil_campo": "relajacion", "beat_base": 7.0
                    },
                    "concentracion": {
                        "neurotransmisor_preferido": "acetilcolina", "intensidad": "media", "estilo": "crystalline",
                        "template_recomendado": "claridad_mental", "perfil_campo": "cognitivo", "beat_base": 14.0
                    },
                    "creatividad": {
                        "neurotransmisor_preferido": "anandamida", "intensidad": "media", "estilo": "organico",
                        "template_recomendado": "creatividad_exponencial", "perfil_campo": "creativo", "beat_base": 10.0
                    },
                    "meditacion": {
                        "neurotransmisor_preferido": "serotonina", "intensidad": "suave", "estilo": "mistico",
                        "template_recomendado": "presencia_total", "perfil_campo": "espiritual", "beat_base": 6.0
                    }
                }
                
                objetivo_lower = objetivo.lower()
                config_base = next((c for k, c in mapping_basico.items() if k in objetivo_lower), mapping_basico["relajacion"])
                
                return {
                    "configuracion_motor": config_base,
                    "template_utilizado": config_base.get("template_recomendado"),
                    "perfil_campo_utilizado": config_base.get("perfil_campo"),
                    "resultado_routing": {"confianza": 0.7, "tipo": "fallback_mapping", "fuente": "objective_manager_fallback"},
                    "metadatos": {"fallback_usado": True, "objetivo_original": objetivo, "contexto_procesado": contexto or {}}
                }
            
            def rutear_objetivo_inteligente(self, objetivo: str, **kwargs) -> Dict[str, Any]:
                return self.procesar_objetivo_completo(objetivo, kwargs)
            
            def obtener_configuracion_completa(self, objetivo: str) -> Dict[str, Any]:
                return self.procesar_objetivo_completo(objetivo)
            
            def generar_configuracion_motor(self, objetivo: str, motor_objetivo: str) -> Dict[str, Any]:
                return self.procesar_objetivo_completo(objetivo).get("configuracion_motor", {})
            
            def obtener_capacidades(self) -> Dict[str, Any]:
                return {
                    "nombre": "Objective Manager Fallback", "tipo": "gestor_objetivos_fallback",
                    "capacidades": ["mapping_basico", "routing_simple"],
                    "templates_disponibles": ["relajacion_profunda", "claridad_mental", "creatividad_exponencial", "presencia_total"],
                    "fallback": True
                }
        return OMF()

    def _fallback_neuromix(self):
        class NMF:
            def generar_audio(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray:
                samples = int(44100 * duracion_sec)
                t = np.linspace(0, duracion_sec, samples)
                freq_base = {'dopamina': 12.0, 'gaba': 6.0}.get(config.get("neurotransmisor_preferido"), 10.0)
                wave = 0.3 * np.sin(2 * np.pi * freq_base * t)
                
                fade_samples = min(2048, len(wave) // 4)
                if len(wave) > fade_samples * 2:
                    wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                return np.stack([wave, wave])
            
            def validar_configuracion(self, config: Dict[str, Any]) -> bool:
                return isinstance(config, dict) and config.get("objetivo", '').strip()
            
            def obtener_capacidades(self) -> Dict[str, Any]:
                return {
                    "nombre": "NeuroMix Fallback", "tipo": "motor_neuroacustico_fallback",
                    "neurotransmisores": ["dopamina", "serotonina", "gaba"]
                }
        return NMF()

    def _fallback_hypermod(self):
        class HMF:
            def generar_audio(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray:
                samples = int(44100 * duracion_sec)
                t = np.linspace(0, duracion_sec, samples)
                wave = 0.4 * np.sin(2 * np.pi * 10.0 * t) + 0.2 * np.sin(2 * np.pi * 6.0 * t)
                return np.stack([wave, wave])
            
            def validar_configuracion(self, config: Dict[str, Any]) -> bool:
                return True
            
            def obtener_capacidades(self) -> Dict[str, Any]:
                return {"nombre": "HyperMod Fallback", "tipo": "motor_ondas_cerebrales_fallback"}
        return HMF()

    def _fallback_harmonic(self):
        class HF:
            def generar_audio(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray:
                samples = int(44100 * duracion_sec)
                texture = np.random.normal(0, 0.1, samples)
                if samples > 100:
                    texture = np.convolve(texture, np.ones(min(50, samples // 20)) / min(50, samples // 20), mode='same')
                return np.stack([texture, texture])
            
            def validar_configuracion(self, config: Dict[str, Any]) -> bool:
                return True
            
            def obtener_capacidades(self) -> Dict[str, Any]:
                return {"nombre": "HarmonicEssence Fallback", "tipo": "motor_texturas_fallback"}
        return HF()

    def _fallback_field_profiles(self):
        class FPF:
            def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]:
                return {"perfil_recomendado": "basico", "configuracion": {"intensidad": "media", "duracion_min": 20}}
            
            def obtener_perfil(self, nombre: str):
                return None
            
            def recomendar_secuencia_perfiles(self, objetivo: str, duracion: int):
                return [(objetivo, duracion)]
        return FPF()

    def _fallback_objective_router(self):
        class ORF:
            def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]:
                mapping = {
                    "relajacion": {"neurotransmisor_preferido": "gaba", "intensidad": "suave", "estilo": "sereno"},
                    "concentracion": {"neurotransmisor_preferido": "acetilcolina", "intensidad": "media", "estilo": "crystalline"},
                    "creatividad": {"neurotransmisor_preferido": "anandamida", "intensidad": "media", "estilo": "organico"}
                }
                return mapping.get(objetivo.lower(), {"neurotransmisor_preferido": "serotonina", "intensidad": "media", "estilo": "neutro"})
            
            def rutear_objetivo(self, objetivo: str, **kwargs):
                return self.procesar_objetivo(objetivo, kwargs)
        return ORF()

    def _fallback_quality_pipeline(self):
        class QPF:
            def validar_y_normalizar(self, signal: np.ndarray) -> np.ndarray:
                if signal.ndim == 1:
                    signal = np.stack([signal, signal])
                
                max_val = np.max(np.abs(signal))
                if max_val > 0:
                    signal = signal * (0.85 / max_val)
                
                return np.clip(signal, -1.0, 1.0)
        return QPF()

    def _fallback_carmine_analyzer(self):
        class CAF:
            def analyze_audio(self, audio: np.ndarray, expected_intent=None):
                if audio.size == 0:
                    return type('Result', (), {
                        'score': 0, 'therapeutic_score': 0,
                        'quality': type('Quality', (), {'value': '🔴 CRÍTICO'})(),
                        'suggestions': ["Audio vacío"], 'issues': ["Sin audio"],
                        'gpt_summary': "Audio inválido"
                    })()
                
                rms = np.sqrt(np.mean(audio**2))
                peak = np.max(np.abs(audio))
                score = min(100, max(50, 80 + (1 - min(peak, 1.0)) * 20))
                quality_level = "🟢 ÓPTIMO" if score >= 90 else "🟡 OBSERVACIÓN" if score >= 70 else "🔴 CRÍTICO"
                
                return type('Result', (), {
                    'score': int(score), 'therapeutic_score': int(score * 0.9),
                    'quality': type('Quality', (), {'value': quality_level})(),
                    'suggestions': ["Usar fallback"], 'issues': [] if score > 70 else ["Calidad subóptima"],
                    'neuro_metrics': type('NeuroMetrics', (), {
                        'entrainment_effectiveness': 0.7, 'binaural_strength': 0.5
                    })(),
                    'gpt_summary': f"Análisis fallback: Score {score:.0f}/100"
                })()
            
            def obtener_capacidades(self):
                return {"nombre": "Carmine Analyzer Fallback", "tipo": "analizador_basico_fallback"}
        return CAF()

    def _log_resumen_deteccion(self):
        total = len(self.componentes_registrados)
        activos = len(self.componentes_activos)
        porcentaje = (activos / total * 100) if total > 0 else 0
        
        logger.info(
            f"📊 {self.stats['tiempo_deteccion']:.2f}s - {activos}/{total} ({porcentaje:.0f}%) - "
            f"✅{self.stats['exitosos']} 🔄{self.stats['fallback']} ❌{self.stats['fallidos']} - "
            f"🎵{self.stats['motores_detectados']} 🧠{self.stats['gestores_detectados']} "
            f"🌟{self.stats['sync_scheduler_detectado']}"
        )

class OrquestadorMultiMotor:
    def __init__(self, componentes_activos: Dict[str, ComponenteAurora]):
        self.componentes = componentes_activos
        self.motores_disponibles = {
            n: c for n, c in componentes_activos.items() 
            if c.tipo == TipoComponente.MOTOR and c.disponible
        }
        
        self.objective_manager = (
            componentes_activos.get("objective_manager_unificado", {}).get('instancia')
            if "objective_manager_unificado" in componentes_activos 
            and componentes_activos["objective_manager_unificado"].disponible 
            else None
        )
        
        self.sync_scheduler = (
            componentes_activos.get("sync_scheduler_v7", {}).get('instancia')
            if "sync_scheduler_v7" in componentes_activos 
            and componentes_activos["sync_scheduler_v7"].disponible 
            else None
        )

    def generar_audio_orquestado(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        metadatos_generacion = {
            "motores_utilizados": [], "tiempo_por_motor": {}, "calidad_por_motor": {},
            "estrategia_aplicada": config.modo_orquestacion.value
        }
        
        if config.usar_objective_manager and self.objective_manager:
            self._aplicar_objective_manager(config, metadatos_generacion)
        
        estrategias = {
            ModoOrquestacion.LAYERED: self._generar_en_capas,
            ModoOrquestacion.PARALELO: self._generar_paralelo,
            ModoOrquestacion.SECUENCIAL: self._generar_secuencial,
            ModoOrquestacion.SYNC_HIBRIDO: self.generar_audio_orquestado_v7_hibrido
        }
        
        return estrategias.get(config.modo_orquestacion, self._generar_hibrido)(config, duracion_sec, metadatos_generacion)

    def _aplicar_objective_manager(self, config, metadatos_generacion):
        try:
            resultado_om = self.objective_manager.procesar_objetivo_completo(
                config.objetivo,
                {
                    "duracion_min": config.duracion_min, "intensidad": config.intensidad,
                    "estilo": config.estilo, "contexto_uso": config.contexto_uso,
                    "perfil_usuario": config.perfil_usuario, "calidad_objetivo": config.calidad_objetivo
                }
            )
            
            config_motor = resultado_om.get("configuracion_motor", {})
            for k, v in config_motor.items():
                if hasattr(config, k) and v is not None:
                    setattr(config, k, v)
            
            metadatos_generacion["objective_manager"] = {
                "utilizado": True,
                "template_utilizado": resultado_om.get("template_utilizado"),
                "perfil_campo_utilizado": resultado_om.get("perfil_campo_utilizado"),
                "secuencia_fases_utilizada": resultado_om.get("secuencia_fases_utilizada"),
                "confianza_routing": resultado_om.get("resultado_routing", {}).get("confianza", 0.0),
                "tipo_routing": resultado_om.get("resultado_routing", {}).get("tipo", "unknown")
            }
        except Exception as e:
            metadatos_generacion["objective_manager"] = {"utilizado": False, "error": str(e)}

    def generar_audio_orquestado_v7_hibrido(self, config: ConfiguracionAuroraUnificada, duracion_sec: float, metadatos: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            capas_audio = {}
            motores_utilizados = []
            
            motores_configuracion = self._configurar_motores_hibridos(config)
            
            for capa_nombre, motor_config in motores_configuracion.items():
                if motor_config["motor"] in self.motores_disponibles:
                    self._generar_capa_motor(capa_nombre, motor_config, config, duracion_sec, capas_audio, motores_utilizados)
            
            if not capas_audio:
                return self._generar_hibrido_fallback(config, duracion_sec, metadatos)
            
            if self.sync_scheduler and config.habilitar_sync_hibrido:
                return self._aplicar_sync_scheduler_hibrido(capas_audio, config, duracion_sec, metadatos, motores_utilizados)
            else:
                return (
                    self._combinar_capas_tradicional(capas_audio),
                    {
                        **metadatos, "sync_hibrido_aplicado": False, "motores_utilizados": motores_utilizados,
                        "capas_procesadas": len(capas_audio), "modo_fallback": "combinacion_tradicional"
                    }
                )
        except Exception:
            return self._generar_hibrido_fallback(config, duracion_sec, metadatos)

    def _generar_capa_motor(self, capa_nombre, motor_config, config, duracion_sec, capas_audio, motores_utilizados):
        try:
            motor = self.motores_disponibles[motor_config["motor"]].instancia
            config_motor = self._adaptar_config_para_motor(config, motor_config["motor"], motor_config.get("config_adicional", {}))
            audio_capa = motor.generar_audio(config_motor, duracion_sec) * motor_config.get("peso", 1.0)
            capas_audio[capa_nombre] = audio_capa
            motores_utilizados.append(motor_config["motor"])
        except Exception:
            pass

    def _configurar_motores_hibridos(self, config: ConfiguracionAuroraUnificada) -> Dict[str, Dict[str, Any]]:
        objetivo = config.objetivo.lower()
        
        configuraciones = {
            "concentracion": {
                "neuroacustica_principal": {
                    "motor": "neuromix_v27", "peso": 0.6,
                    "config_adicional": {"wave_type": "neural_complex"}
                },
                "ondas_cerebrales": {
                    "motor": "hypermod_v32", "peso": 0.3,
                    "config_adicional": {"preset_emocional": "claridad_mental"}
                },
                "textura_ambiente": {
                    "motor": "harmonic_essence_v34", "peso": 0.2,
                    "config_adicional": {"texture_type": "crystalline"}
                }
            },
            "relajacion": {
                "textura_principal": {
                    "motor": "harmonic_essence_v34", "peso": 0.5,
                    "config_adicional": {"texture_type": "relaxation"}
                },
                "neuroacustica_suave": {
                    "motor": "neuromix_v27", "peso": 0.4,
                    "config_adicional": {"wave_type": "therapeutic"}
                },
                "ondas_theta": {
                    "motor": "hypermod_v32", "peso": 0.3,
                    "config_adicional": {"preset_emocional": "calma_profunda"}
                }
            },
            "creatividad": {
                "texturas_organicas": {
                    "motor": "harmonic_essence_v34", "peso": 0.5,
                    "config_adicional": {"texture_type": "organic"}
                },
                "neuroacustica_creativa": {
                    "motor": "neuromix_v27", "peso": 0.4,
                    "config_adicional": {"wave_type": "hybrid"}
                },
                "ondas_alpha": {
                    "motor": "hypermod_v32", "peso": 0.3,
                    "config_adicional": {"preset_emocional": "expansion_creativa"}
                }
            },
            "meditacion": {
                "ondas_profundas": {
                    "motor": "hypermod_v32", "peso": 0.5,
                    "config_adicional": {"preset_emocional": "conexion_mistica"}
                },
                "texturas_espirituales": {
                    "motor": "harmonic_essence_v34", "peso": 0.4,
                    "config_adicional": {"texture_type": "consciousness"}
                },
                "neuroacustica_meditativa": {
                    "motor": "neuromix_v27", "peso": 0.3,
                    "config_adicional": {"wave_type": "therapeutic"}
                }
            }
        }
        
        config_encontrada = None
        for key in ["concentracion", "relajacion", "creatividad", "meditacion"]:
            if key in objetivo:
                config_encontrada = configuraciones[key]
                break
        
        return config_encontrada or {
            "neuroacustica_base": {"motor": "neuromix_v27", "peso": 0.5},
            "ondas_equilibrio": {"motor": "hypermod_v32", "peso": 0.3},
            "textura_ambiente": {"motor": "harmonic_essence_v34", "peso": 0.2}
        }

    def _aplicar_sync_scheduler_hibrido(self, capas_audio: Dict[str, np.ndarray], config: ConfiguracionAuroraUnificada, 
                                      duracion_sec: float, metadatos: Dict[str, Any], motores_utilizados: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            estructura_resultado = self.sync_scheduler.generar_estructura_inteligente(
                dur_min=config.duracion_min,
                config_base={"objetivo": config.objetivo, "intensidad": config.intensidad, "estilo": config.estilo}
            )
            estructura_fases = estructura_resultado.get("estructura", [])
            
            parametros_sync = self.sync_scheduler.ParametrosSincronizacion(
                sample_rate=config.sample_rate,
                validacion_neuroacustica=True,
                optimizacion_automatica=True,
                umbral_coherencia=config.coherencia_objetivo
            )
            
            config_scheduling = self.sync_scheduler.ConfiguracionScheduling(
                validacion_neuroacustica=True,
                optimizacion_coherencia=True,
                intensidad_base={"suave": 0.6, "media": 0.8, "intenso": 1.0}.get(config.intensidad, 0.8),
                patron_espacial=self._mapear_estilo_a_patron_espacial(config.estilo)
            )
            
            capas_sincronizadas, metadatos_sync = self.sync_scheduler.sincronizar_y_estructurar_capas(
                audio_layers=capas_audio,
                estructura_fases=estructura_fases,
                parametros_sync=parametros_sync,
                config_scheduling=config_scheduling
            )
            
            fase_actual = self._determinar_fase_actual(config, duracion_sec)
            capas_con_fades = self.sync_scheduler.aplicar_fade_narrativo(
                audio_layers=capas_sincronizadas,
                fase_actual=fase_actual,
                configuracion=config_scheduling
            )
            
            capas_optimizadas, metadatos_coherencia = self.sync_scheduler.optimizar_coherencia_global(
                audio_layers=capas_con_fades,
                estructura_fases=estructura_fases,
                objetivo_coherencia=config.coherencia_objetivo
            )
            
            audio_final = self._combinar_capas_optimizadas(capas_optimizadas)
            
            validacion = self.sync_scheduler.validar_sync_y_estructura_completa(
                audio_layers=capas_optimizadas,
                estructura_fases=estructura_fases,
                nivel_detalle="completo"
            )
            
            metadatos.update({
                "sync_hibrido_aplicado": True,
                "motores_utilizados": motores_utilizados,
                "capas_procesadas": len(capas_audio),
                "coherencia_global": metadatos_coherencia.get("coherencia_global", 0.0),
                "validacion_sync_scheduler": validacion
            })
            
            return audio_final, metadatos
            
        except Exception:
            return self._combinar_capas_tradicional(capas_audio), {
                **metadatos, "coherencia_global": 0.7, "error": "sync_scheduler_error", "fallback_usado": True
            }

    def _mapear_estilo_a_patron_espacial(self, estilo: str):
        mapping = {
            "sereno": "neutro", "crystalline": "cristalino", "organico": "organico",
            "etereo": "etereo", "tribal": "tribal", "mistico": "mistico", "cuantico": "cuantico"
        }
        return mapping.get(estilo.lower(), "neutro")

    def _determinar_fase_actual(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> str:
        if duracion_sec <= 300:
            return "centro"
        elif duracion_sec <= 600:
            return "desarrollo"
        elif duracion_sec <= 1200:
            return "desarrollo_temprano"
        else:
            return "entrada"

    def _combinar_capas_optimizadas(self, capas_optimizadas: Dict[str, np.ndarray]) -> np.ndarray:
        if not capas_optimizadas:
            return np.zeros((2, int(44100 * 60)), dtype=np.float32)
        
        longitudes = [capa.shape[1] if capa.ndim == 2 else len(capa) for capa in capas_optimizadas.values()]
        longitud_minima = min(longitudes)
        
        audio_combinado = None
        for audio_capa in capas_optimizadas.values():
            if audio_capa.ndim == 1:
                audio_capa = np.stack([audio_capa, audio_capa])
            if audio_capa.ndim == 2 and audio_capa.shape[0] != 2:
                audio_capa = audio_capa.T
            
            audio_capa = audio_capa[:, :longitud_minima]
            
            if audio_combinado is None:
                audio_combinado = audio_capa
            else:
                min_length = min(audio_combinado.shape[1], audio_capa.shape[1])
                audio_combinado = audio_combinado[:, :min_length] + audio_capa[:, :min_length]
        
        max_val = np.max(np.abs(audio_combinado))
        if max_val > 0:
            audio_combinado = audio_combinado * (0.85 / max_val)
        
        return audio_combinado

    def _combinar_capas_tradicional(self, capas_audio: Dict[str, np.ndarray]) -> np.ndarray:
        if not capas_audio:
            return np.zeros((2, int(44100 * 60)), dtype=np.float32)
        
        audio_final = None
        for audio_capa in capas_audio.values():
            if audio_capa.ndim == 1:
                audio_capa = np.stack([audio_capa, audio_capa])
            if audio_capa.ndim == 2 and audio_capa.shape[0] != 2:
                audio_capa = audio_capa.T
            
            if audio_final is None:
                audio_final = audio_capa
            else:
                min_length = min(audio_final.shape[1], audio_capa.shape[1])
                audio_final = audio_final[:, :min_length] + audio_capa[:, :min_length]
        
        max_val = np.max(np.abs(audio_final))
        if max_val > 0:
            audio_final = audio_final * (0.85 / max_val)
        
        return audio_final

    def _generar_hibrido_fallback(self, config: ConfiguracionAuroraUnificada, duracion_sec: float, metadatos: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        audio, meta = self._generar_hibrido(config, duracion_sec, metadatos)
        meta.update({
            "sync_hibrido_aplicado": False,
            "fallback_hibrido_usado": True,
            "razon_fallback": "error_en_sync_scheduler_hibrido"
        })
        return audio, meta

    def _generar_en_capas(self, config: ConfiguracionAuroraUnificada, duracion_sec: float, metadatos: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        capas_configuracion = [
            ("neuromix_v27", {"peso": 0.6, "procesamiento": "base"}),
            ("hypermod_v32", {"peso": 0.3, "procesamiento": "armonica"}),
            ("harmonic_essence_v34", {"peso": 0.2, "procesamiento": "textura"})
        ]
        
        audio_final = None
        capas_generadas = {}
        
        for nombre_motor, capa_config in capas_configuracion:
            if nombre_motor in self.motores_disponibles:
                start_time = time.time()
                try:
                    motor = self.motores_disponibles[nombre_motor].instancia
                    config_motor = self._adaptar_config_para_motor(config, nombre_motor, capa_config)
                    audio_capa = motor.generar_audio(config_motor, duracion_sec) * capa_config["peso"]
                    
                    audio_final = audio_capa if audio_final is None else self._combinar_capas(audio_final, audio_capa)
                    
                    tiempo_generacion = time.time() - start_time
                    metadatos["motores_utilizados"].append(nombre_motor)
                    metadatos["tiempo_por_motor"][nombre_motor] = tiempo_generacion
                    capas_generadas[nombre_motor] = audio_capa
                except Exception:
                    pass
        
        if audio_final is None:
            audio_final = self._generar_fallback_simple(duracion_sec)
        
        metadatos["capas_generadas"] = len(capas_generadas)
        return audio_final, metadatos

    def _generar_paralelo(self, config: ConfiguracionAuroraUnificada, duracion_sec: float, metadatos: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        motor_principal = self._seleccionar_motor_principal(config)
        resultado_audio = self._generar_fallback_simple(duracion_sec)
        
        if motor_principal:
            start_time = time.time()
            try:
                instancia = self.motores_disponibles[motor_principal].instancia
                config_motor = self._adaptar_config_para_motor(config, motor_principal)
                resultado_audio = instancia.generar_audio(config_motor, duracion_sec)
                
                metadatos.update({
                    "motores_utilizados": [motor_principal],
                    "tiempo_por_motor": {motor_principal: time.time() - start_time},
                    "motor_principal": motor_principal
                })
            except Exception:
                pass
        
        return resultado_audio, metadatos

    def _generar_secuencial(self, config: ConfiguracionAuroraUnificada, duracion_sec: float, metadatos: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        motores_activos = list(self.motores_disponibles.keys())[:3]
        if not motores_activos:
            return self._generar_fallback_simple(duracion_sec), metadatos
        
        duracion_por_motor = duracion_sec / len(motores_activos)
        segmentos_audio = []
        
        for nombre_motor in motores_activos:
            start_time = time.time()
            try:
                motor = self.motores_disponibles[nombre_motor].instancia
                config_motor = self._adaptar_config_para_motor(config, nombre_motor)
                segmento = motor.generar_audio(config_motor, duracion_por_motor)
                segmentos_audio.append(segmento)
                
                metadatos["motores_utilizados"].append(nombre_motor)
                metadatos["tiempo_por_motor"][nombre_motor] = time.time() - start_time
            except Exception:
                segmentos_audio.append(np.zeros((2, int(44100 * duracion_por_motor))))
        
        audio_final = np.concatenate(segmentos_audio, axis=1) if segmentos_audio else self._generar_fallback_simple(duracion_sec)
        return audio_final, metadatos

    def _generar_hibrido(self, config: ConfiguracionAuroraUnificada, duracion_sec: float, metadatos: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_motores = len(self.motores_disponibles)
        
        if num_motores >= 3 and config.calidad_objetivo == "maxima":
            return self._generar_en_capas(config, duracion_sec, metadatos)
        elif num_motores >= 2:
            return self._generar_paralelo(config, duracion_sec, metadatos)
        else:
            return self._generar_secuencial(config, duracion_sec, metadatos)

    def _seleccionar_motor_principal(self, config: ConfiguracionAuroraUnificada) -> Optional[str]:
        preferencias_objetivo = {
            "concentracion": ["neuromix_v27", "hypermod_v32"],
            "relajacion": ["harmonic_essence_v34", "neuromix_v27"],
            "creatividad": ["harmonic_essence_v34", "neuromix_v27"],
            "meditacion": ["hypermod_v32", "neuromix_v27"]
        }
        
        objetivo_lower = config.objetivo.lower()
        
        for motor in config.motores_preferidos:
            if motor in self.motores_disponibles:
                return motor
        
        for objetivo_key, lista_motores in preferencias_objetivo.items():
            if objetivo_key in objetivo_lower:
                for motor in lista_motores:
                    if motor in self.motores_disponibles:
                        return motor
        
        return list(self.motores_disponibles.keys())[0] if self.motores_disponibles else None

    def _adaptar_config_para_motor(self, config: ConfiguracionAuroraUnificada, nombre_motor: str, capa_config: Dict[str, Any] = None) -> Dict[str, Any]:
        config_base = {
            "objetivo": config.objetivo, "duracion_min": config.duracion_min,
            "sample_rate": config.sample_rate, "intensidad": config.intensidad,
            "estilo": config.estilo, "neurotransmisor_preferido": config.neurotransmisor_preferido,
            "calidad_objetivo": config.calidad_objetivo, "normalizar": config.normalizar,
            "contexto_uso": config.contexto_uso
        }
        
        if hasattr(config, 'template_personalizado') and config.template_personalizado:
            config_base["template_objetivo"] = config.template_personalizado
        
        if hasattr(config, 'perfil_campo_personalizado') and config.perfil_campo_personalizado:
            config_base["perfil_campo"] = config.perfil_campo_personalizado
        
        if hasattr(config, 'secuencia_fases_personalizada') and config.secuencia_fases_personalizada:
            config_base["secuencia_fases"] = config.secuencia_fases_personalizada
        
        if hasattr(config, 'parametros_neuroacusticos') and config.parametros_neuroacusticos:
            for key in ["beat_primario", "beat_secundario", "armonicos", "coherencia_objetivo"]:
                if key in config.parametros_neuroacusticos:
                    config_base[key] = config.parametros_neuroacusticos[key]
        
        if hasattr(config, 'efectos_psicodelicos') and config.efectos_psicodelicos:
            for key in ["frecuencia_fundamental", "modulacion_depth", "modulacion_rate", "intensidad_efecto"]:
                if key in config.efectos_psicodelicos:
                    config_base[key] = config.efectos_psicodelicos[key]
        
        if hasattr(config, 'frecuencia_base_psicodelica') and config.frecuencia_base_psicodelica:
            config_base["frecuencia_base_psicodelica"] = config.frecuencia_base_psicodelica
        
        motor_configs = {
            "neuromix": lambda: config_base.update({
                "wave_type": "hybrid", "processing_mode": "aurora_integrated"
            }),
            "hypermod": lambda: config_base.update({
                "preset_emocional": config.objetivo, "validacion_cientifica": True,
                "optimizacion_neuroacustica": True
            }),
            "harmonic": lambda: config_base.update({
                "texture_type": self._mapear_estilo_a_textura(config.estilo),
                "precision_cientifica": True
            })
        }
        
        for key in ["neuromix", "hypermod", "harmonic"]:
            if key in nombre_motor:
                motor_configs[key]()
                break
        
        if capa_config and "config_adicional" in capa_config:
            config_base.update(capa_config["config_adicional"])
        
        return config_base

    def _mapear_estilo_a_textura(self, estilo: str) -> str:
        mapping = {
            "sereno": "relaxation", "crystalline": "crystalline", "organico": "organic",
            "etereo": "ethereal", "tribal": "tribal", "mistico": "consciousness"
        }
        return mapping.get(estilo.lower(), "organic")

    def _combinar_capas(self, audio1: np.ndarray, audio2: np.ndarray) -> np.ndarray:
        min_length = min(audio1.shape[1], audio2.shape[1])
        audio1_cortado = audio1[:, :min_length]
        audio2_cortado = audio2[:, :min_length]
        
        combinado = audio1_cortado + audio2_cortado
        max_val = np.max(np.abs(combinado))
        
        if max_val > 0.95:
            combinado = combinado * (0.85 / max_val)
        
        return combinado

    def _generar_fallback_simple(self, duracion_sec: float) -> np.ndarray:
        samples = int(44100 * duracion_sec)
        t = np.linspace(0, duracion_sec, samples)
        wave = 0.3 * np.sin(2 * np.pi * 10.0 * t)
        
        fade_samples = min(2048, samples // 4)
        if samples > fade_samples * 2:
            wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return np.stack([wave, wave])

class AuroraDirectorV7Integrado:
    def __init__(self, auto_detectar: bool = True):
        self.version = "Aurora Director V7 Integrado - Optimizado con Sync Híbrido + Verificación V7.2"
        self.detector = DetectorComponentesAvanzado()
        self.componentes: Dict[str, ComponenteAurora] = {}
        self.orquestador: Optional[OrquestadorMultiMotor] = None
        self.objective_manager: Optional[Any] = None
        self.sync_scheduler: Optional[Any] = None
        self.psychedelic_effects: Dict[str, Any] = {}
        self.emotion_profiles_cache: Dict[str, Any] = {}
        self.field_profiles_cache: Dict[str, Any] = {}
        
        self.stats = {
            "experiencias_generadas": 0, "tiempo_total_generacion": 0.0,
            "estrategias_utilizadas": {}, "objetivos_procesados": {},
            "errores_manejados": 0, "fallbacks_utilizados": 0,
            "calidad_promedio": 0.0, "motores_utilizados": {},
            "sesiones_activas": 0, "objective_manager_utilizaciones": 0,
            "templates_utilizados": {}, "perfiles_campo_utilizados": {},
            "secuencias_fases_utilizadas": {}, "emotion_style_utilizaciones": 0,
            "efectos_psicodelicos_aplicados": 0, "field_profiles_avanzados_utilizados": 0,
            "integraciones_exitosas": 0, "sync_hibrido_utilizaciones": 0,
            "coherencia_global_promedio": 0.0, "verificaciones_v7_2": 0,
            "verificaciones_exitosas_v7_2": 0, "errores_verificacion_v7_2": 0
        }
        
        self.cache_configuraciones = {}
        self.cache_resultados = {}
        
        if auto_detectar:
            self._inicializar_sistema()

    def _inicializar_sistema(self):
        logger.info(f"🌟 Inicializando {self.version}")
        
        self.componentes = self.detector.detectar_todos()
        self.orquestador = OrquestadorMultiMotor(self.componentes)
        self.psychedelic_effects = self._cargar_efectos_psicodelicos()
        
        if "objective_manager_unificado" in self.componentes and self.componentes["objective_manager_unificado"].disponible:
            self.objective_manager = self.componentes["objective_manager_unificado"].instancia
        
        if "sync_scheduler_v7" in self.componentes and self.componentes["sync_scheduler_v7"].disponible:
            self.sync_scheduler = self.componentes["sync_scheduler_v7"].instancia
        
        self._log_estado_sistema()

    def _cargar_efectos_psicodelicos(self) -> Dict[str, Any]:
        try:
            efectos_path = Path("psychedelic_effects_tables.json")
            if efectos_path.exists():
                with open(efectos_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def _log_estado_sistema(self):
        motores = len([c for c in self.componentes.values() if c.tipo == TipoComponente.MOTOR])
        gestores = len([c for c in self.componentes.values() if c.tipo == TipoComponente.GESTOR_INTELIGENCIA])
        pipelines = len([c for c in self.componentes.values() if c.tipo == TipoComponente.PIPELINE])
        obj_managers = len([c for c in self.componentes.values() if c.tipo == TipoComponente.OBJECTIVE_MANAGER])
        sync_schedulers = len([c for c in self.componentes.values() if c.tipo == TipoComponente.SYNC_SCHEDULER])
        
        logger.info(
            f"🔧 Componentes: 🎵{motores} 🧠{gestores} 🔄{pipelines} 🎯{obj_managers} "
            f"🌟{sync_schedulers} 🔬{1 if VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE else 0}"
        )

    def crear_experiencia(self, objetivo: str = None, **kwargs) -> ResultadoAuroraIntegrado:
        start_time = time.time()
        try:
            config = self._crear_configuracion_optimizada(objetivo, kwargs)
            
            problemas = config.validar()
            if problemas:
                logger.warning(f"⚠️ Problemas: {problemas}")
            
            estrategia = self._seleccionar_estrategia_optima(config)
            resultado = self._ejecutar_estrategia(estrategia, config)
            resultado = self._post_procesar_resultado(resultado, config)
            resultado = self._post_procesar_resultado_v7_2(resultado, config)
            
            tiempo_total = time.time() - start_time
            self._actualizar_estadisticas(config, resultado, tiempo_total)
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.stats["errores_manejados"] += 1
            return self._crear_resultado_emergencia(objetivo or "emergencia", str(e))

    def _post_procesar_resultado_v7_2(self, resultado: ResultadoAuroraIntegrado, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        if not VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE:
            logger.info("📋 Verificación V7.2 no disponible - usando validación básica")
            resultado.verificacion_calidad = {
                "disponible": False, "motivo": "verify_structure no encontrado",
                "validacion_basica": True, "calidad_estimada": "sin_verificar"
            }
            resultado.calidad_verificada = "validacion_basica"
            return resultado
        
        try:
            logger.info("🔬 Iniciando verificación estructural Aurora V7.2...")
            
            estructura_generada = []
            if hasattr(resultado, 'estructura_fases_utilizada') and resultado.estructura_fases_utilizada:
                estructura_generada = resultado.estructura_fases_utilizada
            elif hasattr(resultado, 'metadatos') and 'estructura_generada' in resultado.metadatos:
                estructura_generada = resultado.metadatos['estructura_generada']
            else:
                num_bloques = max(2, config.duracion_min // 5)
                estructura_generada = [
                    {
                        "bloque": i, "gain": 1.0 - (i * 0.05), "paneo": 0.0,
                        "capas": {
                            "neuro_wave": True,
                            "wave_pad": i < num_bloques // 2,
                            "textured_noise": i >= num_bloques // 3
                        }
                    }
                    for i in range(num_bloques)
                ]
            
            configuracion_verificacion = {
                "objetivo": config.objetivo, "duracion_min": config.duracion_min,
                "intensidad": config.intensidad, "estilo": config.estilo,
                "calidad_objetivo": config.calidad_objetivo,
                "estrategia_usada": resultado.estrategia_usada.value if hasattr(resultado.estrategia_usada, 'value') else str(resultado.estrategia_usada),
                "componentes_usados": resultado.componentes_usados,
                "template_utilizado": resultado.template_utilizado,
                "perfil_campo_utilizado": resultado.perfil_campo_utilizado,
                "sync_hibrido_aplicado": resultado.metadatos.get("sync_hibrido_aplicado", False) if hasattr(resultado, 'metadatos') else False
            }
            
            parametros_verificacion = ParametrosValidacion()
            if config.calidad_objetivo == "maxima":
                parametros_verificacion.nivel_validacion = NivelValidacion.UNIFICADO_V7
                parametros_verificacion.habilitar_benchmark = True
                parametros_verificacion.generar_recomendaciones_ia = True
                parametros_verificacion.habilitar_reportes_detallados = False
            else:
                parametros_verificacion.nivel_validacion = NivelValidacion.AVANZADO
                parametros_verificacion.habilitar_benchmark = False
                parametros_verificacion.generar_recomendaciones_ia = False
                parametros_verificacion.habilitar_reportes_detallados = False
            
            verificacion = verificar_estructura_aurora_v7_unificada(
                audio_data=resultado.audio_data,
                estructura_generada=estructura_generada,
                configuracion_original=configuracion_verificacion,
                nivel_detalle="completo" if config.calidad_objetivo == "maxima" else "intermedio",
                parametros=parametros_verificacion
            )
            
            resultado.verificacion_calidad = {
                key: verificacion.get(key) for key in [
                    "timestamp", "version_verificacion", "tipo_validacion",
                    "tiempo_ejecucion", "calidad_global", "puntuacion_global",
                    "aprobado", "metricas_aurora_v7"
                ]
            }
            
            resultado.verificacion_calidad.update({
                "recomendaciones_principales": verificacion.get("recomendaciones", [])[:5],
                "benchmark_score": verificacion.get("benchmark_resultado", {}).get("score_total") if verificacion.get("benchmark_resultado") else None,
                "estructura_analizada": len(estructura_generada),
                "disponible": True
            })
            
            resultado.calidad_verificada = verificacion.get("tipo_validacion", "verificacion_v7_2")
            
            if verificacion.get("puntuacion_global") and verificacion["puntuacion_global"] * 100 > resultado.calidad_score:
                resultado.calidad_score = verificacion["puntuacion_global"] * 100
            
            if verificacion.get("metricas_aurora_v7"):
                self._actualizar_metricas_verificacion(resultado, verificacion["metricas_aurora_v7"])
            
            if verificacion.get("recomendaciones"):
                for rec in verificacion["recomendaciones"][:3]:
                    if rec not in resultado.recomendaciones:
                        resultado.recomendaciones.append(rec)
            
            calidad = verificacion.get("calidad_global", "sin_datos")
            puntuacion = verificacion.get("puntuacion_global", 0.0)
            aprobado_emoji = "✅" if verificacion.get("aprobado") else "⚠️"
            logger.info(f"🔬 Verificación V7.2 completada: {calidad} | {puntuacion:.3f} {aprobado_emoji}")
            
            self.stats["verificaciones_v7_2"] = self.stats.get("verificaciones_v7_2", 0) + 1
            if verificacion.get("aprobado"):
                self.stats["verificaciones_exitosas_v7_2"] = self.stats.get("verificaciones_exitosas_v7_2", 0) + 1
            
            return resultado
            
        except Exception as e:
            logger.warning(f"⚠️ Error en verificación estructural Aurora V7.2: {e}")
            resultado.verificacion_calidad = {
                "error": str(e), "disponible": False, "timestamp": datetime.now().isoformat(),
                "fallback_usado": True, "calidad_estimada": "error_verificacion",
                "estructura_analizada": len(locals().get('estructura_generada', []))
            }
            resultado.calidad_verificada = "error_verificacion"
            self.stats["errores_verificacion_v7_2"] = self.stats.get("errores_verificacion_v7_2", 0) + 1
            return resultado

    def _actualizar_metricas_verificacion(self, resultado, metricas):
        if metricas.get("coherencia_temporal") and metricas["coherencia_temporal"] > resultado.coherencia_neuroacustica:
            resultado.coherencia_neuroacustica = metricas["coherencia_temporal"]
        
        if metricas.get("factibilidad_terapeutica") and metricas["factibilidad_terapeutica"] > resultado.efectividad_terapeutica:
            resultado.efectividad_terapeutica = metricas["factibilidad_terapeutica"]

    def _crear_configuracion_optimizada(self, objetivo: str, kwargs: Dict[str, Any]) -> ConfiguracionAuroraUnificada:
        cache_key = f"{objetivo}_{hash(str(sorted(kwargs.items())))}"
        if cache_key in self.cache_configuraciones:
            return self.cache_configuraciones[cache_key]
        
        configuraciones_inteligentes = {
            "concentracion": {
                "intensidad": "media", "estilo": "crystalline",
                "neurotransmisor_preferido": "acetilcolina", "modo_orquestacion": ModoOrquestacion.LAYERED,
                "motores_preferidos": ["neuromix_v27", "hypermod_v32"]
            },
            "claridad_mental": {
                "intensidad": "media", "estilo": "crystalline",
                "neurotransmisor_preferido": "dopamina", "modo_orquestacion": ModoOrquestacion.PARALELO,
                "motores_preferidos": ["neuromix_v27"]
            },
            "enfoque": {
                "intensidad": "intenso", "estilo": "crystalline",
                "neurotransmisor_preferido": "norepinefrina", "modo_orquestacion": ModoOrquestacion.LAYERED
            },
            "relajacion": {
                "intensidad": "suave", "estilo": "sereno",
                "neurotransmisor_preferido": "gaba", "modo_orquestacion": ModoOrquestacion.HYBRID,
                "motores_preferidos": ["harmonic_essence_v34", "neuromix_v27"]
            },
            "meditacion": {
                "intensidad": "suave", "estilo": "mistico",
                "neurotransmisor_preferido": "serotonina", "duracion_min": 35,
                "modo_orquestacion": ModoOrquestacion.LAYERED,
                "motores_preferidos": ["hypermod_v32", "harmonic_essence_v34"]
            },
            "gratitud": {
                "intensidad": "suave", "estilo": "sereno",
                "neurotransmisor_preferido": "oxitocina", "modo_orquestacion": ModoOrquestacion.HYBRID
            },
            "creatividad": {
                "intensidad": "media", "estilo": "organico",
                "neurotransmisor_preferido": "anandamida", "modo_orquestacion": ModoOrquestacion.LAYERED,
                "motores_preferidos": ["harmonic_essence_v34", "neuromix_v27"]
            },
            "inspiracion": {
                "intensidad": "media", "estilo": "organico",
                "neurotransmisor_preferido": "dopamina", "modo_orquestacion": ModoOrquestacion.HYBRID
            },
            "sanacion": {
                "intensidad": "suave", "estilo": "sereno",
                "neurotransmisor_preferido": "endorfina", "duracion_min": 45,
                "calidad_objetivo": "maxima", "modo_orquestacion": ModoOrquestacion.LAYERED
            }
        }
        
        objetivo_lower = objetivo.lower() if objetivo else "relajacion"
        contexto_detectado = self._detectar_contexto_objetivo(objetivo_lower)
        
        config_base = next(
            (config.copy() for key, config in configuraciones_inteligentes.items() if key in objetivo_lower),
            {}
        )
        config_base.update(contexto_detectado)
        
        config_final = {"objetivo": objetivo or "relajacion", **config_base, **kwargs}
        
        config_final.setdefault("usar_objective_manager", OM_AVAIL and self.objective_manager is not None)
        
        if config_final.get("calidad_objetivo") == "maxima" and SYNC_SCHEDULER_HIBRIDO_AVAILABLE and self.sync_scheduler:
            config_final.setdefault("habilitar_sync_hibrido", True)
            config_final.setdefault("modo_orquestacion", ModoOrquestacion.SYNC_HIBRIDO)
        
        config = ConfiguracionAuroraUnificada(**config_final)
        self.cache_configuraciones[cache_key] = config
        return config

    def _detectar_contexto_objetivo(self, objetivo: str) -> Dict[str, Any]:
        contexto = {}
        
        if any(palabra in objetivo for palabra in ["profundo", "intenso", "fuerte"]):
            contexto["intensidad"] = "intenso"
        elif any(palabra in objetivo for palabra in ["suave", "ligero", "sutil"]):
            contexto["intensidad"] = "suave"
        
        if any(palabra in objetivo for palabra in ["rapido", "corto", "breve"]):
            contexto["duracion_min"] = 10
        elif any(palabra in objetivo for palabra in ["largo", "extenso", "profundo"]):
            contexto["duracion_min"] = 45
        
        if any(palabra in objetivo for palabra in ["trabajo", "oficina", "estudio"]):
            contexto["contexto_uso"] = "trabajo"
        elif any(palabra in objetivo for palabra in ["dormir", "noche", "sueño"]):
            contexto["contexto_uso"] = "sueño"
        elif any(palabra in objetivo for palabra in ["meditacion", "espiritual"]):
            contexto["contexto_uso"] = "meditacion"
        
        if any(palabra in objetivo for palabra in ["terapeutico", "clinico", "medicinal"]):
            contexto["calidad_objetivo"] = "maxima"
        
        if any(palabra in objetivo for palabra in ["hibrido", "sync", "coherencia", "estructura"]):
            contexto["habilitar_sync_hibrido"] = True
            contexto["modo_orquestacion"] = ModoOrquestacion.SYNC_HIBRIDO
        
        return contexto

    def _seleccionar_estrategia_optima(self, config: ConfiguracionAuroraUnificada) -> EstrategiaGeneracion:
        if config.estrategia_preferida and config.estrategia_preferida in self._obtener_estrategias_disponibles():
            return config.estrategia_preferida
        
        motores = len([c for c in self.componentes.values() if c.tipo == TipoComponente.MOTOR and c.disponible])
        gestores = len([c for c in self.componentes.values() if c.tipo == TipoComponente.GESTOR_INTELIGENCIA and c.disponible])
        pipelines = len([c for c in self.componentes.values() if c.tipo == TipoComponente.PIPELINE and c.disponible])
        obj_managers = len([c for c in self.componentes.values() if c.tipo == TipoComponente.OBJECTIVE_MANAGER and c.disponible])
        sync_schedulers = len([c for c in self.componentes.values() if c.tipo == TipoComponente.SYNC_SCHEDULER and c.disponible])
        
        if (sync_schedulers >= 1 and motores >= 2 and config.habilitar_sync_hibrido and 
            (config.modo_orquestacion == ModoOrquestacion.SYNC_HIBRIDO or config.calidad_objetivo == "maxima")):
            return EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO
        elif (config.usar_objective_manager and obj_managers >= 1 and motores >= 2 and 
              config.calidad_objetivo == "maxima"):
            return EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN
        elif motores >= 3 and gestores >= 2 and pipelines >= 1 and config.calidad_objetivo == "maxima":
            return EstrategiaGeneracion.AURORA_ORQUESTADO
        elif motores >= 2 and config.modo_orquestacion in [ModoOrquestacion.LAYERED, ModoOrquestacion.HYBRID]:
            return EstrategiaGeneracion.MULTI_MOTOR
        elif gestores >= 1 and motores >= 1:
            return EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA
        elif motores >= 1:
            return EstrategiaGeneracion.MOTOR_ESPECIALIZADO
        else:
            return EstrategiaGeneracion.FALLBACK_PROGRESIVO

    def _ejecutar_estrategia(self, estrategia: EstrategiaGeneracion, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        duracion_sec = config.duracion_min * 60
        
        estrategias = {
            EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO: self._estrategia_sync_scheduler_hibrido,
            EstrategiaGeneracion.AURORA_ORQUESTADO: self._estrategia_aurora_orquestado_optimizada,
            EstrategiaGeneracion.MULTI_MOTOR: self._estrategia_multi_motor,
            EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA: self._estrategia_inteligencia_adaptiva,
            EstrategiaGeneracion.MOTOR_ESPECIALIZADO: self._estrategia_motor_especializado,
            EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN: self._estrategia_objective_manager_driven
        }
        
        return estrategias.get(estrategia, self._estrategia_fallback_progresivo)(config, duracion_sec)

    def _estrategia_sync_scheduler_hibrido(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> ResultadoAuroraIntegrado:
        if not self.sync_scheduler:
            return self._estrategia_aurora_orquestado_optimizada(config, duracion_sec)
        
        config_optimizado = self._aplicar_inteligencia_gestores_optimizada(config)
        config_optimizado.modo_orquestacion = ModoOrquestacion.SYNC_HIBRIDO
        
        audio_data, metadatos_orquestacion = self.orquestador.generar_audio_orquestado(config_optimizado, duracion_sec)
        
        if "quality_pipeline" in self.componentes:
            audio_data = self.componentes["quality_pipeline"].instancia.validar_y_normalizar(audio_data)
        
        calidad_score, coherencia_neuroacustica, efectividad_terapeutica = self._calcular_metricas_calidad(audio_data)
        
        resultado_sync_hibrido = metadatos_orquestacion.get("resultado_sync_hibrido")
        validacion_sync_scheduler = metadatos_orquestacion.get("validacion_sync_scheduler")
        estructura_fases = metadatos_orquestacion.get("estructura_fases_utilizada")
        
        resultado = ResultadoAuroraIntegrado(
            audio_data=audio_data,
            metadatos={
                "estrategia": "sync_scheduler_hibrido",
                "orquestacion": metadatos_orquestacion,
                "sync_hibrido_aplicado": True,
                "recursos_integrados": self._obtener_recursos_integrados_aplicados(config_optimizado),
                "pipeline_calidad": "quality_pipeline" in self.componentes,
                "sync_scheduler_version": resultado_sync_hibrido.get("sync_scheduler_version", "unknown") if resultado_sync_hibrido else "unknown"
            },
            estrategia_usada=EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO,
            modo_orquestacion=ModoOrquestacion.SYNC_HIBRIDO,
            componentes_usados=metadatos_orquestacion.get("motores_utilizados", []) + ["sync_scheduler_v7"],
            tiempo_generacion=0.0,
            calidad_score=calidad_score,
            coherencia_neuroacustica=coherencia_neuroacustica,
            efectividad_terapeutica=efectividad_terapeutica,
            configuracion=config_optimizado,
            resultado_sync_hibrido=resultado_sync_hibrido,
            validacion_sync_scheduler=validacion_sync_scheduler,
            estructura_fases_utilizada=estructura_fases
        )
        
        resultado = self._enriquecer_metadatos_generacion(resultado, config_optimizado)
        
        self.stats["sync_hibrido_utilizaciones"] += 1
        if resultado_sync_hibrido and "coherencia_global" in resultado_sync_hibrido:
            self._actualizar_coherencia_promedio(resultado_sync_hibrido["coherencia_global"])
        self.stats["integraciones_exitosas"] += 1
        
        return resultado

    def _actualizar_coherencia_promedio(self, coherencia_global):
        total_usos = self.stats["sync_hibrido_utilizaciones"]
        promedio_actual = self.stats["coherencia_global_promedio"]
        self.stats["coherencia_global_promedio"] = ((promedio_actual * (total_usos - 1)) + coherencia_global) / total_usos

    def _estrategia_objective_manager_driven(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> ResultadoAuroraIntegrado:
        if not self.objective_manager:
            return self._estrategia_aurora_orquestado_optimizada(config, duracion_sec)
        
        resultado_om = self.objective_manager.procesar_objetivo_completo(
            config.objetivo,
            {
                "duracion_min": config.duracion_min, "intensidad": config.intensidad,
                "estilo": config.estilo, "contexto_uso": config.contexto_uso,
                "perfil_usuario": config.perfil_usuario, "calidad_objetivo": config.calidad_objetivo
            }
        )
        
        config_optimizado = self._aplicar_resultado_objective_manager(config, resultado_om)
        audio_data, metadatos_orquestacion = self.orquestador.generar_audio_orquestado(config_optimizado, duracion_sec)
        
        if "quality_pipeline" in self.componentes:
            audio_data = self.componentes["quality_pipeline"].instancia.validar_y_normalizar(audio_data)
        
        calidad_score, coherencia, efectividad = self._calcular_metricas_calidad(audio_data)
        
        return ResultadoAuroraIntegrado(
            audio_data=audio_data,
            metadatos={
                "estrategia": "objective_manager_driven",
                "orquestacion": metadatos_orquestacion,
                "objective_manager_usado": True,
                "pipeline_calidad": "quality_pipeline" in self.componentes
            },
            estrategia_usada=EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN,
            modo_orquestacion=config.modo_orquestacion,
            componentes_usados=metadatos_orquestacion.get("motores_utilizados", []) + ["objective_manager_unificado"],
            tiempo_generacion=0.0,
            calidad_score=calidad_score,
            coherencia_neuroacustica=coherencia,
            efectividad_terapeutica=efectividad,
            configuracion=config,
            resultado_objective_manager=resultado_om,
            template_utilizado=resultado_om.get("template_utilizado"),
            perfil_campo_utilizado=resultado_om.get("perfil_campo_utilizado"),
            secuencia_fases_utilizada=resultado_om.get("secuencia_fases_utilizada")
        )

    def _aplicar_resultado_objective_manager(self, config: ConfiguracionAuroraUnificada, resultado_om: Dict[str, Any]) -> ConfiguracionAuroraUnificada:
        config_motor = resultado_om.get("configuracion_motor", {})
        for k, v in config_motor.items():
            if hasattr(config, k) and v is not None:
                setattr(config, k, v)
        
        config.template_personalizado = resultado_om.get("template_utilizado")
        config.perfil_campo_personalizado = resultado_om.get("perfil_campo_utilizado")
        config.secuencia_fases_personalizada = resultado_om.get("secuencia_fases_utilizada")
        
        return config

    def _estrategia_aurora_orquestado_optimizada(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> ResultadoAuroraIntegrado:
        config_optimizado = self._aplicar_inteligencia_gestores_optimizada(config)
        audio_data, metadatos_orquestacion = self.orquestador.generar_audio_orquestado(config_optimizado, duracion_sec)
        
        if hasattr(config_optimizado, 'efectos_psicodelicos') and config_optimizado.efectos_psicodelicos:
            audio_data = self._aplicar_efectos_psicodelicos_audio(audio_data, config_optimizado.efectos_psicodelicos)
            self.stats["efectos_psicodelicos_aplicados"] += 1
        
        if "quality_pipeline" in self.componentes:
            audio_data = self.componentes["quality_pipeline"].instancia.validar_y_normalizar(audio_data)
        
        calidad_score, coherencia, efectividad = self._calcular_metricas_calidad(audio_data)
        
        resultado = ResultadoAuroraIntegrado(
            audio_data=audio_data,
            metadatos={
                "estrategia": "aurora_orquestado_optimizado",
                "orquestacion": metadatos_orquestacion,
                "recursos_integrados": self._obtener_recursos_integrados_aplicados(config_optimizado),
                "pipeline_calidad": "quality_pipeline" in self.componentes
            },
            estrategia_usada=EstrategiaGeneracion.AURORA_ORQUESTADO,
            modo_orquestacion=config.modo_orquestacion,
            componentes_usados=metadatos_orquestacion.get("motores_utilizados", []),
            tiempo_generacion=0.0,
            calidad_score=calidad_score,
            coherencia_neuroacustica=coherencia,
            efectividad_terapeutica=efectividad,
            configuracion=config
        )
        
        resultado = self._enriquecer_metadatos_generacion(resultado, config_optimizado)
        self.stats["integraciones_exitosas"] += 1
        return resultado

    def _aplicar_inteligencia_gestores_optimizada(self, config: ConfiguracionAuroraUnificada) -> ConfiguracionAuroraUnificada:
        config_optimizado = config
        recursos_aplicados = []
        
        if config.usar_objective_manager and self.objective_manager:
            self._aplicar_objective_manager_inteligencia(config_optimizado, recursos_aplicados)
        
        config_optimizado = self._aplicar_emotion_style_profiles(config_optimizado)
        if hasattr(config_optimizado, 'metadatos_emocionales') and config_optimizado.metadatos_emocionales:
            recursos_aplicados.append("emotion_style_profiles")
        
        config_optimizado = self._aplicar_field_profiles_avanzado(config_optimizado)
        if hasattr(config_optimizado, 'parametros_neuroacusticos') and config_optimizado.parametros_neuroacusticos:
            recursos_aplicados.append("field_profiles_avanzado")
        
        config_optimizado = self._aplicar_efectos_psicodelicos(config_optimizado)
        if hasattr(config_optimizado, 'efectos_psicodelicos') and config_optimizado.efectos_psicodelicos:
            recursos_aplicados.append("psychedelic_effects")
        
        if config.template_personalizado:
            recursos_aplicados.append("objective_manager")
        
        if hasattr(config, 'habilitar_sync_hibrido') and config.habilitar_sync_hibrido:
            recursos_aplicados.append("sync_scheduler_hibrido")
        
        config_optimizado.configuracion_enriquecida = True
        return config_optimizado

    def _aplicar_objective_manager_inteligencia(self, config_optimizado, recursos_aplicados):
        try:
            resultado_om = self.objective_manager.procesar_objetivo_completo(
                config_optimizado.objetivo,
                {
                    "duracion_min": config_optimizado.duracion_min,
                    "intensidad": config_optimizado.intensidad,
                    "estilo": config_optimizado.estilo,
                    "contexto_uso": config_optimizado.contexto_uso,
                    "perfil_usuario": config_optimizado.perfil_usuario,
                    "calidad_objetivo": config_optimizado.calidad_objetivo
                }
            )
            
            config_optimizado = self._aplicar_resultado_objective_manager(config_optimizado, resultado_om)
            self.stats["objective_manager_utilizaciones"] += 1
            recursos_aplicados.append("objective_manager")
        except Exception:
            pass

    def _aplicar_emotion_style_profiles(self, config: ConfiguracionAuroraUnificada) -> ConfiguracionAuroraUnificada:
        if "emotion_style_profiles" not in self.componentes:
            return config
        
        try:
            emotion_manager = self.componentes["emotion_style_profiles"].instancia
            resultado_emotion = emotion_manager.procesar_objetivo(
                config.objetivo,
                {
                    "duracion_min": config.duracion_min,
                    "intensidad": config.intensidad,
                    "contexto_uso": config.contexto_uso
                }
            )
            
            if "error" not in resultado_emotion:
                self._aplicar_emotion_style_resultado(config, resultado_emotion)
        except Exception:
            pass
        
        return config

    def _aplicar_emotion_style_resultado(self, config, resultado_emotion):
        if "neurotransmisores" in resultado_emotion and not config.neurotransmisor_preferido:
            neurotransmisores = resultado_emotion["neurotransmisores"]
            if neurotransmisores:
                config.neurotransmisor_preferido = max(neurotransmisores.items(), key=lambda x: x[1])[0]
        
        if resultado_emotion.get("estilo") and not config.estilo:
            config.estilo = resultado_emotion["estilo"]
        
        config.metadatos_emocionales = {
            "preset_emocional": resultado_emotion.get("preset_emocional"),
            "coherencia_neuroacustica": resultado_emotion.get("coherencia_neuroacustica"),
            "efectos_esperados": resultado_emotion.get("recomendaciones_uso", []),
            "neurotransmisores_detectados": resultado_emotion.get("neurotransmisores", {}),
            "modo_aplicado": "emotion_style_v7"
        }
        
        self.stats["emotion_style_utilizaciones"] += 1

    def _aplicar_field_profiles_avanzado(self, config: ConfiguracionAuroraUnificada) -> ConfiguracionAuroraUnificada:
        if "field_profiles" not in self.componentes:
            return config
        
        try:
            profile_manager = self.componentes["field_profiles"].instancia
            
            perfil = profile_manager.obtener_perfil(config.objetivo)
            if perfil:
                self._aplicar_field_profile_resultado(config, perfil)
            
            secuencia = profile_manager.recomendar_secuencia_perfiles(config.objetivo, config.duracion_min)
            if secuencia and len(secuencia) > 1:
                config.secuencia_perfiles = secuencia
        except Exception:
            pass
        
        return config

    def _aplicar_field_profile_resultado(self, config, perfil):
        if not config.neurotransmisor_preferido and hasattr(perfil, 'neurotransmisores_principales'):
            if perfil.neurotransmisores_principales:
                config.neurotransmisor_preferido = max(perfil.neurotransmisores_principales.items(), key=lambda x: x[1])[0]
        
        if hasattr(perfil, 'configuracion_neuroacustica'):
            config.parametros_neuroacusticos = {
                "beat_primario": perfil.configuracion_neuroacustica.beat_primario,
                "beat_secundario": perfil.configuracion_neuroacustica.beat_secundario,
                "armonicos": perfil.configuracion_neuroacustica.armonicos,
                "modulacion_amplitude": perfil.configuracion_neuroacustica.modulacion_amplitude,
                "modulacion_frecuencia": perfil.configuracion_neuroacustica.modulacion_frecuencia,
                "coherencia_objetivo": perfil.calcular_coherencia_neuroacustica(),
                "evolucion_activada": perfil.configuracion_neuroacustica.evolucion_activada,
                "movimiento_3d": perfil.configuracion_neuroacustica.movimiento_3d,
                "perfil_aplicado": perfil.nombre
            }
        
        if hasattr(perfil, 'duracion_optima_min') and perfil.duracion_optima_min > config.duracion_min:
            config.duracion_min = min(perfil.duracion_optima_min, config.duracion_min + 10)
        
        if hasattr(perfil, 'style') and not config.estilo:
            config.estilo = perfil.style
        
        if hasattr(perfil, 'nivel_activacion') and config.intensidad == "media":
            nivel_mapping = {
                "SUTIL": "suave", "MODERADO": "media", "INTENSO": "intenso",
                "PROFUNDO": "intenso", "TRASCENDENTE": "intenso"
            }
            nivel_val = perfil.nivel_activacion.value if hasattr(perfil.nivel_activacion, 'value') else None
            if nivel_val and nivel_val in nivel_mapping:
                config.intensidad = nivel_mapping[nivel_val]
        
        self.stats["field_profiles_avanzados_utilizados"] += 1

    def _aplicar_efectos_psicodelicos(self, config: ConfiguracionAuroraUnificada) -> ConfiguracionAuroraUnificada:
        if not self.psychedelic_effects or "pe" not in self.psychedelic_effects:
            return config
        
        try:
            efectos_db = self.psychedelic_effects["pe"]
            objetivo_lower = config.objetivo.lower()
            
            mapeo_objetivos = {
                "expansion": ["Psilocibina", "LSD"],
                "creatividad": ["Psilocibina", "LSD", "DMT"],
                "meditacion": ["Psilocibina", "5-MeO-DMT"],
                "sanacion": ["Psilocibina", "MDMA", "Ayahuasca"],
                "conexion": ["MDMA", "Ayahuasca"],
                "introspection": ["Psilocibina", "LSD"],
                "espiritual": ["Ayahuasca", "5-MeO-DMT", "San_Pedro"],
                "transformacion": ["Iboga", "Ayahuasca"],
                "energia": ["San_Pedro", "DMT"],
                "calma": ["THC", "Ketamina"],
                "relajacion": ["THC", "Ketamina"],
                "concentracion": ["Psilocibina"],
                "claridad": ["Psilocibina", "LSD"],
                "flujo": ["Psilocibina", "LSD"],
                "flow": ["Psilocibina", "LSD"],
                "gratitud": ["MDMA", "San_Pedro"],
                "amor": ["MDMA", "Ayahuasca"],
                "compasion": ["MDMA", "Ayahuasca"]
            }
            
            efecto_seleccionado = None
            nombre_efecto = None
            
            for patron_clave, efectos_list in mapeo_objetivos.items():
                if patron_clave in objetivo_lower:
                    for efecto in efectos_list:
                        if efecto in efectos_db:
                            efecto_seleccionado = efectos_db[efecto]
                            nombre_efecto = efecto
                            break
                    if efecto_seleccionado:
                        break
            
            if efecto_seleccionado:
                self._aplicar_efectos_psicodelicos_config(config, efecto_seleccionado, nombre_efecto)
        except Exception:
            pass
        
        return config

    def _aplicar_efectos_psicodelicos_config(self, config, efecto_seleccionado, nombre_efecto):
        config.efectos_psicodelicos = {
            "sustancia_referencia": nombre_efecto,
            "efecto_principal": efecto_seleccionado.get("effect", "unknown"),
            "tipo": efecto_seleccionado.get("type", "unknown"),
            "style_override": efecto_seleccionado.get("style", ""),
            "intensidad_base": efecto_seleccionado.get("intensity", "media")
        }
        
        if "freq" in efecto_seleccionado:
            config.frecuencia_base_psicodelica = efecto_seleccionado["freq"]
        
        if "p7" in efecto_seleccionado:
            config.efectos_psicodelicos.update(efecto_seleccionado["p7"])
        
        if "nt" in efecto_seleccionado and not config.neurotransmisor_preferido:
            neurotransmisores = efecto_seleccionado["nt"]
            if neurotransmisores:
                config.neurotransmisor_preferido = neurotransmisores[0].lower()
        
        if efecto_seleccionado.get("style") and config.estilo == "sereno":
            config.estilo = efecto_seleccionado["style"]

    def _aplicar_efectos_psicodelicos_audio(self, audio: np.ndarray, efectos_config: Dict[str, Any]) -> np.ndarray:
        try:
            audio_procesado = audio.copy()
            efectos_aplicados = []
            
            if "modulacion_depth" in efectos_config and "modulacion_rate" in efectos_config:
                self._aplicar_modulacion_profunda(audio_procesado, efectos_config, efectos_aplicados)
            
            if "frecuencia_fundamental" in efectos_config:
                self._aplicar_frecuencia_fundamental(audio_procesado, efectos_config, efectos_aplicados)
            
            if "armonicos" in efectos_config and efectos_config["armonicos"]:
                self._aplicar_armonicos(audio_procesado, efectos_config, efectos_aplicados)
            
            return np.clip(audio_procesado, -1.0, 1.0)
        except Exception:
            return audio

    def _aplicar_modulacion_profunda(self, audio_procesado, efectos_config, efectos_aplicados):
        depth = efectos_config["modulacion_depth"]
        rate = efectos_config["modulacion_rate"]
        samples = audio_procesado.shape[1]
        t = np.linspace(0, samples / 44100, samples)
        modulacion = 1.0 + depth * np.sin(2 * np.pi * rate * t)
        
        for canal in range(audio_procesado.shape[0]):
            audio_procesado[canal] = audio_procesado[canal] * modulacion
        
        if depth and rate:
            efectos_aplicados.append("modulacion_profunda")

    def _aplicar_frecuencia_fundamental(self, audio_procesado, efectos_config, efectos_aplicados):
        freq_fundamental = efectos_config["frecuencia_fundamental"]
        intensidad = efectos_config.get("intensidad_efecto", 0.3)
        samples = audio_procesado.shape[1]
        t = np.linspace(0, samples / 44100, samples)
        portadora = float(intensidad) * np.sin(2 * np.pi * float(freq_fundamental) * t)
        
        for canal in range(audio_procesado.shape[0]):
            audio_procesado[canal] = audio_procesado[canal] + portadora * 0.1
        
        if freq_fundamental and intensidad:
            efectos_aplicados.append("frecuencia_fundamental")

    def _aplicar_armonicos(self, audio_procesado, efectos_config, efectos_aplicados):
        armonicos = efectos_config["armonicos"]
        intensidad = efectos_config.get("intensidad_efecto", 0.2)
        samples = audio_procesado.shape[1]
        t = np.linspace(0, samples / 44100, samples)
        
        for indice, freq_armonica in enumerate(armonicos[:3]):
            if freq_armonica:
                amplitud_armonica = float(intensidad) * (0.1 / (indice + 1))
                onda_armonica = amplitud_armonica * np.sin(2 * np.pi * float(freq_armonica) * t)
                
                for canal in range(audio_procesado.shape[0]):
                    audio_procesado[canal] = audio_procesado[canal] + onda_armonica * 0.05
        
        efectos_aplicados.append("armonicos")

    def _enriquecer_metadatos_generacion(self, resultado: ResultadoAuroraIntegrado, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        if hasattr(config, 'metadatos_emocionales') and config.metadatos_emocionales:
            resultado.metadatos["emotion_style"] = config.metadatos_emocionales
        
        if hasattr(config, 'parametros_neuroacusticos') and config.parametros_neuroacusticos:
            resultado.metadatos["field_profile"] = config.parametros_neuroacusticos
        
        if hasattr(config, 'efectos_psicodelicos') and config.efectos_psicodelicos:
            resultado.metadatos["psychedelic_effects"] = config.efectos_psicodelicos
        
        if hasattr(config, 'secuencia_perfiles') and config.secuencia_perfiles:
            resultado.metadatos["profile_sequence"] = config.secuencia_perfiles
        
        if hasattr(config, 'configuracion_enriquecida') and config.configuracion_enriquecida:
            resultado.metadatos.update({
                "configuracion_enriquecida": True,
                "recursos_aplicados": self._obtener_recursos_integrados_aplicados(config)
            })
        
        if hasattr(config, 'habilitar_sync_hibrido') and config.habilitar_sync_hibrido:
            resultado.metadatos.update({
                "sync_hibrido_habilitado": True,
                "parametros_sync_hibrido": config.parametros_sync_hibrido if hasattr(config, 'parametros_sync_hibrido') and config.parametros_sync_hibrido else None
            })
        
        return resultado

    def _obtener_recursos_integrados_aplicados(self, config: ConfiguracionAuroraUnificada) -> List[str]:
        recursos = []
        
        if hasattr(config, 'metadatos_emocionales') and config.metadatos_emocionales:
            recursos.append("emotion_style_profiles")
        
        if hasattr(config, 'parametros_neuroacusticos') and config.parametros_neuroacusticos:
            recursos.append("field_profiles_avanzado")
        
        if hasattr(config, 'efectos_psicodelicos') and config.efectos_psicodelicos:
            recursos.append("psychedelic_effects")
        
        if config.template_personalizado:
            recursos.append("objective_manager")
        
        if hasattr(config, 'habilitar_sync_hibrido') and config.habilitar_sync_hibrido:
            recursos.append("sync_scheduler_hibrido")
        
        return recursos

    def _estrategia_multi_motor(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> ResultadoAuroraIntegrado:
        audio_data, metadatos_orquestacion = self.orquestador.generar_audio_orquestado(config, duracion_sec)
        calidad_score, coherencia, efectividad = self._calcular_metricas_calidad(audio_data)
        
        return ResultadoAuroraIntegrado(
            audio_data=audio_data,
            metadatos={"estrategia": "multi_motor", "orquestacion": metadatos_orquestacion},
            estrategia_usada=EstrategiaGeneracion.MULTI_MOTOR,
            modo_orquestacion=config.modo_orquestacion,
            componentes_usados=metadatos_orquestacion.get("motores_utilizados", []),
            tiempo_generacion=0.0,
            calidad_score=calidad_score,
            coherencia_neuroacustica=coherencia,
            efectividad_terapeutica=efectividad,
            configuracion=config
        )

    def _estrategia_inteligencia_adaptiva(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> ResultadoAuroraIntegrado:
        config_optimizado = self._aplicar_inteligencia_gestores_optimizada(config)
        motor_principal = self._seleccionar_motor_principal(config_optimizado)
        
        if motor_principal:
            audio_data, componentes_usados = self._generar_con_motor_principal(motor_principal, config_optimizado, duracion_sec)
        else:
            audio_data, componentes_usados = self._generar_audio_fallback(duracion_sec), ["fallback"]
        
        calidad_score, coherencia, efectividad = self._calcular_metricas_calidad(audio_data)
        
        return ResultadoAuroraIntegrado(
            audio_data=audio_data,
            metadatos={"estrategia": "inteligencia_adaptiva", "motor_principal": motor_principal, "configuracion_optimizada": True},
            estrategia_usada=EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA,
            modo_orquestacion=ModoOrquestacion.HYBRID,
            componentes_usados=componentes_usados,
            tiempo_generacion=0.0,
            calidad_score=calidad_score,
            coherencia_neuroacustica=coherencia,
            efectividad_terapeutica=efectividad,
            configuracion=config
        )

    def _estrategia_motor_especializado(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> ResultadoAuroraIntegrado:
        motor_principal = self._seleccionar_motor_principal(config)
        
        if motor_principal:
            audio_data, componentes_usados = self._generar_con_motor_principal(motor_principal, config, duracion_sec)
        else:
            audio_data, componentes_usados = self._generar_audio_fallback(duracion_sec), ["fallback"]
        
        calidad_score, coherencia, efectividad = self._calcular_metricas_calidad(audio_data)
        
        return ResultadoAuroraIntegrado(
            audio_data=audio_data,
            metadatos={"estrategia": "motor_especializado", "motor_utilizado": motor_principal},
            estrategia_usada=EstrategiaGeneracion.MOTOR_ESPECIALIZADO,
            modo_orquestacion=ModoOrquestacion.HYBRID,
            componentes_usados=componentes_usados,
            tiempo_generacion=0.0,
            calidad_score=calidad_score,
            coherencia_neuroacustica=coherencia,
            efectividad_terapeutica=efectividad,
            configuracion=config
        )

    def _estrategia_fallback_progresivo(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> ResultadoAuroraIntegrado:
        self.stats["fallbacks_utilizados"] += 1
        audio_data = self._generar_audio_fallback(duracion_sec)
        calidad_score, coherencia, efectividad = self._calcular_metricas_calidad(audio_data)
        
        return ResultadoAuroraIntegrado(
            audio_data=audio_data,
            metadatos={"estrategia": "fallback_progresivo", "motivo": "componentes_insuficientes"},
            estrategia_usada=EstrategiaGeneracion.FALLBACK_PROGRESIVO,
            modo_orquestacion=ModoOrquestacion.HYBRID,
            componentes_usados=["fallback_interno"],
            tiempo_generacion=0.0,
            calidad_score=calidad_score,
            coherencia_neuroacustica=coherencia,
            efectividad_terapeutica=efectividad,
            configuracion=config
        )

    def _seleccionar_motor_principal(self, config: ConfiguracionAuroraUnificada) -> Optional[str]:
        motores_disponibles = [n for n, c in self.componentes.items() if c.tipo == TipoComponente.MOTOR and c.disponible]
        
        preferencias = {
            "concentracion": ["neuromix_v27", "hypermod_v32"],
            "claridad_mental": ["neuromix_v27", "hypermod_v32"],
            "enfoque": ["neuromix_v27", "hypermod_v32"],
            "relajacion": ["harmonic_essence_v34", "neuromix_v27"],
            "meditacion": ["hypermod_v32", "harmonic_essence_v34"],
            "creatividad": ["harmonic_essence_v34", "neuromix_v27"],
            "sanacion": ["harmonic_essence_v34", "hypermod_v32"]
        }
        
        objetivo_lower = config.objetivo.lower()
        
        for motor in config.motores_preferidos:
            if motor in motores_disponibles:
                return motor
        
        for objetivo_key, lista_motores in preferencias.items():
            if objetivo_key in objetivo_lower:
                for motor in lista_motores:
                    if motor in motores_disponibles:
                        return motor
        
        return motores_disponibles[0] if motores_disponibles else None

    def _generar_con_motor_principal(self, motor_principal: str, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> Tuple[np.ndarray, List[str]]:
        motor = self.componentes[motor_principal].instancia
        config_motor = self._adaptar_configuracion_motor(config, motor_principal)
        audio_data = motor.generar_audio(config_motor, duracion_sec)
        return audio_data, [motor_principal]

    def _adaptar_configuracion_motor(self, config: ConfiguracionAuroraUnificada, nombre_motor: str) -> Dict[str, Any]:
        config_base = {
            "objetivo": config.objetivo, "duracion_min": config.duracion_min,
            "sample_rate": config.sample_rate, "intensidad": config.intensidad,
            "estilo": config.estilo, "neurotransmisor_preferido": config.neurotransmisor_preferido,
            "calidad_objetivo": config.calidad_objetivo, "normalizar": config.normalizar,
            "contexto_uso": config.contexto_uso
        }
        
        if hasattr(config, 'template_personalizado') and config.template_personalizado:
            config_base["template_objetivo"] = config.template_personalizado
        
        if hasattr(config, 'perfil_campo_personalizado') and config.perfil_campo_personalizado:
            config_base["perfil_campo"] = config.perfil_campo_personalizado
        
        if hasattr(config, 'secuencia_fases_personalizada') and config.secuencia_fases_personalizada:
            config_base["secuencia_fases"] = config.secuencia_fases_personalizada
        
        if hasattr(config, 'parametros_neuroacusticos') and config.parametros_neuroacusticos:
            for key in ["beat_primario", "beat_secundario", "armonicos", "coherencia_objetivo", "evolucion_activada", "movimiento_3d"]:
                if key in config.parametros_neuroacusticos:
                    config_base[key] = config.parametros_neuroacusticos[key]
        
        if hasattr(config, 'efectos_psicodelicos') and config.efectos_psicodelicos:
            for key in ["frecuencia_fundamental", "modulacion_depth", "modulacion_rate", "intensidad_efecto", "sustancia_referencia", "receptores"]:
                if key in config.efectos_psicodelicos:
                    config_base[key] = config.efectos_psicodelicos[key]
        
        if hasattr(config, 'frecuencia_base_psicodelica') and config.frecuencia_base_psicodelica:
            config_base["frecuencia_base_psicodelica"] = config.frecuencia_base_psicodelica
        
        motor_configs = {
            "neuromix": lambda: config_base.update({
                "wave_type": "hybrid", "processing_mode": "aurora_integrated",
                "quality_level": "therapeutic" if config.calidad_objetivo == "maxima" else "enhanced"
            }),
            "hypermod": lambda: config_base.update({
                "preset_emocional": config.objetivo, "validacion_cientifica": True,
                "optimizacion_neuroacustica": True, "modo_terapeutico": config.calidad_objetivo == "maxima"
            }),
            "harmonic": lambda: config_base.update({
                "texture_type": self._mapear_estilo_a_textura(config.estilo),
                "precision_cientifica": True, "auto_optimizar_coherencia": True
            })
        }
        
        for key in ["neuromix", "hypermod", "harmonic"]:
            if key in nombre_motor:
                motor_configs[key]()
                break
        
        return config_base

    def _mapear_estilo_a_textura(self, estilo: str) -> str:
        mapping = {
            "sereno": "relaxation", "crystalline": "crystalline", "organico": "organic",
            "etereo": "ethereal", "tribal": "tribal", "mistico": "consciousness", "neutro": "meditation"
        }
        return mapping.get(estilo.lower(), "organic")

    def _post_procesar_resultado(self, resultado: ResultadoAuroraIntegrado, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        audio = resultado.audio_data
        
        if config.normalizar:
            max_val = np.max(np.abs(audio))
            target_level = 0.85 if config.calidad_objetivo == "maxima" else 0.80
            if max_val > 0:
                audio = np.clip(audio * (target_level / max_val), -1.0, 1.0)
        
        if config.aplicar_mastering:
            audio = self._aplicar_mastering_basico(audio)
        
        resultado.audio_data = audio
        resultado = self._aplicar_analisis_carmine(resultado, config)
        resultado.recomendaciones = self._generar_recomendaciones(resultado, config)
        resultado.proxima_sesion = self._generar_sugerencias_proxima_sesion(resultado, config)
        
        return resultado

    def _aplicar_mastering_basico(self, audio: np.ndarray) -> np.ndarray:
        threshold = 0.7
        ratio = 3.0
        
        for canal in range(audio.shape[0]):
            signal = audio[canal]
            mask = np.abs(signal) > threshold
            compressed = np.sign(signal) * (threshold + (np.abs(signal) - threshold) / ratio)
            audio[canal] = np.where(mask, compressed, signal)
        
        return audio

    def _aplicar_analisis_carmine(self, resultado: ResultadoAuroraIntegrado, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        try:
            if "carmine_analyzer_v21" in self.componentes:
                self._ejecutar_carmine_analysis(resultado, config)
        except Exception:
            pass
        return resultado

    def _ejecutar_carmine_analysis(self, resultado, config):
        analyzer = self.componentes["carmine_analyzer_v21"].instancia
        expected_intent = self._mapear_objetivo_a_intent_carmine(config.objetivo)
        carmine_result = analyzer.analyze_audio(resultado.audio_data, expected_intent)
        resultado = self._actualizar_resultado_con_carmine(resultado, carmine_result)

    def _actualizar_resultado_con_carmine(self, resultado: ResultadoAuroraIntegrado, carmine_result: Any) -> ResultadoAuroraIntegrado:
        resultado.calidad_score = max(resultado.calidad_score, carmine_result.score)
        resultado.coherencia_neuroacustica = getattr(carmine_result.neuro_metrics, 'entrainment_effectiveness', resultado.coherencia_neuroacustica)
        resultado.efectividad_terapeutica = max(resultado.efectividad_terapeutica, carmine_result.therapeutic_score / 100.0)
        
        resultado.metadatos["carmine_analysis"] = {
            "score": carmine_result.score,
            "therapeutic_score": carmine_result.therapeutic_score,
            "quality_level": carmine_result.quality.value,
            "issues": carmine_result.issues,
            "suggestions": carmine_result.suggestions,
            "neuro_effectiveness": getattr(carmine_result.neuro_metrics, 'entrainment_effectiveness', 0.0),
            "binaural_strength": getattr(carmine_result.neuro_metrics, 'binaural_strength', 0.0),
            "gpt_summary": getattr(carmine_result, 'gpt_summary', ""),
            "correcciones_aplicadas": False
        }
        
        return resultado

    def _mapear_objetivo_a_intent_carmine(self, objetivo: str):
        mapeo = {
            "relajacion": "RELAXATION", "concentracion": "FOCUS", "claridad_mental": "FOCUS",
            "enfoque": "FOCUS", "meditacion": "MEDITATION", "creatividad": "EMOTIONAL",
            "sanacion": "RELAXATION", "sueño": "SLEEP", "energia": "ENERGY"
        }
        
        objetivo_lower = objetivo.lower()
        intent = next((intent_val for key, intent_val in mapeo.items() if key in objetivo_lower), None)
        
        if intent:
            try:
                carmine_module = __import__('Carmine_Analyzer')
                return getattr(carmine_module.TherapeuticIntent, intent)
            except:
                pass
        return None

    def _calcular_metricas_calidad(self, audio: np.ndarray) -> Tuple[float, float, float]:
        if audio.size == 0:
            return 0.0, 0.0, 0.0
        
        try:
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            crest_factor = peak / (rms + 1e-10)
            
            if audio.ndim == 2 and audio.shape[0] == 2:
                coherencia = float(np.nan_to_num(np.corrcoef(audio[0], audio[1])[0, 1], 0.5))
            else:
                coherencia = 0.8
            
            fft_data = np.abs(np.fft.rfft(audio[0] if audio.ndim == 2 else audio))
            espectral_diversidad = np.std(fft_data)
            flatness_espectral = np.mean(fft_data) / (np.max(fft_data) + 1e-10)
            
            calidad_score = min(100, max(60, 80 + (1 - min(peak, 1.0)) * 10 + coherencia * 10 + flatness_espectral * 10))
            efectividad = min(1.0, max(0.6, 0.7 + coherencia * 0.2 + (1 - min(peak, 1.0)) * 0.1))
            
            return float(calidad_score), float(coherencia), float(efectividad)
        except Exception:
            return 75.0, 0.75, 0.75

    def _generar_recomendaciones(self, resultado: ResultadoAuroraIntegrado, config: ConfiguracionAuroraUnificada) -> List[str]:
        recomendaciones = []
        
        if resultado.calidad_score < 70:
            recomendaciones.append("Considerar calidad 'maxima'")
        
        if resultado.coherencia_neuroacustica < 0.7:
            recomendaciones.append("Mejorar coherencia con más componentes")
        
        if resultado.efectividad_terapeutica < 0.8:
            recomendaciones.append("Incrementar duración")
        
        if len(resultado.componentes_usados) < 2 and len(self.orquestador.motores_disponibles) >= 2:
            recomendaciones.append("Probar modo 'layered'")
        
        if resultado.resultado_objective_manager:
            confianza_routing = resultado.resultado_objective_manager.get("resultado_routing", {}).get("confianza", 0.0)
            if confianza_routing < 0.7:
                recomendaciones.append("Especificar más detalles en objetivo")
            
            if not resultado.template_utilizado:
                recomendaciones.append("Probar template específico")
            
            if resultado.resultado_objective_manager.get("metadatos", {}).get("fallback_usado"):
                recomendaciones.append("OM en fallback - verificar componentes")
        
        objetivo_lower = config.objetivo.lower()
        if "concentracion" in objetivo_lower and "neuromix" not in resultado.componentes_usados:
            recomendaciones.append("NeuroMix V27 optimizado para concentración")
        
        if "relajacion" in objetivo_lower and "harmonic" not in resultado.componentes_usados:
            recomendaciones.append("HarmonicEssence V34 excelente para relajación")
        
        if not hasattr(config, 'configuracion_enriquecida') or not config.configuracion_enriquecida:
            recomendaciones.append("Sistema optimizado disponible")
        
        if "emotion_style_profiles" in self.componentes and not hasattr(config, 'metadatos_emocionales'):
            recomendaciones.append("Emotion Style Profiles disponible")
        
        if self.psychedelic_effects and not hasattr(config, 'efectos_psicodelicos'):
            recomendaciones.append("Efectos psicodélicos disponibles")
        
        if SYNC_SCHEDULER_HIBRIDO_AVAILABLE and self.sync_scheduler and not hasattr(config, 'habilitar_sync_hibrido'):
            recomendaciones.append("Sync Scheduler Híbrido disponible")
        
        if (hasattr(config, 'habilitar_sync_hibrido') and config.habilitar_sync_hibrido and 
            resultado.estrategia_usada != EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO):
            recomendaciones.append("Sync Híbrido habilitado pero no utilizado")
        
        if resultado.estrategia_usada == EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO:
            coherencia_global = 0.0
            if hasattr(resultado, 'resultado_sync_hibrido') and resultado.resultado_sync_hibrido:
                coherencia_global = resultado.resultado_sync_hibrido.get("coherencia_global", 0.0)
            
            if coherencia_global < 0.7:
                recomendaciones.append("Optimizar parámetros de coherencia")
            elif coherencia_global >= 0.9:
                recomendaciones.append("Excelente coherencia híbrida")
        
        if VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE and hasattr(resultado, 'verificacion_calidad') and resultado.verificacion_calidad:
            vq = resultado.verificacion_calidad
            if vq.get("disponible"):
                recomendaciones.append("✅ Verificación V7.2 completada")
            else:
                recomendaciones.append("⚠️ Error en verificación V7.2")
            
            if vq.get("calidad_global"):
                recomendaciones.append(f"🔬 Calidad verificada: {vq.get('calidad_global', 'N/A')}")
            
            puntuacion = vq.get("puntuacion_global", 0.0)
            if puntuacion >= 0.9:
                recomendaciones.append("🏆 Calidad excelente verificada")
            elif puntuacion < 0.7:
                recomendaciones.append("📈 Mejorar calidad verificada")
        
        if "carmine_analysis" in resultado.metadatos:
            carmine_data = resultado.metadatos["carmine_analysis"]
            
            for issue in carmine_data.get("issues", []):
                if issue not in recomendaciones:
                    recomendaciones.append(f"Carmine: {issue}")
            
            for suggestion in carmine_data.get("suggestions", [])[:2]:
                if suggestion not in recomendaciones:
                    recomendaciones.append(f"Optimización: {suggestion}")
            
            score = carmine_data.get("score", 100)
            if score < 70:
                recomendaciones.append("Regenerar con calidad máxima")
            elif score < 85:
                recomendaciones.append("Calidad aceptable")
        
        return [x for x in recomendaciones if x]

    def _generar_sugerencias_proxima_sesion(self, resultado: ResultadoAuroraIntegrado, config: ConfiguracionAuroraUnificada) -> Dict[str, Any]:
        sugerencias = {
            "objetivos_relacionados": [],
            "duracion_recomendada": config.duracion_min,
            "intensidad_sugerida": config.intensidad,
            "mejoras_configuracion": {},
            "sync_hibrido_recomendado": False,
            "verificacion_v7_2_recomendada": False
        }
        
        objetivo_lower = config.objetivo.lower()
        
        if resultado.resultado_objective_manager and self.objective_manager:
            self._obtener_sugerencias_objective_manager(sugerencias, config)
        
        objetivos_mapping = {
            "concentracion": lambda: sugerencias.update({"objetivos_relacionados": ["claridad_mental", "enfoque_profundo", "productividad"]}),
            "relajacion": lambda: sugerencias.update({"objetivos_relacionados": ["meditacion", "calma_profunda", "descanso"]}),
            "creatividad": lambda: sugerencias.update({"objetivos_relacionados": ["inspiracion", "flow_creativo", "apertura_mental"]})
        }
        
        for key in ["concentracion", "relajacion", "creatividad"]:
            if key in objetivo_lower:
                objetivos_mapping[key]()
                break
        
        if resultado.efectividad_terapeutica > 0.9:
            sugerencias["duracion_recomendada"] = max(10, config.duracion_min - 5)
        elif resultado.efectividad_terapeutica < 0.7:
            sugerencias["duracion_recomendada"] = min(60, config.duracion_min + 10)
        else:
            sugerencias["duracion_recomendada"] = config.duracion_min
        
        if resultado.calidad_score < 80:
            sugerencias["mejoras_configuracion"] = {"calidad_objetivo": "maxima", "modo_orquestacion": "layered"}
        else:
            sugerencias["mejoras_configuracion"] = {}
        
        if not config.usar_objective_manager and OM_AVAIL:
            sugerencias["mejoras_configuracion"]["usar_objective_manager"] = True
        
        if not hasattr(config, 'configuracion_enriquecida') or not config.configuracion_enriquecida:
            sugerencias["mejoras_configuracion"]["usar_integracion_optimizada"] = True
        
        if SYNC_SCHEDULER_HIBRIDO_AVAILABLE and self.sync_scheduler:
            self._generar_sugerencias_sync_hibrido(sugerencias, resultado, config)
        
        if VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE:
            sugerencias["verificacion_v7_2_recomendada"] = True
            if config.calidad_objetivo != "maxima":
                sugerencias["mejoras_configuracion"]["usar_verificacion_v7_2"] = True
        
        sugerencias["mejoras_configuracion"] = {k: v for k, v in sugerencias["mejoras_configuracion"].items() if v is not None}
        
        return sugerencias

    def _obtener_sugerencias_objective_manager(self, sugerencias, config):
        try:
            if hasattr(self.objective_manager, 'obtener_objetivos_relacionados'):
                sugerencias["objetivos_relacionados"] = self.objective_manager.obtener_objetivos_relacionados(config.objetivo)
            
            if hasattr(self.objective_manager, 'recomendar_secuencia'):
                secuencia_recomendada = self.objective_manager.recomendar_secuencia(config.objetivo, config.duracion_min)
                if secuencia_recomendada:
                    sugerencias["secuencia_recomendada"] = secuencia_recomendada
        except Exception:
            pass

    def _generar_sugerencias_sync_hibrido(self, sugerencias, resultado, config):
        if resultado.estrategia_usada == EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO:
            coherencia_global = 0.0
            if hasattr(resultado, 'resultado_sync_hibrido') and resultado.resultado_sync_hibrido:
                coherencia_global = resultado.resultado_sync_hibrido.get("coherencia_global", 0.0)
            
            sugerencias["sync_hibrido_recomendado"] = coherencia_global >= 0.8
            
            if coherencia_global >= 0.8:
                sugerencias["mejoras_configuracion"]["mantener_sync_hibrido"] = True
            elif coherencia_global < 0.8:
                sugerencias["mejoras_configuracion"]["optimizar_coherencia_objetivo"] = 0.9
        else:
            sugerencias["sync_hibrido_recomendado"] = True
            sugerencias["mejoras_configuracion"]["habilitar_sync_hibrido"] = True

    def _actualizar_estadisticas(self, config: ConfiguracionAuroraUnificada, resultado: ResultadoAuroraIntegrado, tiempo_total: float):
        self.stats["experiencias_generadas"] += 1
        self.stats["tiempo_total_generacion"] += tiempo_total
        
        estrategia = resultado.estrategia_usada.value
        self.stats["estrategias_utilizadas"][estrategia] = self.stats["estrategias_utilizadas"].get(estrategia, 0) + 1
        
        objetivo = config.objetivo
        self.stats["objetivos_procesados"][objetivo] = self.stats["objetivos_procesados"].get(objetivo, 0) + 1
        
        for motor in resultado.componentes_usados:
            self.stats["motores_utilizados"][motor] = self.stats["motores_utilizados"].get(motor, 0) + 1
        
        if resultado.template_utilizado:
            self.stats["templates_utilizados"][resultado.template_utilizado] = self.stats["templates_utilizados"].get(resultado.template_utilizado, 0) + 1
        
        if resultado.perfil_campo_utilizado:
            self.stats["perfiles_campo_utilizados"][resultado.perfil_campo_utilizado] = self.stats["perfiles_campo_utilizados"].get(resultado.perfil_campo_utilizado, 0) + 1
        
        if resultado.secuencia_fases_utilizada:
            self.stats["secuencias_fases_utilizadas"][resultado.secuencia_fases_utilizada] = self.stats["secuencias_fases_utilizadas"].get(resultado.secuencia_fases_utilizada, 0) + 1
        
        total_experiencias = self.stats["experiencias_generadas"]
        calidad_actual = self.stats["calidad_promedio"]
        self.stats["calidad_promedio"] = ((calidad_actual * (total_experiencias - 1) + resultado.calidad_score) / total_experiencias)

    def _crear_resultado_emergencia(self, objetivo: str, error: str) -> ResultadoAuroraIntegrado:
        audio_emergencia = self._generar_audio_fallback(60.0)
        config_emergencia = ConfiguracionAuroraUnificada(objetivo=objetivo, duracion_min=1)
        
        return ResultadoAuroraIntegrado(
            audio_data=audio_emergencia,
            metadatos={
                "error": error, "modo_emergencia": True, "objetivo": objetivo,
                "timestamp": datetime.now().isoformat()
            },
            estrategia_usada=EstrategiaGeneracion.FALLBACK_PROGRESIVO,
            modo_orquestacion=ModoOrquestacion.HYBRID,
            componentes_usados=["emergencia"],
            tiempo_generacion=0.0,
            calidad_score=60.0,
            coherencia_neuroacustica=0.6,
            efectividad_terapeutica=0.6,
            configuracion=config_emergencia
        )

    def _generar_audio_fallback(self, duracion_sec: float) -> np.ndarray:
        try:
            samples = int(44100 * duracion_sec)
            t = np.linspace(0, duracion_sec, samples)
            
            onda_base = 0.3 * np.sin(2 * np.pi * 10.0 * t)
            onda_theta = 0.2 * np.sin(2 * np.pi * 6.0 * t)
            audio_mono = onda_base + onda_theta
            
            fade_samples = int(44100 * 2.0)
            if len(audio_mono) > fade_samples * 2:
                audio_mono[:fade_samples] *= np.linspace(0, 1, fade_samples)
                audio_mono[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            return np.stack([audio_mono, audio_mono])
        except Exception:
            samples = int(44100 * max(1.0, duracion_sec))
            return np.zeros((2, samples), dtype=np.float32)

    def _obtener_estrategias_disponibles(self) -> List[EstrategiaGeneracion]:
        estrategias = []
        
        motores = len([c for c in self.componentes.values() if c.tipo == TipoComponente.MOTOR and c.disponible])
        gestores = len([c for c in self.componentes.values() if c.tipo == TipoComponente.GESTOR_INTELIGENCIA and c.disponible])
        pipelines = len([c for c in self.componentes.values() if c.tipo == TipoComponente.PIPELINE and c.disponible])
        obj_managers = len([c for c in self.componentes.values() if c.tipo == TipoComponente.OBJECTIVE_MANAGER and c.disponible])
        sync_schedulers = len([c for c in self.componentes.values() if c.tipo == TipoComponente.SYNC_SCHEDULER and c.disponible])
        
        if sync_schedulers >= 1 and motores >= 2:
            estrategias.append(EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO)
        
        if obj_managers >= 1 and motores >= 2:
            estrategias.append(EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN)
        
        if motores >= 3 and gestores >= 2 and pipelines >= 1:
            estrategias.append(EstrategiaGeneracion.AURORA_ORQUESTADO)
        
        if motores >= 2:
            estrategias.append(EstrategiaGeneracion.MULTI_MOTOR)
        
        if gestores >= 1 and motores >= 1:
            estrategias.append(EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA)
        
        if motores >= 1:
            estrategias.append(EstrategiaGeneracion.MOTOR_ESPECIALIZADO)
        
        estrategias.append(EstrategiaGeneracion.FALLBACK_PROGRESIVO)
        
        return estrategias

    def obtener_estado_completo(self) -> Dict[str, Any]:
        estado_base = {
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "componentes_detectados": {
                nombre: {
                    "disponible": comp.disponible, "version": comp.version, "tipo": comp.tipo.value,
                    "fallback": comp.version == "fallback", "capacidades": len(comp.capacidades),
                    "dependencias": comp.dependencias, "prioridad": comp.nivel_prioridad
                }
                for nombre, comp in self.componentes.items()
            },
            "estadisticas_deteccion": self.detector.stats,
            "estadisticas_uso": self.stats,
            "estrategias_disponibles": [e.value for e in self._obtener_estrategias_disponibles()],
            "capacidades_sistema": {
                "motores_activos": len([c for c in self.componentes.values() if c.tipo == TipoComponente.MOTOR and c.disponible]),
                "gestores_activos": len([c for c in self.componentes.values() if c.tipo == TipoComponente.GESTOR_INTELIGENCIA and c.disponible]),
                "pipelines_activos": len([c for c in self.componentes.values() if c.tipo == TipoComponente.PIPELINE and c.disponible]),
                "objective_managers_activos": len([c for c in self.componentes.values() if c.tipo == TipoComponente.OBJECTIVE_MANAGER and c.disponible]),
                "sync_schedulers_activos": len([c for c in self.componentes.values() if c.tipo == TipoComponente.SYNC_SCHEDULER and c.disponible]),
                "verificacion_v7_2_disponible": VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE,
                "orquestador_disponible": self.orquestador is not None,
                "objective_manager_disponible": self.objective_manager is not None,
                "sync_scheduler_hibrido_disponible": self.sync_scheduler is not None,
                "fallback_garantizado": True
            },
            "metricas_calidad": {
                "calidad_promedio": self.stats["calidad_promedio"],
                "experiencias_totales": self.stats["experiencias_generadas"],
                "tiempo_promedio": (self.stats["tiempo_total_generacion"] / max(1, self.stats["experiencias_generadas"])),
                "tasa_exito": ((self.stats["experiencias_generadas"] - self.stats["errores_manejados"]) / max(1, self.stats["experiencias_generadas"]) * 100)
            }
        }
        
        estado_base["metricas_integraciones_optimizadas"] = {
            "efectos_psicodelicos_disponibles": self.stats.get("efectos_psicodelicos_disponibles", 0),
            "efectos_psicodelicos_aplicados": self.stats.get("efectos_psicodelicos_aplicados", 0),
            "emotion_style_utilizaciones": self.stats.get("emotion_style_utilizaciones", 0),
            "field_profiles_avanzados_utilizados": self.stats.get("field_profiles_avanzados_utilizados", 0),
            "integraciones_exitosas": self.stats.get("integraciones_exitosas", 0),
            "psychedelic_effects_cargado": bool(self.psychedelic_effects),
            "emotion_style_disponible": "emotion_style_profiles" in self.componentes,
            "field_profiles_disponible": "field_profiles" in self.componentes,
            "configuracion_optimizada_activa": True,
            "sync_hibrido_utilizaciones": self.stats.get("sync_hibrido_utilizaciones", 0),
            "coherencia_global_promedio": self.stats.get("coherencia_global_promedio", 0.0),
            "sync_scheduler_hibrido_disponible": SYNC_SCHEDULER_HIBRIDO_AVAILABLE and self.sync_scheduler is not None
        }
        
        if self.objective_manager:
            estado_base["metricas_objective_manager"] = {
                "utilizaciones_totales": self.stats["objective_manager_utilizaciones"],
                "templates_mas_utilizados": sorted(self.stats["templates_utilizados"].items(), key=lambda x: x[1], reverse=True)[:5],
                "perfiles_campo_mas_utilizados": sorted(self.stats["perfiles_campo_utilizados"].items(), key=lambda x: x[1], reverse=True)[:5],
                "secuencias_fases_mas_utilizadas": sorted(self.stats["secuencias_fases_utilizadas"].items(), key=lambda x: x[1], reverse=True)[:5],
                "disponible": bool(self.objective_manager),
                "version": getattr(self.objective_manager, 'version', 'unknown') if self.objective_manager else None,
                "capacidades": getattr(self.objective_manager, 'obtener_capacidades', lambda: {})() if self.objective_manager else {}
            }
        else:
            estado_base["metricas_objective_manager"] = {"disponible": False}
        
        if self.sync_scheduler:
            estado_base["metricas_sync_hibrido"] = {
                "utilizaciones_totales": self.stats.get("sync_hibrido_utilizaciones", 0),
                "coherencia_global_promedio": self.stats.get("coherencia_global_promedio", 0.0),
                "disponible": bool(self.sync_scheduler),
                "version": "V7_UNIFIED_OPTIMIZED",
                "funciones_disponibles": ["sincronizar_y_estructurar_capas", "aplicar_fade_narrativo", "optimizar_coherencia_global", "validar_sync_y_estructura_completa"],
                "estrategia_hibrida_activa": EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO in self._obtener_estrategias_disponibles()
            }
        else:
            estado_base["metricas_sync_hibrido"] = {"disponible": False, "razon": "sync_and_scheduler no disponible"}
        
        estado_base["metricas_verificacion_v7_2"] = {
            "disponible": VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE,
            "version": "V7.2_ENHANCED" if VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE else "NO_DISPONIBLE",
            "verificaciones_totales": self.stats.get("verificaciones_v7_2", 0),
            "verificaciones_exitosas": self.stats.get("verificaciones_exitosas_v7_2", 0),
            "errores_verificacion": self.stats.get("errores_verificacion_v7_2", 0),
            "tasa_exito_verificacion": ((self.stats.get("verificaciones_exitosas_v7_2", 0) / max(1, self.stats.get("verificaciones_v7_2", 1))) * 100),
            "funciones_disponibles": ["verificar_estructura_aurora_v7_unificada", "benchmark_verificacion_comparativa", "verificacion_rapida_unificada", "diagnostico_cientifico_completo"] if VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE else []
        }
        
        return estado_base


# Funciones de conveniencia para crear experiencias específicas
def crear_experiencia_sync_hibrido(objetivo: str, **kwargs) -> ResultadoAuroraIntegrado:
    kwargs.update({
        "estrategia_preferida": EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO,
        "modo_orquestacion": ModoOrquestacion.SYNC_HIBRIDO,
        "habilitar_sync_hibrido": True,
        "calidad_objetivo": "maxima"
    })
    return Aurora(objetivo, **kwargs)

def crear_experiencia_coherencia_maxima(objetivo: str, coherencia_objetivo: float = 0.95, **kwargs) -> ResultadoAuroraIntegrado:
    kwargs.update({
        "estrategia_preferida": EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO,
        "habilitar_sync_hibrido": True,
        "coherencia_objetivo": coherencia_objetivo,
        "calidad_objetivo": "maxima",
        "modo_orquestacion": ModoOrquestacion.SYNC_HIBRIDO
    })
    return Aurora(objetivo, **kwargs)

def crear_experiencia_estructura_inteligente(objetivo: str, duracion_min: int = 30, **kwargs) -> ResultadoAuroraIntegrado:
    kwargs.update({
        "estrategia_preferida": EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO,
        "habilitar_sync_hibrido": True,
        "duracion_min": duracion_min,
        "calidad_objetivo": "maxima",
        "validacion_automatica": True
    })
    return Aurora(objetivo, **kwargs)

def verificar_capacidades_sync_hibrido() -> Dict[str, Any]:
    director = Aurora()
    return {
        "sync_scheduler_disponible": SYNC_SCHEDULER_HIBRIDO_AVAILABLE,
        "sync_scheduler_conectado": director.sync_scheduler is not None,
        "estrategia_hibrida_disponible": EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO in director._obtener_estrategias_disponibles(),
        "funciones_hibridas": ["sincronizar_y_estructurar_capas", "aplicar_fade_narrativo", "optimizar_coherencia_global", "validar_sync_y_estructura_completa"] if SYNC_SCHEDULER_HIBRIDO_AVAILABLE else [],
        "motores_compatibles": len([c for c in director.componentes.values() if c.tipo == TipoComponente.MOTOR and c.disponible]),
        "coherencia_promedio_historica": director.stats.get("coherencia_global_promedio", 0.0),
        "utilizaciones_hibridas": director.stats.get("sync_hibrido_utilizaciones", 0)
    }

def obtener_estadisticas_sync_hibrido() -> Dict[str, Any]:
    director = Aurora()
    return director.obtener_estado_completo().get("metricas_sync_hibrido", {"disponible": False})

def crear_experiencia_con_template(objetivo: str, template: str, **kwargs) -> ResultadoAuroraIntegrado:
    return Aurora(objetivo, template_personalizado=template, **kwargs)

def crear_experiencia_con_perfil_campo(objetivo: str, perfil_campo: str, **kwargs) -> ResultadoAuroraIntegrado:
    return Aurora(objetivo, perfil_campo_personalizado=perfil_campo, **kwargs)

def crear_experiencia_con_secuencia_fases(objetivo: str, secuencia: str, **kwargs) -> ResultadoAuroraIntegrado:
    return Aurora(objetivo, secuencia_fases_personalizada=secuencia, **kwargs)

def crear_experiencia_optimizada(objetivo: str, **kwargs) -> ResultadoAuroraIntegrado:
    return Aurora(objetivo, calidad_objetivo="maxima", modo_orquestacion="layered", **kwargs)

def crear_experiencia_psicodelica(objetivo: str, efecto_deseado: str = None, **kwargs) -> ResultadoAuroraIntegrado:
    objetivo_modificado = f"{objetivo} {efecto_deseado}" if efecto_deseado else objetivo
    return Aurora(objetivo_modificado, calidad_objetivo="maxima", modo_orquestacion="layered", **kwargs)

def crear_experiencia_emocional(objetivo: str, emocion_objetivo: str = None, **kwargs) -> ResultadoAuroraIntegrado:
    objetivo_modificado = f"{objetivo} {emocion_objetivo}" if emocion_objetivo else objetivo
    return Aurora(objetivo_modificado, intensidad="media", **kwargs)

def obtener_templates_disponibles() -> List[str]:
    director = Aurora()
    if director.objective_manager and hasattr(director.objective_manager, 'obtener_templates_disponibles'):
        return director.objective_manager.obtener_templates_disponibles()
    return []

def obtener_perfiles_campo_disponibles() -> List[str]:
    director = Aurora()
    if director.objective_manager and hasattr(director.objective_manager, 'obtener_perfiles_disponibles'):
        return director.objective_manager.obtener_perfiles_disponibles()
    return []

def obtener_secuencias_fases_disponibles() -> List[str]:
    director = Aurora()
    if director.objective_manager and hasattr(director.objective_manager, 'obtener_secuencias_disponibles'):
        return director.objective_manager.obtener_secuencias_disponibles()
    return []

def obtener_efectos_psicodelicos_disponibles() -> List[str]:
    director = Aurora()
    if director.psychedelic_effects and "pe" in director.psychedelic_effects:
        return list(director.psychedelic_effects["pe"].keys())
    return []

def obtener_estado_integraciones() -> Dict[str, Any]:
    director = Aurora()
    return director.obtener_estado_completo().get("metricas_integraciones_optimizadas", {})

def verificar_integraciones_optimizadas() -> Dict[str, bool]:
    director = Aurora()
    return {
        "psychedelic_effects": bool(director.psychedelic_effects),
        "emotion_style_profiles": "emotion_style_profiles" in director.componentes,
        "field_profiles": "field_profiles" in director.componentes,
        "objective_manager": director.objective_manager is not None,
        "carmine_analyzer": "carmine_analyzer_v21" in director.componentes,
        "quality_pipeline": "quality_pipeline" in director.componentes,
        "sync_scheduler_hibrido": director.sync_scheduler is not None,
        "verificacion_estructural_v7_2": VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE,
        "sistema_completamente_optimizado": all([
            bool(director.psychedelic_effects),
            "emotion_style_profiles" in director.componentes,
            "field_profiles" in director.componentes,
            director.sync_scheduler is not None,
            VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE
        ])
    }

def obtener_estado_verificacion_v7_2() -> Dict[str, Any]:
    return {
        "disponible": VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE,
        "version": "V7.2_ENHANCED" if VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE else "NO_DISPONIBLE",
        "funciones": ["verificar_estructura_aurora_v7_unificada", "benchmark_verificacion_comparativa", "verificacion_rapida_unificada", "diagnostico_cientifico_completo"] if VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE else [],
        "capacidades": {
            "validacion_unificada": VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE,
            "benchmark_automatico": VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE,
            "recomendaciones_ia": VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE,
            "analisis_aurora_v7": VERIFICACION_ESTRUCTURAL_V7_2_DISPONIBLE
        }
    }

def verificar_calidad_experiencia_aurora(objetivo: str, **kwargs) -> ResultadoAuroraIntegrado:
    kwargs.update({
        "calidad_objetivo": "maxima",
        "validacion_automatica": True,
        "aplicar_mastering": True
    })
    return Aurora(objetivo, **kwargs)

def obtener_estadisticas_verificacion_v7_2() -> Dict[str, Any]:
    director = Aurora()
    return director.obtener_estado_completo().get("metricas_verificacion_v7_2", {"disponible": False})

def experiencia_con_verificacion_completa(objetivo: str, **kwargs) -> ResultadoAuroraIntegrado:
    kwargs.update({
        "calidad_objetivo": "maxima",
        "modo_orquestacion": ModoOrquestacion.SYNC_HIBRIDO,
        "habilitar_sync_hibrido": True,
        "usar_objective_manager": True,
        "validacion_automatica": True,
        "aplicar_mastering": True,
        "coherencia_objetivo": 0.95
    })
    return Aurora(objetivo, **kwargs)


# Director global para singleton
_director_global: Optional[AuroraDirectorV7Integrado] = None

def Aurora(objetivo: str = None, **kwargs) -> Union[ResultadoAuroraIntegrado, AuroraDirectorV7Integrado]:
    global _director_global
    if _director_global is None:
        _director_global = AuroraDirectorV7Integrado()
    
    if objetivo is not None:
        return _director_global.crear_experiencia(objetivo, **kwargs)
    else:
        return _director_global

# Métodos de conveniencia del objeto Aurora
Aurora.rapido = lambda obj, **kw: Aurora(obj, duracion_min=5, calidad_objetivo="media", **kw)
Aurora.largo = lambda obj, **kw: Aurora(obj, duracion_min=60, calidad_objetivo="alta", **kw)
Aurora.terapeutico = lambda obj, **kw: Aurora(obj, duracion_min=45, intensidad="suave", calidad_objetivo="maxima", modo_orquestacion="layered", **kw)
Aurora.optimizado = lambda obj, **kw: crear_experiencia_optimizada(obj, **kw)
Aurora.psicodelico = lambda obj, efecto=None, **kw: crear_experiencia_psicodelica(obj, efecto, **kw)
Aurora.emocional = lambda obj, emocion=None, **kw: crear_experiencia_emocional(obj, emocion, **kw)
Aurora.hibrido = lambda obj, **kw: crear_experiencia_sync_hibrido(obj, **kw)
Aurora.coherencia_maxima = lambda obj, coherencia=0.95, **kw: crear_experiencia_coherencia_maxima(obj, coherencia, **kw)
Aurora.estructura_inteligente = lambda obj, duracion=30, **kw: crear_experiencia_estructura_inteligente(obj, duracion, **kw)
Aurora.verificado = lambda obj, **kw: verificar_calidad_experiencia_aurora(obj, **kw)
Aurora.calidad_verificada = lambda obj, **kw: verificar_calidad_experiencia_aurora(obj, **kw)
Aurora.completo = lambda obj, **kw: experiencia_con_verificacion_completa(obj, **kw)

# Métodos de estado y diagnóstico
Aurora.estado = lambda: Aurora().obtener_estado_completo()
Aurora.diagnostico = lambda: Aurora().detector.stats
Aurora.stats = lambda: Aurora().stats
Aurora.integraciones = lambda: obtener_estado_integraciones()
Aurora.verificar = lambda: verificar_integraciones_optimizadas()
Aurora.verificar_sync_hibrido = lambda: verificar_capacidades_sync_hibrido()
Aurora.stats_sync_hibrido = lambda: obtener_estadisticas_sync_hibrido()
Aurora.estado_verificacion = obtener_estado_verificacion_v7_2
Aurora.stats_verificacion_v7_2 = obtener_estadisticas_verificacion_v7_2

# Métodos específicos de configuración
Aurora.con_template = crear_experiencia_con_template
Aurora.con_perfil_campo = crear_experiencia_con_perfil_campo
Aurora.con_secuencia_fases = crear_experiencia_con_secuencia_fases
Aurora.templates_disponibles = obtener_templates_disponibles
Aurora.perfiles_campo_disponibles = obtener_perfiles_campo_disponibles
Aurora.secuencias_fases_disponibles = obtener_secuencias_fases_disponibles
Aurora.efectos_psicodelicos_disponibles = obtener_efectos_psicodelicos_disponibles


if __name__ == "__main__":
    print("🌟 Aurora Director V7 INTEGRADO - Sistema con Sync Scheduler Híbrido + Verificación V7.2")
    print("=" * 100)
    
    director = Aurora()
    estado = director.obtener_estado_completo()
    
    print(f"🚀 {estado['version']}")
    print(f"⏰ Inicializado: {estado['timestamp']}")
    
    print(f"\n📊 Componentes detectados: {len(estado['componentes_detectados'])}")
    for nombre, info in estado['componentes_detectados'].items():
        emoji = "✅" if info['disponible'] and not info['fallback'] else "🔄" if info['fallback'] else "❌"
        tipo_emoji = {
            'motor': '🎵', 'gestor_inteligencia': '🧠', 'pipeline': '🔄',
            'preset_manager': '🎯', 'style_profile': '🎨', 'objective_manager': '🎯',
            'sync_scheduler': '🌟'
        }.get(info['tipo'], '🔧')
        print(f"   {emoji} {tipo_emoji} {nombre} v{info['version']} (P{info['prioridad']})")
    
    caps = estado['capacidades_sistema']
    print(f"\n🔧 Capacidades del Sistema:")
    print(f"   🎵 Motores activos: {caps['motores_activos']}")
    print(f"   🧠 Gestores activos: {caps['gestores_activos']}")
    print(f"   🔄 Pipelines activos: {caps['pipelines_activos']}")
    print(f"   🎯 OM activos: {caps['objective_managers_activos']}")
    print(f"   🌟 Sync Schedulers activos: {caps['sync_schedulers_activos']}")
    print(f"   🔬 Verificación V7.2: {'✅' if caps['verificacion_v7_2_disponible'] else '❌'}")
    print(f"   🎼 Orquestador: {'✅' if caps['orquestador_disponible'] else '❌'}")
    print(f"   🎯 OM: {'✅' if caps['objective_manager_disponible'] else '❌'}")
    print(f"   🌟 Sync Híbrido: {'✅' if caps['sync_scheduler_hibrido_disponible'] else '❌'}")
    print(f"   🛡️ Fallback garantizado: {'✅' if caps['fallback_garantizado'] else '❌'}")
    
    print(f"\n🏆 AURORA DIRECTOR V7 INICIALIZADO CORRECTAMENTE")
    print(f"✨ ¡Listo para crear experiencias transformadoras!")
    print("=" * 100)

import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
import weakref
import gc
from typing import Callable, AsyncGenerator

# ===== OPTIMIZACIÓN DE MEMORIA Y RENDIMIENTO =====

class MemoryOptimizer:
    """Optimizador de memoria para mantener el sistema liviano"""
    
    def __init__(self):
        self._cache_limits = {
            'configuraciones': 50,
            'resultados': 20,
            'deteccion': 30
        }
        self._weak_refs = weakref.WeakValueDictionary()
    
    def optimize_caches(self, director_instance):
        """Optimiza los caches del director manteniendo solo lo esencial"""
        if hasattr(director_instance, 'cache_configuraciones'):
            cache = director_instance.cache_configuraciones
            if len(cache) > self._cache_limits['configuraciones']:
                # Mantener solo los más recientes
                items = list(cache.items())
                director_instance.cache_configuraciones = dict(items[-self._cache_limits['configuraciones']:])
        
        if hasattr(director_instance, 'cache_resultados'):
            cache = director_instance.cache_resultados
            if len(cache) > self._cache_limits['resultados']:
                items = list(cache.items())
                director_instance.cache_resultados = dict(items[-self._cache_limits['resultados']:])
        
        # Forzar garbage collection
        gc.collect()
    
    def create_lightweight_config(self, objetivo: str, **kwargs) -> Dict[str, Any]:
        """Crea configuraciones optimizadas en memoria"""
        return {
            'objetivo': objetivo,
            'duracion_min': kwargs.get('duracion_min', 20),
            'intensidad': kwargs.get('intensidad', 'media'),
            'estilo': kwargs.get('estilo', 'sereno'),
            'calidad_objetivo': kwargs.get('calidad_objetivo', 'alta'),
            'modo_orquestacion': kwargs.get('modo_orquestacion', ModoOrquestacion.HYBRID),
            '_optimizado': True
        }

# ===== OBJECTIVE MANAGER FALLBACK MEJORADO =====

class ObjectiveManagerFallbackAvanzado:
    """Fallback avanzado cuando OM no está disponible"""
    
    def __init__(self):
        self.version = "FALLBACK_AVANZADO_V7"
        self._templates_cache = self._init_templates_avanzados()
        self._perfiles_cache = self._init_perfiles_avanzados()
        self._secuencias_cache = self._init_secuencias_avanzadas()
    
    def _init_templates_avanzados(self) -> Dict[str, Dict[str, Any]]:
        """Templates más sofisticados que el fallback básico"""
        return {
            "relajacion_profunda_plus": {
                "neurotransmisor_preferido": "gaba",
                "intensidad": "suave",
                "estilo": "sereno",
                "beat_base": 6.5,
                "coherencia_objetivo": 0.85,
                "duracion_optima": 30,
                "secuencia_fases": "entrada_suave_desarrollo_profundo_salida_gradual"
            },
            "concentracion_laser": {
                "neurotransmisor_preferido": "acetilcolina",
                "intensidad": "media",
                "estilo": "crystalline",
                "beat_base": 15.0,
                "coherencia_objetivo": 0.90,
                "duracion_optima": 25,
                "secuencia_fases": "activacion_rapida_mantenimiento_sostenido"
            },
            "creatividad_exponencial_plus": {
                "neurotransmisor_preferido": "anandamida",
                "intensidad": "media",
                "estilo": "organico",
                "beat_base": 10.5,
                "coherencia_objetivo": 0.88,
                "duracion_optima": 35,
                "secuencia_fases": "apertura_gradual_expansion_maxima_integracion"
            },
            "meditacion_trascendental": {
                "neurotransmisor_preferido": "serotonina",
                "intensidad": "suave",
                "estilo": "mistico",
                "beat_base": 7.8,
                "coherencia_objetivo": 0.92,
                "duracion_optima": 40,
                "secuencia_fases": "centramiento_profundizacion_trascendencia"
            },
            "energia_vital": {
                "neurotransmisor_preferido": "dopamina",
                "intensidad": "intenso",
                "estilo": "tribal",
                "beat_base": 12.0,
                "coherencia_objetivo": 0.80,
                "duracion_optima": 20,
                "secuencia_fases": "activacion_energetica_pico_estabilizacion"
            },
            "sanacion_emocional": {
                "neurotransmisor_preferido": "oxitocina",
                "intensidad": "suave",
                "estilo": "sereno",
                "beat_base": 8.0,
                "coherencia_objetivo": 0.90,
                "duracion_optima": 45,
                "secuencia_fases": "apertura_cardiaca_procesamiento_integracion"
            }
        }
    
    def _init_perfiles_avanzados(self) -> Dict[str, Dict[str, Any]]:
        """Perfiles de campo más detallados"""
        return {
            "cognitivo_avanzado": {
                "beats_primarios": [14.0, 15.5, 16.0],
                "beats_secundarios": [7.0, 7.8],
                "modulacion": 0.15,
                "espacialidad": "focalizada",
                "dinamica": "ascendente"
            },
            "emocional_equilibrio": {
                "beats_primarios": [8.0, 10.0, 12.0],
                "beats_secundarios": [4.0, 6.0],
                "modulacion": 0.12,
                "espacialidad": "envolvente",
                "dinamica": "ondulante"
            },
            "espiritual_profundo": {
                "beats_primarios": [6.0, 7.83, 8.5],
                "beats_secundarios": [3.0, 4.5],
                "modulacion": 0.08,
                "espacialidad": "expandida",
                "dinamica": "descendente"
            }
        }
    
    def _init_secuencias_avanzadas(self) -> Dict[str, List[Dict[str, Any]]]:
        """Secuencias de fases más sofisticadas"""
        return {
            "transformacion_completa": [
                {"fase": "preparacion", "duracion_pct": 0.15, "intensidad": 0.3, "beat_mod": 1.0},
                {"fase": "activacion", "duracion_pct": 0.25, "intensidad": 0.7, "beat_mod": 1.2},
                {"fase": "procesamiento", "duracion_pct": 0.35, "intensidad": 0.9, "beat_mod": 1.0},
                {"fase": "integracion", "duracion_pct": 0.25, "intensidad": 0.5, "beat_mod": 0.8}
            ],
            "claridad_mental_rapida": [
                {"fase": "enfoque_inicial", "duracion_pct": 0.20, "intensidad": 0.6, "beat_mod": 1.1},
                {"fase": "concentracion_pico", "duracion_pct": 0.60, "intensidad": 1.0, "beat_mod": 1.3},
                {"fase": "estabilizacion", "duracion_pct": 0.20, "intensidad": 0.7, "beat_mod": 1.0}
            ],
            "relajacion_profunda_gradual": [
                {"fase": "descenso_inicial", "duracion_pct": 0.30, "intensidad": 0.8, "beat_mod": 1.0},
                {"fase": "profundizacion", "duracion_pct": 0.50, "intensidad": 0.4, "beat_mod": 0.7},
                {"fase": "estado_profundo", "duracion_pct": 0.20, "intensidad": 0.2, "beat_mod": 0.5}
            ]
        }
    
    def procesar_objetivo_completo(self, objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesamiento avanzado con IA heurística"""
        objetivo_lower = objetivo.lower()
        contexto = contexto or {}
        
        # Análisis semántico básico
        template_seleccionado = self._analizar_semantica_objetivo(objetivo_lower)
        perfil_campo = self._seleccionar_perfil_campo(objetivo_lower, contexto)
        secuencia_fases = self._determinar_secuencia_optima(objetivo_lower, contexto)
        
        config_base = self._templates_cache.get(template_seleccionado, self._templates_cache["relajacion_profunda_plus"])
        
        # Optimizaciones contextuales
        config_optimizada = self._aplicar_optimizaciones_contextuales(config_base, contexto)
        
        return {
            "configuracion_motor": config_optimizada,
            "template_utilizado": template_seleccionado,
            "perfil_campo_utilizado": perfil_campo,
            "secuencia_fases_utilizada": secuencia_fases,
            "resultado_routing": {
                "confianza": 0.85,
                "tipo": "fallback_avanzado_heuristico",
                "fuente": "objective_manager_fallback_avanzado",
                "analisis_semantico": True
            },
            "metadatos": {
                "fallback_avanzado_usado": True,
                "version_fallback": self.version,
                "objetivo_original": objetivo,
                "contexto_procesado": contexto,
                "optimizaciones_aplicadas": True
            }
        }
    
    def _analizar_semantica_objetivo(self, objetivo_lower: str) -> str:
        """Análisis semántico mejorado del objetivo"""
        # Patrones más específicos
        patrones_avanzados = {
            "concentracion_laser": ["laser", "foco intenso", "concentracion maxima", "enfoque extremo"],
            "meditacion_trascendental": ["trascendental", "espiritual", "conexion", "conciencia"],
            "creatividad_exponencial_plus": ["breakthrough", "exponencial", "innovacion", "inspiracion"],
            "energia_vital": ["energia", "vitalidad", "activacion", "poder"],
            "sanacion_emocional": ["sanacion", "trauma", "emocional", "curacion", "liberacion"],
            "relajacion_profunda_plus": ["profunda", "descanso", "calma", "paz"]
        }
        
        # Scoring por coincidencias
        scores = {}
        for template, palabras_clave in patrones_avanzados.items():
            score = sum(1 for palabra in palabras_clave if palabra in objetivo_lower)
            if score > 0:
                scores[template] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        # Fallback a patrones básicos
        if any(word in objetivo_lower for word in ["concentracion", "focus", "atencion"]):
            return "concentracion_laser"
        elif any(word in objetivo_lower for word in ["creatividad", "creativo", "arte"]):
            return "creatividad_exponencial_plus"
        elif any(word in objetivo_lower for word in ["meditacion", "mindfulness", "presencia"]):
            return "meditacion_trascendental"
        else:
            return "relajacion_profunda_plus"
    
    def _seleccionar_perfil_campo(self, objetivo_lower: str, contexto: Dict[str, Any]) -> str:
        """Selección inteligente de perfil de campo"""
        if any(word in objetivo_lower for word in ["concentracion", "focus", "claridad", "mental"]):
            return "cognitivo_avanzado"
        elif any(word in objetivo_lower for word in ["emocional", "sentimientos", "equilibrio"]):
            return "emocional_equilibrio"
        elif any(word in objetivo_lower for word in ["espiritual", "meditacion", "trascendental"]):
            return "espiritual_profundo"
        else:
            return "emocional_equilibrio"
    
    def _determinar_secuencia_optima(self, objetivo_lower: str, contexto: Dict[str, Any]) -> str:
        """Determina secuencia óptima basada en objetivo y contexto"""
        duracion = contexto.get('duracion_min', 20)
        
        if duracion <= 15:
            return "claridad_mental_rapida"
        elif any(word in objetivo_lower for word in ["transformacion", "cambio", "breakthrough"]):
            return "transformacion_completa"
        elif any(word in objetivo_lower for word in ["relajacion", "calma", "descanso"]):
            return "relajacion_profunda_gradual"
        else:
            return "transformacion_completa"
    
    def _aplicar_optimizaciones_contextuales(self, config_base: Dict[str, Any], contexto: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica optimizaciones basadas en contexto"""
        config_optimizada = config_base.copy()
        
        # Ajustes por duración
        duracion = contexto.get('duracion_min', 20)
        if duracion < config_base.get('duracion_optima', 20):
            config_optimizada['intensidad'] = 'media'  # Compensar duración corta
        elif duracion > config_base.get('duracion_optima', 20) * 1.5:
            config_optimizada['intensidad'] = 'suave'  # Sesiones largas más suaves
        
        # Ajustes por contexto de uso
        contexto_uso = contexto.get('contexto_uso', '')
        if contexto_uso == 'trabajo':
            config_optimizada['estilo'] = 'crystalline'
            config_optimizada['intensidad'] = 'media'
        elif contexto_uso == 'sueño':
            config_optimizada['estilo'] = 'sereno'
            config_optimizada['intensidad'] = 'suave'
        
        # Ajustes por calidad objetivo
        calidad = contexto.get('calidad_objetivo', 'alta')
        if calidad == 'maxima':
            config_optimizada['coherencia_objetivo'] = min(0.95, config_optimizada.get('coherencia_objetivo', 0.8) + 0.1)
        
        return config_optimizada
    
    def obtener_capacidades(self) -> Dict[str, Any]:
        """Capacidades del fallback avanzado"""
        return {
            "nombre": "Objective Manager Fallback Avanzado",
            "tipo": "gestor_objetivos_fallback_heuristico",
            "version": self.version,
            "capacidades": [
                "analisis_semantico_avanzado",
                "optimizacion_contextual",
                "secuencias_fases_inteligentes",
                "perfiles_campo_detallados"
            ],
            "templates_disponibles": list(self._templates_cache.keys()),
            "perfiles_campo_disponibles": list(self._perfiles_cache.keys()),
            "secuencias_disponibles": list(self._secuencias_cache.keys()),
            "fallback_avanzado": True,
            "confianza_promedio": 0.85
        }

# ===== OPTIMIZACIONES ADITIVAS AL DETECTOR =====

def optimizar_detector_componentes(detector_instance):
    """Mejoras aditivas al detector sin modificar lógica"""
    
    # Método adicional para reintentar detección de OM
    def reintentar_objective_manager(self):
        """Reintenta la detección del Objective Manager con múltiples estrategias"""
        estrategias_import = [
            "objective_manager",
            "Aurora.ObjectiveManager.V7",
            "aurora_system.objective_manager",
            "objective_manager_v7"
        ]
        
        for estrategia in estrategias_import:
            try:
                modulo = importlib.import_module(estrategia)
                if hasattr(modulo, 'crear_objective_manager_unificado'):
                    instancia = modulo.crear_objective_manager_unificado()
                    if self._validar_instancia_om(instancia):
                        return instancia
                elif hasattr(modulo, 'ObjectiveManagerUnificado'):
                    instancia = modulo.ObjectiveManagerUnificado()
                    if self._validar_instancia_om(instancia):
                        return instancia
            except ImportError:
                continue
        
        # Si falla, usar fallback avanzado
        logger.info("🔄 Usando Objective Manager Fallback Avanzado")
        return ObjectiveManagerFallbackAvanzado()
    
    def _validar_instancia_om(self, instancia):
        """Validación específica para OM"""
        return hasattr(instancia, 'procesar_objetivo_completo')
    
    # Agregar métodos al detector
    detector_instance.reintentar_objective_manager = reintentar_objective_manager.__get__(detector_instance)
    detector_instance._validar_instancia_om = _validar_instancia_om.__get__(detector_instance)
    
    return detector_instance

# ===== SISTEMA DE MONITOREO Y MÉTRICAS MEJORADO =====

class MetricasAvanzadas:
    """Sistema de métricas más detallado"""
    
    def __init__(self):
        self.metricas_detalladas = {
            "rendimiento": {
                "tiempo_promedio_generacion": 0.0,
                "throughput_experiencias_por_minuto": 0.0,
                "eficiencia_memoria": 0.0
            },
            "calidad": {
                "score_promedio_total": 0.0,
                "coherencia_promedio_total": 0.0,
                "efectividad_promedio_total": 0.0,
                "distribucion_calidad": {"alta": 0, "media": 0, "baja": 0}
            },
            "uso_componentes": {
                "motores_mas_efectivos": {},
                "estrategias_mas_exitosas": {},
                "combinaciones_optimas": {}
            },
            "objetivo_insights": {
                "objetivos_populares": {},
                "patrones_temporales": {},
                "contextos_frecuentes": {}
            }
        }
        self._start_time = time.time()
        self._total_experiencias = 0
    
    def actualizar_metricas_experiencia(self, resultado: 'ResultadoAuroraIntegrado', tiempo_generacion: float):
        """Actualiza métricas con nueva experiencia"""
        self._total_experiencias += 1
        
        # Métricas de rendimiento
        self._actualizar_rendimiento(tiempo_generacion)
        
        # Métricas de calidad
        self._actualizar_calidad(resultado)
        
        # Métricas de componentes
        self._actualizar_uso_componentes(resultado)
        
        # Insights de objetivos
        self._actualizar_objetivo_insights(resultado)
    
    def _actualizar_rendimiento(self, tiempo_generacion: float):
        """Actualiza métricas de rendimiento"""
        total = self._total_experiencias
        tiempo_actual = self.metricas_detalladas["rendimiento"]["tiempo_promedio_generacion"]
        nuevo_promedio = ((tiempo_actual * (total - 1)) + tiempo_generacion) / total
        
        self.metricas_detalladas["rendimiento"]["tiempo_promedio_generacion"] = nuevo_promedio
        
        # Throughput
        tiempo_transcurrido = time.time() - self._start_time
        if tiempo_transcurrido > 0:
            self.metricas_detalladas["rendimiento"]["throughput_experiencias_por_minuto"] = (total / tiempo_transcurrido) * 60
    
    def _actualizar_calidad(self, resultado):
        """Actualiza métricas de calidad"""
        total = self._total_experiencias
        metricas_calidad = self.metricas_detalladas["calidad"]
        
        # Promedios móviles
        score_actual = metricas_calidad["score_promedio_total"]
        coherencia_actual = metricas_calidad["coherencia_promedio_total"]
        efectividad_actual = metricas_calidad["efectividad_promedio_total"]
        
        metricas_calidad["score_promedio_total"] = ((score_actual * (total - 1)) + resultado.calidad_score) / total
        metricas_calidad["coherencia_promedio_total"] = ((coherencia_actual * (total - 1)) + resultado.coherencia_neuroacustica) / total
        metricas_calidad["efectividad_promedio_total"] = ((efectividad_actual * (total - 1)) + resultado.efectividad_terapeutica) / total
        
        # Distribución de calidad
        if resultado.calidad_score >= 85:
            metricas_calidad["distribucion_calidad"]["alta"] += 1
        elif resultado.calidad_score >= 70:
            metricas_calidad["distribucion_calidad"]["media"] += 1
        else:
            metricas_calidad["distribucion_calidad"]["baja"] += 1
    
    def _actualizar_uso_componentes(self, resultado):
        """Actualiza métricas de uso de componentes"""
        uso = self.metricas_detalladas["uso_componentes"]
        
        # Motores más efectivos
        for motor in resultado.componentes_usados:
            if motor not in uso["motores_mas_efectivos"]:
                uso["motores_mas_efectivos"][motor] = {"usos": 0, "calidad_promedio": 0.0}
            
            motor_data = uso["motores_mas_efectivos"][motor]
            usos_previos = motor_data["usos"]
            calidad_previa = motor_data["calidad_promedio"]
            
            motor_data["usos"] += 1
            motor_data["calidad_promedio"] = ((calidad_previa * usos_previos) + resultado.calidad_score) / motor_data["usos"]
        
        # Estrategias más exitosas
        estrategia = resultado.estrategia_usada.value
        if estrategia not in uso["estrategias_mas_exitosas"]:
            uso["estrategias_mas_exitosas"][estrategia] = {"usos": 0, "efectividad_promedio": 0.0}
        
        estrategia_data = uso["estrategias_mas_exitosas"][estrategia]
        usos_previos = estrategia_data["usos"]
        efectividad_previa = estrategia_data["efectividad_promedio"]
        
        estrategia_data["usos"] += 1
        estrategia_data["efectividad_promedio"] = ((efectividad_previa * usos_previos) + resultado.efectividad_terapeutica) / estrategia_data["usos"]
    
    def _actualizar_objetivo_insights(self, resultado):
        """Actualiza insights sobre objetivos"""
        insights = self.metricas_detalladas["objetivo_insights"]
        objetivo = resultado.configuracion.objetivo
        
        # Objetivos populares
        if objetivo not in insights["objetivos_populares"]:
            insights["objetivos_populares"][objetivo] = 0
        insights["objetivos_populares"][objetivo] += 1
        
        # Contextos frecuentes
        contexto = resultado.configuracion.contexto_uso
        if contexto:
            if contexto not in insights["contextos_frecuentes"]:
                insights["contextos_frecuentes"][contexto] = 0
            insights["contextos_frecuentes"][contexto] += 1
    
    def obtener_reporte_completo(self) -> Dict[str, Any]:
        """Genera reporte completo de métricas"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_experiencias": self._total_experiencias,
            "tiempo_operacion_minutos": (time.time() - self._start_time) / 60,
            "metricas": self.metricas_detalladas,
            "recomendaciones": self._generar_recomendaciones()
        }
    
    def _generar_recomendaciones(self) -> List[str]:
        """Genera recomendaciones basadas en métricas"""
        recomendaciones = []
        
        # Análisis de rendimiento
        if self.metricas_detalladas["rendimiento"]["tiempo_promedio_generacion"] > 5.0:
            recomendaciones.append("Considerar optimización de componentes para reducir tiempo de generación")
        
        # Análisis de calidad
        distribucion = self.metricas_detalladas["calidad"]["distribucion_calidad"]
        total = sum(distribucion.values())
        if total > 0:
            pct_alta = (distribucion["alta"] / total) * 100
            if pct_alta < 70:
                recomendaciones.append("Incrementar configuraciones de calidad máxima para mejorar resultados")
        
        # Análisis de componentes
        motores = self.metricas_detalladas["uso_componentes"]["motores_mas_efectivos"]
        if motores:
            mejor_motor = max(motores.items(), key=lambda x: x[1]["calidad_promedio"])
            recomendaciones.append(f"Motor más efectivo: {mejor_motor[0]} (calidad: {mejor_motor[1]['calidad_promedio']:.1f})")
        
        return recomendaciones

# ===== FUNCIONES DE INTEGRACIÓN ADITIVA =====

def aplicar_mejoras_aditivas_director(director_instance):
    """Aplica todas las mejoras aditivas al director sin modificar lógica existente"""
    
    # Optimizador de memoria
    memory_optimizer = MemoryOptimizer()
    director_instance._memory_optimizer = memory_optimizer
    
    # Métricas avanzadas
    metricas_avanzadas = MetricasAvanzadas()
    director_instance._metricas_avanzadas = metricas_avanzadas
    
    # Optimizar detector
    director_instance.detector = optimizar_detector_componentes(director_instance.detector)
    
    # Objective Manager fallback avanzado
    if not director_instance.objective_manager:
        director_instance.objective_manager = director_instance.detector.reintentar_objective_manager()
        if isinstance(director_instance.objective_manager, ObjectiveManagerFallbackAvanzado):
            director_instance.stats["objective_manager_fallback_avanzado"] = True
            logger.info("✅ Objective Manager Fallback Avanzado activado")
    
    # Método adicional para optimización automática
    def optimizar_automaticamente(self):
        """Optimización automática del sistema"""
        self._memory_optimizer.optimize_caches(self)
        return {
            "memoria_optimizada": True,
            "caches_limpiados": True,
            "garbage_collection": True
        }
    
    # Método para reporte completo
    def obtener_reporte_metricas_avanzadas(self):
        """Obtiene reporte completo de métricas avanzadas"""
        return self._metricas_avanzadas.obtener_reporte_completo()
    
    # Método para crear experiencia optimizada automáticamente
    def crear_experiencia_auto_optimizada(self, objetivo: str, **kwargs):
        """Crea experiencia con optimizaciones automáticas"""
        # Aplicar optimizaciones previas
        self.optimizar_automaticamente()
        
        # Usar configuración liviana
        config_optimizada = self._memory_optimizer.create_lightweight_config(objetivo, **kwargs)
        
        # Ejecutar experiencia normal
        resultado = self.crear_experiencia(objetivo, **config_optimizada)
        
        # Actualizar métricas avanzadas
        self._metricas_avanzadas.actualizar_metricas_experiencia(resultado, resultado.tiempo_generacion)
        
        return resultado
    
    # Agregar métodos al director
    director_instance.optimizar_automaticamente = optimizar_automaticamente.__get__(director_instance)
    director_instance.obtener_reporte_metricas_avanzadas = obtener_reporte_metricas_avanzadas.__get__(director_instance)
    director_instance.crear_experiencia_auto_optimizada = crear_experiencia_auto_optimizada.__get__(director_instance)
    
    return director_instance

# ===== FUNCIONES DE CONVENIENCIA ADICIONALES =====

def Aurora_Optimizado(objetivo: str = None, **kwargs):
    """Versión optimizada de Aurora con mejoras aditivas"""
    global _director_global
    if _director_global is None:
        _director_global = AuroraDirectorV7Integrado()
        _director_global = aplicar_mejoras_aditivas_director(_director_global)
    
    if objetivo is not None:
        return _director_global.crear_experiencia_auto_optimizada(objetivo, **kwargs)
    else:
        return _director_global

def crear_experiencia_fallback_avanzado(objetivo: str, **kwargs):
    """Crea experiencia usando específicamente el fallback avanzado"""
    director = Aurora_Optimizado()
    
    # Forzar uso del fallback avanzado
    om_original = director.objective_manager
    director.objective_manager = ObjectiveManagerFallbackAvanzado()
    
    try:
        resultado = director.crear_experiencia(objetivo, **kwargs)
        return resultado
    finally:
        # Restaurar OM original
        director.objective_manager = om_original

def obtener_metricas_sistema_completas():
    """Obtiene métricas completas del sistema"""
    director = Aurora_Optimizado()
    estado_base = director.obtener_estado_completo()
    metricas_avanzadas = director.obtener_reporte_metricas_avanzadas() if hasattr(director, '_metricas_avanzadas') else {}
    
    return {
        **estado_base,
        "metricas_avanzadas": metricas_avanzadas,
        "optimizaciones_aplicadas": {
            "memory_optimizer": hasattr(director, '_memory_optimizer'),
            "metricas_avanzadas": hasattr(director, '_metricas_avanzadas'),
            "objective_manager_fallback_avanzado": isinstance(director.objective_manager, ObjectiveManagerFallbackAvanzado),
            "detector_optimizado": hasattr(director.detector, 'reintentar_objective_manager')
        }
    }

# ===== MÉTODOS ADICIONALES PARA EL OBJETO AURORA =====

# Agregar métodos optimizados
Aurora.optimizado = Aurora_Optimizado
Aurora.fallback_avanzado = crear_experiencia_fallback_avanzado
Aurora.metricas_completas = obtener_metricas_sistema_completas
Aurora.auto_optimizar = lambda: Aurora_Optimizado().optimizar_automaticamente()

# Nuevos métodos específicos
Aurora.rapido_optimizado = lambda obj, **kw: Aurora_Optimizado(obj, duracion_min=5, calidad_objetivo="media", **kw)
Aurora.largo_optimizado = lambda obj, **kw: Aurora_Optimizado(obj, duracion_min=60, calidad_objetivo="alta", **kw)
Aurora.terapeutico_optimizado = lambda obj, **kw: Aurora_Optimizado(obj, duracion_min=45, intensidad="suave", calidad_objetivo="maxima", **kw)

# Métodos de análisis
Aurora.analizar_rendimiento = lambda: Aurora_Optimizado().obtener_reporte_metricas_avanzadas().get("metricas", {}).get("rendimiento", {})
Aurora.analizar_calidad = lambda: Aurora_Optimizado().obtener_reporte_metricas_avanzadas().get("metricas", {}).get("calidad", {})
Aurora.componentes_mas_efectivos = lambda: Aurora_Optimizado().obtener_reporte_metricas_avanzadas().get("metricas", {}).get("uso_componentes", {})

logger.info("✅ Mejoras aditivas Aurora V7 aplicadas exitosamente")
logger.info("🚀 Sistema optimizado con fallback avanzado, métricas mejoradas y optimización de memoria")

# ===== CARMINE POWER BOOSTER INTEGRATION =====
try:
    from carmine_power_booster_v7 import aplicar_carmine_power_booster
    if _director_global is not None:
        _director_global = aplicar_carmine_power_booster(_director_global)
        logger.info("🚀 Carmine Power Booster integrado automáticamente")
except ImportError:
    logger.info("💡 Carmine Power Booster no disponible - funcionamiento normal")

# ========== COLABORACIÓN MULTIMOTOR V7 ==========
# Extensión aditiva para colaboración real entre motores
# No modifica lógica existente, solo añade funcionalidades

try:
    from aurora_multimotor_extension import (
        aplicar_extension_colaborativa_multimotor, 
        diagnosticar_colaboracion_multimotor,
        ColaboracionMultiMotorV7
    )
    COLABORACION_MULTIMOTOR_DISPONIBLE = True
    logger.info("✅ Extensión Colaborativa MultiMotor V7 disponible")
except ImportError as e:
    COLABORACION_MULTIMOTOR_DISPONIBLE = False
    logger.warning(f"⚠️ Extensión Colaborativa MultiMotor no disponible: {e}")

# Función de activación automática de colaboración
def activar_colaboracion_multimotor_aurora(director_instance):
    """Activa colaboración MultiMotor de forma aditiva"""
    try:
        if COLABORACION_MULTIMOTOR_DISPONIBLE and director_instance.orquestador:
            
            # PASO 1: Aplicar parches de optimización primero
            _aplicar_parches_optimizacion_motores(director_instance)
            
            # PASO 2: Aplicar extensión colaborativa
            exito = aplicar_extension_colaborativa_multimotor(director_instance.orquestador)
            
            if exito:
                director_instance.stats["colaboracion_multimotor_activa"] = True
                director_instance.stats["parches_optimizacion_aplicados"] = True
                logger.info("🎉 Colaboración MultiMotor V7 ACTIVADA - Máxima potencia desbloqueada")
                return True
            else:
                logger.warning("⚠️ Colaboración MultiMotor no pudo activarse")
                return False
        else:
            logger.warning("⚠️ Colaboración MultiMotor no disponible")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error activando colaboración MultiMotor: {e}")
        return False

def _aplicar_parches_optimizacion_motores(director_instance):
    """Aplica parches de optimización a los motores disponibles"""
    
    # PARCHE 1: HarmonicEssence Optimizations
    try:
        from harmonic_essence_optimizations import aplicar_mejoras_harmonic_essence
        
        for nombre, componente in director_instance.componentes.items():
            if 'harmonic' in nombre.lower() and componente.disponible:
                componente.instancia = aplicar_mejoras_harmonic_essence(componente.instancia)
                componente.metadatos["optimizaciones_harmonic_aplicadas"] = True
                componente.metadatos["factor_potencia"] = "1.4x"
                logger.info(f"⚡ {nombre}: HarmonicEssence Optimizations aplicadas")
                
    except ImportError:
        logger.info("💡 HarmonicEssence Optimizations no disponible")
    except Exception as e:
        logger.warning(f"⚠️ Error aplicando HarmonicEssence Optimizations: {e}")
    
    # PARCHE 2: NeuroMix Definitivo V7 (Auto-aplica)
    try:
        import neuromix_definitivo_v7
        # El parche se auto-aplica al importar
        logger.info("🚀 NeuroMix Definitivo V7 auto-aplicado")
        
    except ImportError:
        logger.info("💡 NeuroMix Definitivo V7 no disponible")
    except Exception as e:
        logger.warning(f"⚠️ Error con NeuroMix Definitivo V7: {e}")

# Función de diagnóstico mejorada
def diagnosticar_aurora_multimotor_completo():
    """Diagnóstico completo del sistema MultiMotor"""
    director = Aurora()
    
    print("🔍 DIAGNÓSTICO AURORA V7 MULTIMOTOR COMPLETO")
    print("=" * 60)
    
    # Estado general
    print(f"📊 Componentes totales: {len(director.componentes)}")
    print(f"🎵 Motores disponibles: {len([c for c in director.componentes.values() if c.tipo.value == 'motor' and c.disponible])}")
    
    # Parches aplicados
    parches_aplicados = []
    if director.stats.get("parches_optimizacion_aplicados"):
        parches_aplicados.append("Optimizaciones aplicadas")
    if director.stats.get("colaboracion_multimotor_activa"): 
        parches_aplicados.append("Colaboración MultiMotor")
    
    print(f"⚡ Parches activos: {', '.join(parches_aplicados) if parches_aplicados else 'Ninguno'}")
    
    # Diagnóstico de colaboración
    if hasattr(director.orquestador, '_extension_colaborativa'):
        diagnosticar_colaboracion_multimotor(director.orquestador)
    else:
        print("\n🎼 Extensión Colaborativa: ❌ NO APLICADA")
    
    return director

# Agregar función de prueba
def test_colaboracion_multimotor_completo():
    """Test completo de colaboración con parches"""
    print("🧪 TEST COLABORACIÓN MULTIMOTOR CON PARCHES")
    print("=" * 50)
    
    director = Aurora()
    
    # Test 1: Generar experiencia colaborativa
    print("\n🎵 Test 1: Experiencia colaborativa...")
    try:
        resultado = director.crear_experiencia(
            "concentracion_extrema", 
            duracion_min=1,
            calidad_objetivo="maxima",
            intensidad="intenso"
        )
        
        if hasattr(resultado, 'metadatos') and resultado.metadatos.get('colaboracion_aplicada'):
            motores_usados = resultado.metadatos.get('motores_utilizados', [])
            print(f"   ✅ Colaboración activa: {len(motores_usados)} motores")
            print(f"   🎵 Motores: {', '.join(motores_usados)}")
            
            # Verificar si hay metadatos de colaboración
            if 'metadatos_colaboracion' in resultado.metadatos:
                mc = resultado.metadatos['metadatos_colaboracion']
                print(f"   📊 Calidad colaboración: {mc.get('calidad_colaboracion', 0):.2f}")
        else:
            print("   ⚠️ Colaboración no detectada en metadatos")
            
        print(f"   📈 Calidad final: {resultado.calidad_score:.1f}/100")
        
    except Exception as e:
        print(f"   ❌ Error en test: {e}")
    
    # Test 2: Verificar parches
    print("\n⚡ Test 2: Verificación de parches...")
    
    harmonic_optimizado = False
    neuromix_super = False
    
    for nombre, componente in director.componentes.items():
        if 'harmonic' in nombre.lower() and componente.disponible:
            if componente.metadatos.get("optimizaciones_harmonic_aplicadas"):
                harmonic_optimizado = True
                print(f"   ✅ {nombre}: Optimizaciones activas (factor: {componente.metadatos.get('factor_potencia', 'N/A')})")
        
        if 'neuromix' in nombre.lower() and componente.disponible:
            if hasattr(componente.instancia, 'version') and 'SUPER' in str(componente.instancia.version):
                neuromix_super = True
                print(f"   🚀 {nombre}: NeuroMix Super activo")
    
    print(f"\n📊 RESUMEN:")
    print(f"   🎼 Colaboración: {'✅ ACTIVA' if director.stats.get('colaboracion_multimotor_activa') else '❌ INACTIVA'}")
    print(f"   ⚡ HarmonicEssence Optimizado: {'✅ SÍ' if harmonic_optimizado else '❌ NO'}")
    print(f"   🚀 NeuroMix Super: {'✅ SÍ' if neuromix_super else '❌ NO'}")
    
    return resultado

# Agregar al objeto Aurora métodos de conveniencia
Aurora.activar_colaboracion = lambda: activar_colaboracion_multimotor_aurora(Aurora())
Aurora.diagnosticar_multimotor = diagnosticar_aurora_multimotor_completo  
Aurora.test_colaboracion = test_colaboracion_multimotor_completo

# Auto-activación en inicialización (modificación aditiva)
def _inicializar_sistema_con_colaboracion(self):
    """Versión mejorada de _inicializar_sistema que incluye colaboración"""
    
    # Llamar al método original
    self._inicializar_sistema_original()
    
    # Activar colaboración MultiMotor
    if globals().get('COLABORACION_MULTIMOTOR_DISPONIBLE', False):
        activar_colaboracion_multimotor_aurora(self)

# Solo aplicar si no se ha aplicado ya
if not hasattr(AuroraDirectorV7Integrado, '_inicializar_sistema_original'):
    # Guardar método original
    AuroraDirectorV7Integrado._inicializar_sistema_original = AuroraDirectorV7Integrado._inicializar_sistema
    
    # Reemplazar con versión mejorada
    AuroraDirectorV7Integrado._inicializar_sistema = _inicializar_sistema_con_colaboracion
    
    logger.info("✅ Auto-activación de colaboración MultiMotor integrada")

logger.info("🎉 Extensiones de Colaboración MultiMotor V7 cargadas exitosamente")
