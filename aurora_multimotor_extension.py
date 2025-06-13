"""
Aurora V7 - Extensión Colaborativa MultiMotor OPTIMIZADA
Mejora ADITIVA para activar colaboración real entre los 3 motores principales
OPTIMIZACIÓN CRÍTICA: Eliminación total de repeticiones y mejoras de rendimiento
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
from datetime import datetime
import importlib
import time
import threading
import hashlib
import weakref
from functools import lru_cache, wraps
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("Aurora.MultiMotor.Extension")
logger.setLevel(logging.WARNING)

# ========== SISTEMA DE CACHE OPTIMIZADO CON WEAK REFERENCES ==========

class EstadoProceso(Enum):
    """Estados de los procesos para mejor control"""
    PENDIENTE = "pendiente"
    EN_PROGRESO = "en_progreso"
    COMPLETADO = "completado"
    ERROR = "error"

@dataclass
class RegistroProceso:
    """Registro optimizado de proceso"""
    id_proceso: str
    estado: EstadoProceso
    timestamp_inicio: float
    timestamp_fin: Optional[float] = None
    intentos: int = 0
    error: Optional[str] = None

# Lock global optimizado para operaciones thread-safe
_AURORA_LOCK = threading.RLock()

# Cache global MEJORADO con gestión de memoria automática
class CacheGlobalOptimizado:
    """Cache global con gestión automática de memoria y TTL"""
    
    def __init__(self):
        self._procesos: Dict[str, RegistroProceso] = {}
        self._objetos_procesados: weakref.WeakSet = weakref.WeakSet()
        self._hashes_procesados: Dict[str, float] = {}
        self._interceptor_aplicado = False
        self._lock = threading.RLock()
        self._ttl_default = 300  # 5 minutos
        self._max_entradas = 1000  # Límite de entradas en cache
    
    def obtener_hash_objeto(self, obj) -> str:
        """Genera hash único y eficiente del objeto"""
        try:
            # Combinamos múltiples identificadores para máxima unicidad
            elementos = [
                str(id(obj)),
                type(obj).__name__,
                str(hash(str(type(obj)))),
                str(getattr(obj, '__dict__', {}))[:100]  # Solo primeros 100 chars
            ]
            combined = "|".join(elementos)
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
        except Exception:
            return f"fallback_{id(obj)}_{type(obj).__name__}"
    
    def proceso_en_progreso(self, proceso_id: str) -> bool:
        """Verifica si un proceso está en progreso de forma thread-safe"""
        with self._lock:
            registro = self._procesos.get(proceso_id)
            return registro and registro.estado == EstadoProceso.EN_PROGRESO
    
    def marcar_proceso_iniciado(self, proceso_id: str) -> bool:
        """Marca proceso como iniciado, retorna False si ya estaba iniciado"""
        with self._lock:
            if self.proceso_en_progreso(proceso_id):
                return False
            
            self._procesos[proceso_id] = RegistroProceso(
                id_proceso=proceso_id,
                estado=EstadoProceso.EN_PROGRESO,
                timestamp_inicio=time.time()
            )
            self._limpiar_cache_si_necesario()
            return True
    
    def marcar_proceso_completado(self, proceso_id: str, exitoso: bool = True):
        """Marca proceso como completado"""
        with self._lock:
            if proceso_id in self._procesos:
                registro = self._procesos[proceso_id]
                registro.estado = EstadoProceso.COMPLETADO if exitoso else EstadoProceso.ERROR
                registro.timestamp_fin = time.time()
    
    def fue_procesado_recientemente(self, objeto_hash: str, ttl: float = None) -> bool:
        """Verifica si objeto fue procesado recientemente con TTL configurable"""
        ttl = ttl or self._ttl_default
        with self._lock:
            if objeto_hash in self._hashes_procesados:
                tiempo_transcurrido = time.time() - self._hashes_procesados[objeto_hash]
                return tiempo_transcurrido < ttl
            return False
    
    def marcar_objeto_procesado(self, objeto_hash: str):
        """Marca objeto como procesado con timestamp"""
        with self._lock:
            self._hashes_procesados[objeto_hash] = time.time()
            self._limpiar_cache_si_necesario()
    
    def agregar_objeto_procesado(self, obj):
        """Agrega objeto al WeakSet de objetos procesados"""
        try:
            with self._lock:
                self._objetos_procesados.add(obj)
        except TypeError:
            # Algunos objetos no pueden estar en WeakSet
            pass
    
    def objeto_ya_procesado(self, obj) -> bool:
        """Verifica si objeto ya fue procesado usando WeakSet"""
        try:
            with self._lock:
                return obj in self._objetos_procesados
        except TypeError:
            return False
    
    def _limpiar_cache_si_necesario(self):
        """Limpia cache automáticamente si excede límites"""
        if len(self._hashes_procesados) > self._max_entradas:
            # Eliminar entradas más antiguas
            ahora = time.time()
            entradas_a_eliminar = []
            
            for hash_obj, timestamp in self._hashes_procesados.items():
                if ahora - timestamp > self._ttl_default * 2:  # Doble TTL para limpieza
                    entradas_a_eliminar.append(hash_obj)
            
            for hash_obj in entradas_a_eliminar:
                del self._hashes_procesados[hash_obj]
            
            logger.debug(f"🧹 Cache limpiado: {len(entradas_a_eliminar)} entradas eliminadas")
    
    def limpiar_todo(self):
        """Limpia completamente el cache"""
        with self._lock:
            self._procesos.clear()
            self._hashes_procesados.clear()
            self._objetos_procesados.clear()
            self._interceptor_aplicado = False
    
    @property
    def interceptor_aplicado(self) -> bool:
        with self._lock:
            return self._interceptor_aplicado
    
    @interceptor_aplicado.setter
    def interceptor_aplicado(self, valor: bool):
        with self._lock:
            self._interceptor_aplicado = valor

# Instancia global del cache optimizado
_CACHE_GLOBAL = CacheGlobalOptimizado()

# ========== DECORADORES PARA PREVENCIÓN DE REPETICIONES ==========

def una_sola_vez(ttl: float = 300, usar_argumentos: bool = False):
    """
    Decorador que asegura que una función se ejecute UNA SOLA VEZ por instancia
    
    Analogía: Como un portero que no deja pasar a la misma persona dos veces
    """
    def decorador(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generar ID único basado en función y argumentos si se especifica
            if usar_argumentos:
                args_str = str(args) + str(sorted(kwargs.items()))
                func_id = f"{func.__name__}_{hashlib.sha256(args_str.encode()).hexdigest()[:8]}"
            else:
                # Solo usar el primer argumento (self) para métodos de instancia
                obj_hash = _CACHE_GLOBAL.obtener_hash_objeto(args[0]) if args else "global"
                func_id = f"{func.__name__}_{obj_hash}"
            
            # Verificar si ya fue procesado recientemente
            if _CACHE_GLOBAL.fue_procesado_recientemente(func_id, ttl):
                logger.debug(f"🔄 {func.__name__} ya ejecutado recientemente, omitiendo...")
                return None
            
            # Verificar si está en progreso
            if not _CACHE_GLOBAL.marcar_proceso_iniciado(func_id):
                logger.debug(f"🔄 {func.__name__} ya en progreso...")
                return None
            
            try:
                logger.debug(f"🚀 Ejecutando {func.__name__} por primera vez...")
                resultado = func(*args, **kwargs)
                _CACHE_GLOBAL.marcar_proceso_completado(func_id, True)
                _CACHE_GLOBAL.marcar_objeto_procesado(func_id)
                return resultado
            except Exception as e:
                _CACHE_GLOBAL.marcar_proceso_completado(func_id, False)
                logger.error(f"❌ Error en {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorador

def thread_safe_singleton(func):
    """Decorador para métodos que deben ser singleton thread-safe"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _AURORA_LOCK:
            return func(*args, **kwargs)
    return wrapper

# ========== CLASE PRINCIPAL OPTIMIZADA ==========

class ColaboracionMultiMotorV7Optimizada:
    """
    Extensión OPTIMIZADA que mejora la colaboración entre motores
    
    Analogía: Como un director de orquesta eficiente que coordina UNA SOLA VEZ
    cada sección musical sin repetir instrucciones
    """
    
    _instancias_creadas: weakref.WeakSet = weakref.WeakSet()
    _cache_motores_colaborativos = {}
    
    def __init__(self, orquestador_original):
        # Verificar si esta instancia ya fue procesada
        if _CACHE_GLOBAL.objeto_ya_procesado(orquestador_original):
            logger.debug("✅ Orquestador ya tiene extensión, reutilizando...")
            self._cargar_desde_cache_existente(orquestador_original)
            return
        
        self.id_unico = _CACHE_GLOBAL.obtener_hash_objeto(orquestador_original)
        self.orquestador = orquestador_original
        
        # Agregar a seguimiento
        _CACHE_GLOBAL.agregar_objeto_procesado(orquestador_original)
        ColaboracionMultiMotorV7Optimizada._instancias_creadas.add(self)
        
        # Inicialización optimizada
        self._inicializar_componentes()
        
        logger.info("🎼 Extensión Colaborativa MultiMotor V7 OPTIMIZADA inicializada")
    
    @una_sola_vez(ttl=600)  # TTL más largo para inicialización
    def _inicializar_componentes(self):
        """Inicializa componentes UNA SOLA VEZ con TTL extendido"""
        logger.debug(f"🆕 Inicializando componentes para {self.id_unico[:12]}...")
        
        self.motores_colaborativos = {}
        self.mapeo_nombres = {}
        self.estadisticas_colaboracion = {
            "total_colaboraciones": 0,
            "motores_activos_por_sesion": [],
            "mejoras_aplicadas": 0,
            "rescates_exitosos": 0,
            "motores_reales_activados": 0,
            "detecciones_exitosas": 0,
            "detecciones_fallidas": 0,
            "timestamp_init": time.time()
        }
        
        # Ejecutar procesos principales
        self._ejecutar_rescate_optimizado()
        self._ejecutar_deteccion_optimizada()
        self._guardar_en_cache_global()
    
    def _cargar_desde_cache_existente(self, orquestador_original):
        """Carga configuración desde cache existente de forma eficiente"""
        if hasattr(orquestador_original, '_extension_colaborativa'):
            ext_existente = orquestador_original._extension_colaborativa
            self.motores_colaborativos = getattr(ext_existente, 'motores_colaborativos', {})
            self.mapeo_nombres = getattr(ext_existente, 'mapeo_nombres', {})
            self.estadisticas_colaboracion = getattr(ext_existente, 'estadisticas_colaboracion', {})
            self.orquestador = orquestador_original
            logger.debug(f"🔄 Configuración cargada desde extensión existente")
    
    @una_sola_vez(ttl=300)
    def _ejecutar_rescate_optimizado(self):
        """Ejecuta rescate de motores de forma optimizada"""
        logger.debug("🚀 Ejecutando rescate optimizado de motores...")
        
        director_instance = self._obtener_director_cached()
        if not director_instance:
            logger.warning("⚠️ Director no disponible para rescate")
            return
        
        motores_rescatados = self._rescatar_motores_inteligente(director_instance)
        
        if motores_rescatados > 0:
            self.estadisticas_colaboracion["rescates_exitosos"] += 1
            self.estadisticas_colaboracion["motores_reales_activados"] += motores_rescatados
            logger.debug(f"✅ Rescate optimizado: {motores_rescatados} motores activados")
    
    @lru_cache(maxsize=1)
    def _obtener_director_cached(self):
        """Obtiene el director usando cache LRU para eficiencia"""
        try:
            import aurora_director_v7
        
            # ✅ PRIMERA OPCIÓN: Director global existente
            if hasattr(aurora_director_v7, '_director_global') and aurora_director_v7._director_global:
                return aurora_director_v7._director_global
        
            # ✅ SEGUNDA OPCIÓN: Crear via Aurora()
            director = aurora_director_v7.Aurora()
            if hasattr(director, 'orquestador'):
                return director
            
            # ✅ TERCERA OPCIÓN: Usar singleton pattern de Aurora
            if hasattr(aurora_director_v7, 'obtener_aurora_director_global'):
                return aurora_director_v7.obtener_aurora_director_global()
            
        except ImportError:
            logger.warning("⚠️ No se pudo importar aurora_director_v7")
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo director: {e}")
    
        # ✅ FALLBACK FINAL: Crear director básico si todo falla
        try:
            import aurora_director_v7
            return aurora_director_v7.crear_aurora_director_refactorizado()
        except:
            logger.error("❌ No se pudo obtener ninguna instancia del director")
            return None
    
    def _rescatar_motores_inteligente(self, director_instance) -> int:
        """Rescata motores de forma inteligente y eficiente"""
        motores_rescatados = 0
        
        # Configuración optimizada para rescate
        config_rescate = self._obtener_configuracion_rescate()
        
        # Procesamiento por lotes para eficiencia
        motores_a_rescatar = []
        
        for nombre_motor, componente in director_instance.componentes.items():
            if self._motor_necesita_rescate(componente):
                motores_a_rescatar.append((nombre_motor, componente))
        
        logger.info(f"🔧 Procesando {len(motores_a_rescatar)} motores para rescate...")
        
        # Rescatar motores en lotes
        for nombre_motor, componente in motores_a_rescatar:
            if self._rescatar_motor_individual(nombre_motor, componente, config_rescate):
                motores_rescatados += 1
        
        return motores_rescatados
    
    @lru_cache(maxsize=1)
    def _obtener_configuracion_rescate(self) -> Dict[str, Any]:
        """Obtiene configuración de rescate cacheada"""
        return {
            'neuromix_v27': {
                'modulos': ['neuromix_aurora_v27', 'neuromix_definitivo_v7'],
                'clases': ['AuroraNeuroAcousticEngineV27', 'NeuroMixSuperUnificado'],
                'fallback_types': ['NMF'],
                'prioridad': 1
            },
            'hypermod_v32': {
                'modulos': ['hypermod_v32'],
                'clases': ['HyperModEngineV32AuroraConnected', 'NeuroWaveGenerator'],
                'fallback_types': ['HMF'],
                'prioridad': 1
            },
            'harmonic_essence_v34': {
                'modulos': ['harmonicEssence_v34'],
                'clases': ['HarmonicEssenceV34AuroraConnected'],
                'fallback_types': ['HF'],
                'prioridad': 1
            }
        }
    
    def _motor_necesita_rescate(self, componente) -> bool:
        """Verifica si un motor necesita rescate de forma eficiente"""
        try:
            tipo_actual = type(componente.instancia).__name__
            return tipo_actual.endswith('F')  # Es fallback
        except Exception:
            return False
    
    def _rescatar_motor_individual(self, nombre_motor: str, componente, config_rescate: Dict) -> bool:
        """Rescata un motor individual de forma optimizada"""
        # Generar ID único para este rescate específico
        rescate_id = f"rescate_{nombre_motor}_{id(componente)}"
        
        if not _CACHE_GLOBAL.marcar_proceso_iniciado(rescate_id):
            return False  # Ya en proceso
        
        try:
            tipo_actual = type(componente.instancia).__name__
            
            # Buscar configuración apropiada
            config_motor = None
            for config_name, config_data in config_rescate.items():
                if tipo_actual in config_data['fallback_types']:
                    config_motor = config_data
                    break
            
            if not config_motor:
                return False
            
            # Crear motor real
            motor_real = self._crear_motor_real_eficiente(config_motor)
            
            if motor_real and self._validar_motor_eficiente(motor_real):
                # Aplicar reemplazo
                componente.instancia = motor_real
                componente.version = getattr(motor_real, 'version', 'rescatado_v7')
                componente.disponible = True
                componente.metadatos.update({
                    'rescatado': True,
                    'motor_original_fallback': tipo_actual,
                    'clase_real': type(motor_real).__name__,
                    'timestamp_rescate': datetime.now().isoformat()
                })
                
                logger.info(f"   ✅ {nombre_motor}: {tipo_actual} → {type(motor_real).__name__}")
                _CACHE_GLOBAL.marcar_proceso_completado(rescate_id, True)
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"   ❌ Error rescatando {nombre_motor}: {e}")
            _CACHE_GLOBAL.marcar_proceso_completado(rescate_id, False)
            return False
    
    def _crear_motor_real_eficiente(self, config_motor: Dict):
        """Crea motor real de forma eficiente con cache de módulos"""
        for modulo_nombre in config_motor['modulos']:
            # Usar cache de módulos para evitar importaciones repetidas
            modulo = self._importar_modulo_cached(modulo_nombre)
            
            if modulo:
                for clase_nombre in config_motor['clases']:
                    if hasattr(modulo, clase_nombre):
                        try:
                            ClaseMotor = getattr(modulo, clase_nombre)
                            return ClaseMotor()
                        except Exception as e:
                            logger.debug(f"     ❌ Error creando {clase_nombre}: {e}")
        
        return None
    
    @lru_cache(maxsize=10)
    def _importar_modulo_cached(self, modulo_nombre: str):
        """Importa módulos usando cache LRU para eficiencia"""
        try:
            return importlib.import_module(modulo_nombre)
        except ImportError:
            return None
    
    def _validar_motor_eficiente(self, motor) -> bool:
        """Validación eficiente de motores"""
        return (hasattr(motor, 'generar_audio') and 
                callable(getattr(motor, 'generar_audio')))
    
    @una_sola_vez(ttl=300)
    def _ejecutar_deteccion_optimizada(self):
        """Ejecuta detección optimizada de motores colaborativos"""
        logger.info("🔍 Ejecutando detección optimizada de motores colaborativos...")
        
        # Patrones optimizados con scoring inteligente
        patrones_optimizados = self._obtener_patrones_deteccion_optimizados()
        
        # Clasificación inicial eficiente
        motores_reales, motores_fallback = self._clasificar_motores_eficiente()
        
        logger.info(f"📊 Clasificación: {len(motores_reales)} reales, {len(motores_fallback)} fallbacks")
        
        # Detección por prioridad
        self._detectar_motores_por_prioridad(motores_reales, patrones_optimizados, True)
        self._detectar_motores_por_prioridad(motores_fallback, patrones_optimizados, False)
        
        # Resultado optimizado
        motores_encontrados = len(self.motores_colaborativos)
        logger.info(f"🎼 Detección completada: {motores_encontrados}/3 motores colaborativos")
        
        if motores_encontrados >= 2:
            logger.info("🎉 Sistema listo para colaboración!")
    
    @lru_cache(maxsize=1)
    def _obtener_patrones_deteccion_optimizados(self) -> Dict[str, Any]:
        """Obtiene patrones de detección optimizados con cache"""
        return {
            'neuromix': {
                'keywords': ['neuromix', 'neuro', 'acoustic'],
                'class_patterns': ['NeuroAcoustic', 'NeuroMix', 'Super.*Unificado'],
                'valid_classes': ['AuroraNeuroAcousticEngineV27', 'NeuroMixSuperUnificado'],
                'fallback_types': ['NMF'],
                'score_base': 10
            },
            'harmonic': {
                'keywords': ['harmonic', 'essence', 'textured'],
                'class_patterns': ['Harmonic', 'Essence', 'Textured'],
                'valid_classes': ['HarmonicEssenceV34AuroraConnected'],
                'fallback_types': ['HF'],
                'score_base': 10
            },
            'hypermod': {
                'keywords': ['hypermod', 'hyper', 'wave', 'brainwave'],
                'class_patterns': ['HyperMod', 'Wave.*Gen', 'NeuroWave'],
                'valid_classes': ['HyperModEngineV32AuroraConnected', 'NeuroWaveGenerator'],
                'fallback_types': ['HMF'],
                'score_base': 10
            }
        }
    
    def _clasificar_motores_eficiente(self) -> Tuple[Dict, Dict]:
        """Clasifica motores de forma eficiente"""
        motores_reales = {}
        motores_fallback = {}
        
        for nombre_motor, componente in self.orquestador.motores_disponibles.items():
            tipo_instancia = type(componente.instancia).__name__
            
            if tipo_instancia.endswith('F'):
                motores_fallback[nombre_motor] = componente
            else:
                motores_reales[nombre_motor] = componente
        
        return motores_reales, motores_fallback
    
    def _detectar_motores_por_prioridad(self, motores: Dict, patrones: Dict, son_reales: bool):
        """Detecta motores optimizando por prioridad y evitando repeticiones"""
        for nombre_motor, componente in motores.items():
            if not componente.disponible:
                continue
            
            # Búsqueda optimizada de coincidencias
            mejor_match = self._buscar_mejor_coincidencia_optimizada(
                nombre_motor, 
                type(componente.instancia).__name__, 
                patrones
            )
            
            if mejor_match:
                tipo_motor, score, detalles = mejor_match
                
                # Validación rápida
                if self._validar_motor_colaborativo_rapido(componente, patrones[tipo_motor]):
                    self._agregar_motor_colaborativo_optimizado(
                        tipo_motor, nombre_motor, componente, score, son_reales
                    )
                    
                    estado = "REAL" if son_reales else "FALLBACK"
                    logger.info(f"   ✅ {estado} {tipo_motor}: {nombre_motor} (score: {score})")
                    self.estadisticas_colaboracion['detecciones_exitosas'] += 1
    
    def _buscar_mejor_coincidencia_optimizada(self, nombre_motor: str, tipo_instancia: str, patrones: Dict) -> Optional[Tuple[str, int, Dict]]:
        """Búsqueda optimizada de coincidencias con scoring inteligente"""
        mejor_score = 0
        mejor_match = None
        
        nombre_lower = nombre_motor.lower()
        
        for tipo_motor, config in patrones.items():
            score = 0
            detalles = {}
            
            # Scoring por keywords (optimizado)
            for keyword in config['keywords']:
                if keyword in nombre_lower:
                    score += config['score_base']
                    detalles['keyword_match'] = keyword
                    break
            
            # Scoring por clase exacta (máxima prioridad)
            if tipo_instancia in config['valid_classes']:
                score += config['score_base'] * 2
                detalles['exact_class'] = True
            
            # Scoring por patrones de clase
            import re
            for pattern in config['class_patterns']:
                if re.search(pattern, tipo_instancia, re.IGNORECASE):
                    score += config['score_base'] // 2
                    detalles['pattern_match'] = pattern
                    break
            
            if score > mejor_score:
                mejor_score = score
                mejor_match = (tipo_motor, score, detalles)
        
        return mejor_match if mejor_score >= 10 else None
    
    def _validar_motor_colaborativo_rapido(self, componente, config: Dict) -> bool:
        """Validación rápida y eficiente de motores"""
        try:
            instancia = componente.instancia
            
            # Verificación básica de método requerido
            if not (hasattr(instancia, 'generar_audio') and 
                    callable(getattr(instancia, 'generar_audio'))):
                return False
            
            # Verificación de clase válida
            tipo_actual = type(instancia).__name__
            return (tipo_actual in config['valid_classes'] or 
                    not tipo_actual.endswith('F'))  # No es fallback
            
        except Exception:
            return False
    
    def _agregar_motor_colaborativo_optimizado(self, tipo_motor: str, nombre_motor: str, 
                                             componente, score: int, es_real: bool):
        """Agrega motor colaborativo de forma optimizada"""
        self.motores_colaborativos[tipo_motor] = {
            'nombre_original': nombre_motor,
            'componente': componente,
            'instancia': componente.instancia,
            'capabilities': self._obtener_capacidades_cached(componente),
            'funcional': True,
            'tipo_instancia': type(componente.instancia).__name__,
            'validacion_completa': True,
            'score_deteccion': score,
            'es_motor_real': es_real,
            'preferencia': 'alta' if es_real else 'baja',
            'timestamp_deteccion': time.time()
        }
        self.mapeo_nombres[nombre_motor] = tipo_motor
    
    @lru_cache(maxsize=20)
    def _obtener_capacidades_cached(self, componente) -> Dict[str, Any]:
        """Obtiene capacidades usando cache para eficiencia"""
        try:
            instancia = componente.instancia
            
            # Intentar métodos de capacidades estándar
            for metodo in ['obtener_capacidades', 'get_capabilities']:
                if hasattr(instancia, metodo):
                    result = getattr(instancia, metodo)()
                    if isinstance(result, dict):
                        return result
            
            # Fallback a capacidades básicas
            return {
                "nombre": type(instancia).__name__,
                "tipo": "motor_colaborativo",
                "version": getattr(instancia, 'version', 'unknown'),
                "metodos_disponibles": self._obtener_metodos_cached(instancia)
            }
            
        except Exception:
            return {"capacidades": "error", "fallback": True}
    
    @lru_cache(maxsize=50)
    def _obtener_metodos_cached(self, instancia) -> List[str]:
        """Obtiene métodos disponibles usando cache"""
        try:
            return [attr for attr in dir(instancia) 
                   if not attr.startswith('_') and callable(getattr(instancia, attr))][:10]
        except Exception:
            return []
    
    def _guardar_en_cache_global(self):
        """Guarda configuración en cache global de forma thread-safe"""
        with _AURORA_LOCK:
            ColaboracionMultiMotorV7Optimizada._cache_motores_colaborativos[self.id_unico] = {
                'motores': self.motores_colaborativos.copy(),
                'mapeo': self.mapeo_nombres.copy(),
                'timestamp': time.time()
            }
    
    def aplicar_colaboracion_mejorada(self, config: Dict[str, Any], duracion_sec: float) -> Optional[Dict[str, Any]]:
        """Aplica colaboración mejorada con optimizaciones de rendimiento"""
        if len(self.motores_colaborativos) < 2:
            logger.warning(f"⚠️ Colaboración no disponible: solo {len(self.motores_colaborativos)} motores")
            return None
        
        # ID único para esta colaboración específica
        collab_id = f"collab_{hash(str(config))}_{int(duracion_sec)}_{int(time.time())}"
        
        if not _CACHE_GLOBAL.marcar_proceso_iniciado(collab_id):
            logger.debug("🔄 Colaboración similar ya en proceso...")
            return None
        
        try:
            logger.info(f"🎼 Iniciando colaboración optimizada con {len(self.motores_colaborativos)} motores")
            
            # Configuración optimizada
            config_colaborativa = self._crear_configuracion_optimizada(config)
            
            # Generación paralela de capas (simulada con procesamiento secuencial optimizado)
            capas_colaborativas = self._generar_capas_optimizadas(config, duracion_sec, config_colaborativa)
            
            if capas_colaborativas:
                # Mezcla optimizada
                audio_final = self._mezclar_capas_optimizada(capas_colaborativas)
                
                # Actualizar estadísticas
                self._actualizar_estadisticas_optimizada(capas_colaborativas)
                
                return self._crear_resultado_colaboracion(audio_final, capas_colaborativas, config_colaborativa)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en colaboración optimizada: {e}")
            return None
        finally:
            _CACHE_GLOBAL.marcar_proceso_completado(collab_id)
    
    def _crear_configuracion_optimizada(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea configuración colaborativa optimizada"""
        objetivo = config.get('objetivo', 'relajacion').lower()
        
        # Configuraciones predefinidas optimizadas
        configs_optimizadas = {
            'concentracion': {
                'roles': {'neuromix': 'base_neuroacustica', 'hypermod': 'modulacion_alpha', 'harmonic': 'textura_focus'},
                'pesos': {'neuromix': 0.5, 'hypermod': 0.4, 'harmonic': 0.25}
            },
            'relajacion': {
                'roles': {'harmonic': 'ambiente_principal', 'neuromix': 'ondas_theta', 'hypermod': 'modulacion_suave'},
                'pesos': {'harmonic': 0.5, 'neuromix': 0.35, 'hypermod': 0.25}
            }
        }
        
        # Seleccionar y normalizar configuración
        config_base = configs_optimizadas.get(objetivo, configs_optimizadas['relajacion'])
        
        # Normalizar pesos según motores disponibles
        motores_disponibles = set(self.motores_colaborativos.keys())
        pesos_normalizados = {}
        total_peso = 0
        
        for motor in motores_disponibles:
            if motor in config_base['pesos']:
                pesos_normalizados[motor] = config_base['pesos'][motor]
                total_peso += config_base['pesos'][motor]
        
        # Normalización final
        if total_peso > 0:
            for motor in pesos_normalizados:
                pesos_normalizados[motor] /= total_peso
        
        config_base['pesos'] = pesos_normalizados
        return config_base
    
    def _generar_capas_optimizadas(self, config: Dict, duracion_sec: float, config_colaborativa: Dict) -> Dict[str, Any]:
        """Genera capas de audio de forma optimizada"""
        capas_colaborativas = {}
        
        for tipo_motor, motor_info in self.motores_colaborativos.items():
            try:
                # Configuración específica por motor
                config_motor = self._crear_config_motor_optimizada(config, tipo_motor, config_colaborativa)
                
                # Generación de audio con medición de tiempo
                start_time = time.time()
                instancia = motor_info['instancia']
                audio_capa = instancia.generar_audio(config_motor, duracion_sec)
                tiempo_generacion = time.time() - start_time
                
                # Validación optimizada
                if self._validar_audio_optimizado(audio_capa):
                    peso = config_colaborativa['pesos'][tipo_motor]
                    capas_colaborativas[tipo_motor] = {
                        'audio': audio_capa * peso,
                        'peso': peso,
                        'rol': config_colaborativa['roles'][tipo_motor],
                        'tiempo_generacion': tiempo_generacion,
                        'clase_motor': type(instancia).__name__
                    }
                    logger.debug(f"   ✅ {tipo_motor}: audio generado ({tiempo_generacion:.2f}s)")
                
            except Exception as e:
                logger.warning(f"   ❌ Error en {tipo_motor}: {e}")
                continue
        
        return capas_colaborativas
    
    def _crear_config_motor_optimizada(self, config_base: Dict, tipo_motor: str, config_colaborativa: Dict) -> Dict[str, Any]:
        """Crea configuración optimizada específica para cada motor"""
        config_motor = {
            'objetivo': config_base.get('objetivo', 'relajacion'),
            'duracion_min': config_base.get('duracion_min', 20),
            'sample_rate': config_base.get('sample_rate', 44100),
            'intensidad': config_base.get('intensidad', 'media'),
            'calidad_objetivo': config_base.get('calidad_objetivo', 'alta'),
            'normalizar': config_base.get('normalizar', True)
        }
        
        # Optimizaciones específicas por tipo de motor
        optimizaciones = {
            'neuromix': {'processing_mode': 'aurora_integrated', 'quality_level': 'enhanced'},
            'harmonic': {'enable_neural_fades': True, 'precision_cientifica': True},
            'hypermod': {'validacion_cientifica': True, 'optimizacion_neuroacustica': True}
        }
        
        if tipo_motor in optimizaciones:
            config_motor.update(optimizaciones[tipo_motor])
        
        return config_motor
    
    def _validar_audio_optimizado(self, audio) -> bool:
        """Validación optimizada de audio generado"""
        try:
            return (audio is not None and 
                    isinstance(audio, np.ndarray) and 
                    audio.size > 0 and 
                    not np.all(audio == 0) and 
                    audio.ndim <= 2)
        except Exception:
            return False
    
    def _mezclar_capas_optimizada(self, capas_colaborativas: Dict[str, Dict]) -> np.ndarray:
        """Mezcla capas de forma optimizada usando vectorización numpy"""
        if not capas_colaborativas:
            return np.zeros((2, int(44100 * 10)), dtype=np.float32)
        
        # Determinar dimensiones óptimas
        max_samples = max(
            self._obtener_samples_audio(info['audio']) 
            for info in capas_colaborativas.values()
        )
        
        # Pre-allocar array de mezcla
        mezcla_final = np.zeros((2, max_samples), dtype=np.float32)
        total_peso = sum(info['peso'] for info in capas_colaborativas.values())
        
        # Mezcla vectorizada
        for motor_info in capas_colaborativas.values():
            audio = motor_info['audio']
            peso_normalizado = motor_info['peso'] / total_peso if total_peso > 0 else motor_info['peso']
            
            # Conversión eficiente a estéreo
            audio_stereo = self._convertir_a_estereo_optimizado(audio, max_samples)
            
            # Suma vectorizada
            mezcla_final += audio_stereo * peso_normalizado
        
        # Normalización optimizada
        return self._normalizar_audio_optimizado(mezcla_final)
    
    def _obtener_samples_audio(self, audio: np.ndarray) -> int:
        """Obtiene número de samples de forma eficiente"""
        if audio.ndim == 1:
            return len(audio)
        return audio.shape[1] if audio.shape[0] == 2 else audio.shape[0]
    
    def _convertir_a_estereo_optimizado(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Convierte audio a estéreo de forma optimizada"""
        # Conversión a estéreo
        if audio.ndim == 1:
            audio_stereo = np.stack([audio, audio])
        else:
            audio_stereo = audio.copy()
            if audio_stereo.shape[0] != 2:
                if audio_stereo.shape[1] == 2:
                    audio_stereo = audio_stereo.T
                else:
                    audio_stereo = np.stack([audio_stereo[0], audio_stereo[0]])
        
        # Ajuste de longitud optimizado
        current_samples = audio_stereo.shape[1]
        if current_samples < target_samples:
            # Padding con zeros
            padding = np.zeros((2, target_samples - current_samples), dtype=audio_stereo.dtype)
            audio_stereo = np.concatenate([audio_stereo, padding], axis=1)
        elif current_samples > target_samples:
            # Truncar
            audio_stereo = audio_stereo[:, :target_samples]
        
        return audio_stereo
    
    def _normalizar_audio_optimizado(self, audio: np.ndarray) -> np.ndarray:
        """Normalización optimizada usando operaciones vectorizadas"""
        max_peak = np.max(np.abs(audio))
        
        if max_peak > 0.9:
            return audio * (0.85 / max_peak)
        elif max_peak < 0.3 and max_peak > 0:
            return audio * (0.7 / max_peak)
        
        return audio
    
    def _actualizar_estadisticas_optimizada(self, capas_colaborativas: Dict):
        """Actualiza estadísticas de forma optimizada"""
        self.estadisticas_colaboracion["total_colaboraciones"] += 1
        self.estadisticas_colaboracion["motores_activos_por_sesion"].append(len(capas_colaborativas))
        self.estadisticas_colaboracion["mejoras_aplicadas"] += 1
    
    def _crear_resultado_colaboracion(self, audio_final: np.ndarray, capas_colaborativas: Dict, config_colaborativa: Dict) -> Dict[str, Any]:
        """Crea resultado de colaboración optimizado"""
        return {
            'success': True,
            'audio_data': audio_final,
            'capas_utilizadas': list(capas_colaborativas.keys()),
            'metadatos_colaboracion': {
                'motores_activos': len(capas_colaborativas),
                'roles_asignados': {motor: info['rol'] for motor, info in capas_colaborativas.items()},
                'pesos_aplicados': {motor: info['peso'] for motor, info in capas_colaborativas.items()},
                'clases_motores': {motor: info['clase_motor'] for motor, info in capas_colaborativas.items()},
                'tiempos_generacion': {motor: info['tiempo_generacion'] for motor, info in capas_colaborativas.items()},
                'calidad_colaboracion': self._calcular_calidad_optimizada(capas_colaborativas),
                'configuracion_utilizada': config_colaborativa,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _calcular_calidad_optimizada(self, capas_colaborativas: Dict) -> float:
        """Cálculo optimizado de calidad de colaboración"""
        try:
            motores_activos = len(capas_colaborativas)
            cobertura = min(1.0, motores_activos / 3.0)
            
            # Cálculo vectorizado de balance de pesos
            pesos = np.array([info['peso'] for info in capas_colaborativas.values()])
            balance_pesos = 1.0 - min(1.0, np.var(pesos) / (np.mean(pesos) ** 2 + 1e-6))
            
            # Factor de tiempo optimizado
            tiempos = np.array([info.get('tiempo_generacion', 1.0) for info in capas_colaborativas.values()])
            factor_tiempo = min(1.0, 2.0 / (np.mean(tiempos) + 1.0))
            
            return min(1.0, max(0.0, cobertura * 0.4 + balance_pesos * 0.3 + factor_tiempo * 0.3))
            
        except Exception:
            return 0.5
    
    def obtener_estadisticas_colaboracion(self) -> Dict[str, Any]:
        """Retorna estadísticas optimizadas con cálculos eficientes"""
        motores_activos = self.estadisticas_colaboracion["motores_activos_por_sesion"]
        promedio_motores = np.mean(motores_activos) if motores_activos else 0
        
        return {
            'motores_detectados': len(self.motores_colaborativos),
            'motores_disponibles': list(self.motores_colaborativos.keys()),
            'mapeo_nombres': self.mapeo_nombres,
            'estadisticas': self.estadisticas_colaboracion,
            'promedio_motores_por_sesion': promedio_motores,
            'detalle_motores': {
                motor: {
                    'nombre_original': info['nombre_original'],
                    'tipo_instancia': info['tipo_instancia'],
                    'funcional': info['funcional'],
                    'score_deteccion': info.get('score_deteccion', 0),
                    'es_motor_real': info.get('es_motor_real', False)
                }
                for motor, info in self.motores_colaborativos.items()
            },
            'calidad_sistema': self._evaluar_calidad_sistema_optimizada()
        }
    
    def _evaluar_calidad_sistema_optimizada(self) -> str:
        """Evaluación optimizada de calidad del sistema"""
        motores_detectados = len(self.motores_colaborativos)
        
        if motores_detectados >= 3:
            return "excelente"
        elif motores_detectados >= 2:
            return "buena"
        elif motores_detectados >= 1:
            return "limitada"
        else:
            return "sin_colaboracion"


# ========== FUNCIONES DE APLICACIÓN OPTIMIZADAS ==========

@thread_safe_singleton
def aplicar_extension_colaborativa_optimizada(orquestador_multimotor):
    """
    Aplica la extensión colaborativa OPTIMIZADA SIN REPETICIONES
    
    Analogía: Como instalar un software UNA SOLA VEZ en una computadora
    """
    # Verificar si ya tiene extensión
    if hasattr(orquestador_multimotor, '_extension_colaborativa_optimizada'):
        logger.debug("✅ Extensión optimizada ya aplicada")
        return True
    
    try:
        # Crear extensión optimizada
        extension = ColaboracionMultiMotorV7Optimizada(orquestador_multimotor)
        orquestador_multimotor._extension_colaborativa_optimizada = extension
        
        # Aplicar métodos optimizados
        _aplicar_metodos_optimizados(orquestador_multimotor)
        
        # Aplicar interceptor global UNA SOLA VEZ
        if not _CACHE_GLOBAL.interceptor_aplicado:
            if _aplicar_interceptor_optimizado():
                _CACHE_GLOBAL.interceptor_aplicado = True
        
        # Estadísticas finales
        stats = extension.obtener_estadisticas_colaboracion()
        logger.info(f"🎉 Extensión OPTIMIZADA aplicada: {stats['motores_detectados']}/3 motores, calidad: {stats['calidad_sistema']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error aplicando extensión optimizada: {e}")
        return False

@una_sola_vez(ttl=600)
def _aplicar_metodos_optimizados(orquestador_multimotor):
    """Aplica métodos optimizados UNA SOLA VEZ"""
    def generar_audio_colaborativo_optimizado(self, config, duracion_sec):
        """Método optimizado para generación colaborativa"""
        try:
            if hasattr(self, '_extension_colaborativa_optimizada'):
                resultado = self._extension_colaborativa_optimizada.aplicar_colaboracion_mejorada(config, duracion_sec)
                
                if resultado and resultado.get('success'):
                    # Cache de metadatos optimizado
                    metadatos = {
                        'motores_utilizados': resultado['capas_utilizadas'],
                        'colaboracion_aplicada': True,
                        'metadatos_colaboracion': resultado['metadatos_colaboracion'],
                        'colaboracion_multimotor_optimizada': True,
                        'motor_principal': resultado['capas_utilizadas'][0] if resultado['capas_utilizadas'] else 'multimotor'
                    }
                    
                    # Cache thread-safe
                    with _AURORA_LOCK:
                        if not hasattr(self, '_metadatos_cache_optimizado'):
                            self._metadatos_cache_optimizado = {}
                        self._metadatos_cache_optimizado.update(metadatos)
                    
                    return resultado['audio_data'], metadatos
                    
        except Exception as e:
            logger.debug(f"⚠️ Error colaboración optimizada: {e}")
        
        # Fallback optimizado
        if hasattr(self, 'generar_audio_orquestado_original'):
            return self.generar_audio_orquestado_original(config, duracion_sec)
        else:
            return self.generar_audio_orquestado(config, duracion_sec)
    
    # Aplicar parche solo si no existe
    if not hasattr(orquestador_multimotor, 'generar_audio_orquestado_original'):
        orquestador_multimotor.generar_audio_orquestado_original = orquestador_multimotor.generar_audio_orquestado
        orquestador_multimotor.generar_audio_orquestado = generar_audio_colaborativo_optimizado.__get__(orquestador_multimotor)
        logger.debug("✅ Métodos optimizados aplicados")

@una_sola_vez(ttl=3600)  # TTL largo para interceptor global
def _aplicar_interceptor_optimizado() -> bool:
    """Aplica interceptor optimizado UNA SOLA VEZ globalmente"""
    try:
        import sys
        
        # Obtener módulo director
        if 'aurora_director_v7' in sys.modules:
            modulo_director = sys.modules['aurora_director_v7']
        else:
            import aurora_director_v7
            modulo_director = aurora_director_v7
        
        if not hasattr(modulo_director, 'AuroraDirectorV7Integrado'):
            return False
        
        ClaseDirector = modulo_director.AuroraDirectorV7Integrado
        
        # Verificar si ya se aplicó
        if hasattr(ClaseDirector, '_interceptor_optimizado_aplicado'):
            return True
        
        # Guardar método original
        if not hasattr(ClaseDirector, '_post_procesar_original'):
            ClaseDirector._post_procesar_original = getattr(ClaseDirector, '_post_procesar_resultado_v7_2', 
                                                           getattr(ClaseDirector, '_post_procesar_resultado', None))
        
        if not ClaseDirector._post_procesar_original:
            return False
        
        def _post_procesar_optimizado(self, resultado, config):
            """Post-procesamiento optimizado con corrección de metadatos"""
            # Procesamiento original
            try:
                resultado_procesado = self._post_procesar_original(resultado, config)
            except Exception as e:
                logger.warning(f"⚠️ Error en post-procesamiento: {e}")
                resultado_procesado = resultado
            
            # Corrección optimizada de metadatos
            try:
                if (hasattr(self, 'orquestador') and 
                    hasattr(self.orquestador, '_metadatos_cache_optimizado')):
                    
                    with _AURORA_LOCK:
                        cache = self.orquestador._metadatos_cache_optimizado
                        
                        if hasattr(resultado_procesado, 'metadatos') and cache:
                            resultado_procesado.metadatos.update(cache)
                            resultado_procesado.metadatos['interceptor_optimizado_aplicado'] = True
                            
                            logger.debug("✅ Metadatos optimizados aplicados")
                        
                        # Limpiar cache después de usar
                        cache.clear()
                        
            except Exception as e:
                logger.warning(f"⚠️ Error en corrección optimizada: {e}")
            
            return resultado_procesado
        
        # Aplicar interceptor optimizado
        ClaseDirector._post_procesar_resultado_v7_2 = _post_procesar_optimizado
        ClaseDirector._interceptor_optimizado_aplicado = True
        
        logger.info("✅ Interceptor optimizado aplicado")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Error aplicando interceptor optimizado: {e}")
        return False


# ========== FUNCIONES DE UTILIDAD OPTIMIZADAS ==========

def diagnosticar_sistema_optimizado():
    """Diagnóstico optimizado del sistema completo"""
    print("🔍 DIAGNÓSTICO SISTEMA AURORA V7 MULTIMOTOR OPTIMIZADO")
    print("=" * 80)
    
    try:
        from aurora_director_v7 import Aurora
        director = Aurora()
        
        print(f"\n📊 ESTADO GENERAL:")
        print(f"   🌟 Director: {type(director).__name__}")
        print(f"   📦 Componentes: {len(director.componentes)}")
        print(f"   🎵 Motores: {len(director.orquestador.motores_disponibles)}")
        
        # Análisis optimizado de motores
        motores_reales = 0
        motores_fallback = 0
        
        for nombre, componente in director.orquestador.motores_disponibles.items():
            tipo = type(componente.instancia).__name__
            if tipo.endswith('F'):
                motores_fallback += 1
                print(f"   ⚠️ {nombre}: {tipo} (FALLBACK)")
            else:
                motores_reales += 1
                print(f"   ✅ {nombre}: {tipo} (REAL)")
        
        print(f"\n📈 RESUMEN: {motores_reales} reales, {motores_fallback} fallbacks")
        
        # Estado de extensión optimizada
        if hasattr(director.orquestador, '_extension_colaborativa_optimizada'):
            ext = director.orquestador._extension_colaborativa_optimizada
            stats = ext.obtener_estadisticas_colaboracion()
            
            print(f"\n🎼 EXTENSIÓN OPTIMIZADA: ✅ ACTIVA")
            print(f"   🎵 Motores colaborativos: {stats['motores_detectados']}/3")
            print(f"   🏆 Calidad: {stats['calidad_sistema']}")
            print(f"   📊 Colaboraciones: {stats['estadisticas']['total_colaboraciones']}")
        else:
            print(f"\n🎼 EXTENSIÓN OPTIMIZADA: ❌ NO APLICADA")
        
        print("=" * 80)
        return director
        
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")
        return None

def test_colaboracion_optimizada():
    """Test de colaboración optimizada"""
    print("🧪 TEST COLABORACIÓN MULTIMOTOR OPTIMIZADA")
    print("=" * 60)
    
    try:
        from aurora_director_v7 import Aurora
        director = Aurora()
        
        # Aplicar extensión optimizada
        if aplicar_extension_colaborativa_optimizada(director.orquestador):
            print("✅ Extensión optimizada aplicada")
        
        # Test de experiencia
        print(f"\n🎵 GENERANDO EXPERIENCIA OPTIMIZADA...")
        resultado = director.crear_experiencia(
            "test_optimizado",
            duracion_min=1,
            calidad_objetivo="maxima"
        )
        
        # Análisis de resultados
        colaboracion = resultado.metadatos.get('colaboracion_aplicada', False)
        motores = resultado.metadatos.get('motores_utilizados', [])
        optimizada = resultado.metadatos.get('colaboracion_multimotor_optimizada', False)
        
        print(f"\n📊 RESULTADOS OPTIMIZADOS:")
        print(f"   🎯 Calidad: {resultado.calidad_score:.1f}/100")
        print(f"   🎼 Colaboración: {'✅' if colaboracion else '❌'}")
        print(f"   ⚡ Optimizada: {'✅' if optimizada else '❌'}")
        print(f"   🎵 Motores: {len(motores)} - {', '.join(motores)}")
        
        if len(motores) >= 2 and (colaboracion or optimizada):
            print(f"\n🎉 ¡COLABORACIÓN OPTIMIZADA EXITOSA!")
        
        return resultado
        
    except Exception as e:
        print(f"❌ Error en test optimizado: {e}")
        return None

def limpiar_cache_optimizado():
    """Limpia el cache optimizado completamente"""
    _CACHE_GLOBAL.limpiar_todo()
    
    # Limpiar caches LRU
    ColaboracionMultiMotorV7Optimizada._obtener_director_cached.cache_clear()
    
    print("🧹 Cache optimizado limpiado completamente")


if __name__ == "__main__":
    print("🎼 Aurora V7 - Extensión Colaborativa MultiMotor OPTIMIZADA")
    print("⚡ Mejoras de rendimiento y eliminación completa de repeticiones")
    print("🚀 Cache inteligente con WeakReferences y TTL automático")
    print("🔧 Decoradores de prevención y thread-safety mejorado")
    print("📊 Procesamiento vectorizado y optimizaciones numpy")
    
    # Ejecutar test optimizado
    print(f"\n" + "="*80)
    print("🧪 EJECUTANDO TEST OPTIMIZADO...")
    test_colaboracion_optimizada()
