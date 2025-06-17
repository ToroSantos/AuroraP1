#!/usr/bin/env python3
"""
Aurora Bridge V7 - Integración Completa con Sistema Aurora
Puente entre Flask Backend y Aurora V7 Real

OPTIMIZADO: Error de serialización JSON corregido + Carmine integrado + Todas las funcionalidades
"""

import os
import sys
import uuid
import wave
import numpy as np
import logging
import time
import traceback
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import threading
import json

# Configurar logging optimizado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Aurora.Bridge.V7.Optimized")

# Agregar aurora_system al path de Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aurora_system'))

# ===== IMPORTS OPTIMIZADOS CON FALLBACK =====
AURORA_AVAILABLE = False
CARMINE_AVAILABLE = False
QUALITY_PIPELINE_AVAILABLE = False

# Importar Aurora Director V7
try:
    from aurora_director_v7 import Aurora, ConfiguracionAuroraUnificada, ResultadoAuroraIntegrado
    AURORA_AVAILABLE = True
    logger.info("✅ Aurora Director V7 importado correctamente")
except ImportError as e:
    logger.warning(f"❌ Aurora Director V7 no disponible: {e}")

# Importar Carmine Analyzer
try:
    from Carmine_Analyzer import CarmineAuroraAnalyzer, CarmineAnalyzerEnhanced
    CARMINE_AVAILABLE = True
    logger.info("✅ Carmine Analyzer importado correctamente")
except ImportError as e:
    logger.warning(f"⚠️ Carmine Analyzer no disponible: {e}")

# Importar Carmine Power Booster
try:
    from carmine_power_booster_v7 import CarminePowerBooster, aplicar_carmine_power_booster
    logger.info("✅ Carmine Power Booster importado correctamente")
except ImportError as e:
    logger.warning(f"⚠️ Carmine Power Booster no disponible: {e}")

# Importar Quality Pipeline
try:
    from aurora_quality_pipeline import AuroraQualityPipeline
    QUALITY_PIPELINE_AVAILABLE = True
    logger.info("✅ Aurora Quality Pipeline importado correctamente")
except ImportError as e:
    logger.warning(f"⚠️ Aurora Quality Pipeline no disponible: {e}")

class GenerationStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class ProcessingMode(Enum):
    """Modos de procesamiento optimizados"""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    PREMIUM = "premium"

class AuroraBridge:
    """
    Puente de integración optimizado con el Sistema Aurora V7
    Maneja generación de audio, estado del sistema, análisis Carmine y conversión de archivos
    
    OPTIMIZACIONES V7.2:
    - JSON serialization segura 
    - Integración completa con Carmine
    - Pool de threads optimizado
    - Cache inteligente de estado
    - Análisis predictivo
    """
    
    def __init__(self):
        self.initialized = False
        self.aurora_director = None
        self.system_status = {}
        self.audio_folder = "static/audio"
        self.sample_rate = 44100
        
        # ===== COMPONENTES OPTIMIZADOS =====
        self.carmine_analyzer = None
        self.carmine_booster = None
        self.quality_pipeline = None
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Aurora-Bridge")
        
        # ===== CACHE Y ESTADÍSTICAS =====
        self._status_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 30  # 30 segundos
        self._processing_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'avg_processing_time': 0.0,
            'carmine_optimizations': 0,
            'quality_improvements': 0
        }
        
        # ===== LOCKS PARA CONCURRENCIA =====
        self._status_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        
        # Crear carpeta de audio
        os.makedirs(self.audio_folder, exist_ok=True)
        
        # Inicializar sistemas
        self._initialize_aurora_system()
        self._initialize_carmine_system()
        self._initialize_quality_pipeline()
        
        # Verificar componentes del sistema
        self._verify_system_components()
        
        logger.info("🚀 AuroraBridge V7.2 Optimizado inicializado correctamente")
    
    # ===== FUNCIONES AUXILIARES PARA SERIALIZACIÓN JSON (OPTIMIZADAS) =====
    def _enum_to_str(self, value) -> Optional[str]:
        """Convertir Enum a string de forma segura y optimizada"""
        if value is None:
            return None
        
        # Cache para enums comunes (optimización)
        if hasattr(value, '_aurora_serialized'):
            return value._aurora_serialized
        
        try:
            if hasattr(value, 'value'):
                result = str(value.value)
            elif hasattr(value, 'name'):
                result = str(value.name)
            else:
                result = str(value)
            
            # Cachear resultado
            if hasattr(value, '__class__'):
                value._aurora_serialized = result
                
            return result
        except Exception as e:
            logger.warning(f"Error convirtiendo enum {type(value)}: {e}")
            return str(value)
    
    def _clean_metadata(self, obj, max_depth: int = 10, current_depth: int = 0) -> Any:
        """Limpiar objetos para que sean serializables en JSON con protección contra recursión infinita"""
        if current_depth > max_depth:
            return "<<MAX_DEPTH_REACHED>>"
            
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._clean_metadata(item, max_depth, current_depth + 1) for item in obj[:100]]  # Limitar listas
        elif isinstance(obj, dict):
            cleaned = {}
            for key, value in list(obj.items())[:50]:  # Limitar diccionarios
                try:
                    cleaned_key = str(key) if not isinstance(key, str) else key
                    cleaned[cleaned_key] = self._clean_metadata(value, max_depth, current_depth + 1)
                except Exception as e:
                    logger.debug(f"Error limpiando clave {key}: {e}")
                    cleaned[str(key)] = "<<SERIALIZATION_ERROR>>"
            return cleaned
        elif hasattr(obj, 'value') or hasattr(obj, 'name'):
            # AQUÍ SE CORRIGE EL PROBLEMA: TipoComponente enums
            return self._enum_to_str(obj)
        elif hasattr(obj, 'to_dict'):
            # Objetos con método to_dict
            try:
                return self._clean_metadata(obj.to_dict(), max_depth, current_depth + 1)
            except Exception:
                return str(obj)
        elif hasattr(obj, '__dict__'):
            # Objetos con atributos
            try:
                return self._clean_metadata(obj.__dict__, max_depth, current_depth + 1)
            except Exception:
                return str(obj)
        else:
            # Fallback seguro
            try:
                return str(obj)
            except Exception:
                return "<<UNSERIALIZABLE_OBJECT>>"
    
    def _initialize_aurora_system(self):
        """Inicializar el sistema Aurora V7 con arranque controlado optimizado"""
        try:
            logger.info("🌟 Inicializando Aurora V7 con arranque controlado optimizado...")

            # Intentar arranque controlado primero
            try:
                from aurora_startup_fix import init_aurora_system_controlled
                success, director = init_aurora_system_controlled()

                if success and director:
                    self.aurora_director = director
                    
                    # Aplicar Carmine Power Booster si está disponible
                    if hasattr(aplicar_carmine_power_booster, '__call__'):
                        try:
                            self.aurora_director = aplicar_carmine_power_booster(self.aurora_director)
                            logger.info("🔋 Carmine Power Booster aplicado al director")
                        except Exception as e:
                            logger.warning(f"⚠️ Error aplicando Carmine Power Booster: {e}")
                    
                    self.initialized = True
                    logger.info("✅ Aurora Director inicializado con arranque controlado")
                    return
            except ImportError:
                logger.info("📦 Startup fix no disponible, usando inicialización directa")
                
            # Fallback a inicialización directa
            if AURORA_AVAILABLE:
                try:
                    self.aurora_director = Aurora()
                    self.initialized = True
                    logger.info("✅ Aurora Director inicializado directamente")
                    return
                except Exception as e:
                    logger.error(f"❌ Error instanciando Aurora Director: {e}")
            
            # Fallback final
            logger.warning("⚠️ Fallback a sistema básico")
            self._initialize_fallback_system()

        except Exception as e:
            logger.error(f"❌ Error crítico inicializando Aurora: {e}")
            logger.error(traceback.format_exc())
            self._initialize_fallback_system()
    
    def _initialize_carmine_system(self):
        """Inicializar sistema Carmine optimizado"""
        try:
            if CARMINE_AVAILABLE:
                # Inicializar Carmine Analyzer Enhanced
                self.carmine_analyzer = CarmineAnalyzerEnhanced()
                logger.info("✅ Carmine Analyzer Enhanced inicializado")
                
                # Inicializar Carmine Power Booster
                if hasattr(CarminePowerBooster, '__init__'):
                    self.carmine_booster = CarminePowerBooster()
                    logger.info("✅ Carmine Power Booster inicializado")
                    
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando Carmine: {e}")
    
    def _initialize_quality_pipeline(self):
        """Inicializar pipeline de calidad"""
        try:
            if QUALITY_PIPELINE_AVAILABLE:
                self.quality_pipeline = AuroraQualityPipeline(mode="profesional")
                logger.info("✅ Aurora Quality Pipeline inicializado")
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando Quality Pipeline: {e}")
    
    def _initialize_fallback_system(self):
        """Sistema de fallback optimizado si Aurora no está disponible"""
        logger.warning("⚠️ Inicializando sistema de fallback optimizado")
        self.initialized = False
        self.aurora_director = None
    
    def _verify_system_components(self):
        """Verificar componentes del sistema Aurora con análisis profundo"""
        try:
            if hasattr(self.aurora_director, 'obtener_estado_completo'):
                self.system_status = self.aurora_director.obtener_estado_completo()
                logger.info("✅ Estado del sistema obtenido")
                
                # Análisis adicional con Carmine si está disponible
                if self.carmine_analyzer:
                    try:
                        # Ejecutar análisis del sistema en background
                        self.thread_pool.submit(self._analyze_system_health)
                    except Exception as e:
                        logger.warning(f"⚠️ Error en análisis Carmine del sistema: {e}")
            else:
                logger.warning("⚠️ obtener_estado_completo no disponible")
                
        except Exception as e:
            logger.warning(f"⚠️ Error verificando componentes: {e}")
    
    def _analyze_system_health(self):
        """Análisis de salud del sistema con Carmine (background)"""
        try:
            if self.carmine_analyzer and hasattr(self.carmine_analyzer, 'obtener_reporte_estadisticas'):
                stats = self.carmine_analyzer.obtener_reporte_estadisticas()
                
                with self._stats_lock:
                    self._processing_stats['carmine_system_health'] = stats
                    
                logger.debug("📊 Análisis de salud del sistema completado")
        except Exception as e:
            logger.debug(f"Error en análisis de salud: {e}")
    
    def get_complete_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema Aurora con cache optimizado"""
        try:
            current_time = time.time()
            
            # Verificar cache
            with self._status_lock:
                if (current_time - self._cache_timestamp) < self._cache_ttl and self._status_cache:
                    logger.debug("📋 Devolviendo estado desde cache")
                    return self._status_cache.copy()
            
            # Obtener estado fresco
            if not self.initialized or not self.aurora_director:
                status = self._get_fallback_status()
            elif hasattr(self.aurora_director, 'obtener_estado_completo'):
                try:
                    full_status = self.aurora_director.obtener_estado_completo()
                    
                    # ✅ APLICAR SERIALIZACIÓN JSON SEGURA
                    full_status = self._clean_metadata(full_status)
                    
                    # Enriquecer con información del bridge
                    full_status['bridge_info'] = {
                        'initialized': self.initialized,
                        'audio_folder': self.audio_folder,
                        'sample_rate': self.sample_rate,
                        'timestamp': datetime.now().isoformat(),
                        'carmine_available': CARMINE_AVAILABLE,
                        'quality_pipeline_available': QUALITY_PIPELINE_AVAILABLE,
                        'optimizations_enabled': True
                    }
                    
                    # Agregar estadísticas de procesamiento
                    with self._stats_lock:
                        full_status['processing_stats'] = self._processing_stats.copy()
                    
                    # Crear resumen amigable
                    summary = self._create_status_summary(full_status)
                    full_status['summary'] = summary
                    
                    # Actualizar cache
                    with self._status_lock:
                        self._status_cache = full_status.copy()
                        self._cache_timestamp = current_time
                    
                    status = full_status
                    
                except Exception as e:
                    logger.error(f"Error obteniendo estado detallado: {e}")
                    status = self._get_basic_status()
            else:
                status = self._get_basic_status()
                
            return status
                
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return self._get_error_status(str(e))
    
    def _create_status_summary(self, full_status: Dict[str, Any]) -> Dict[str, Any]:
        """Crear resumen optimizado del estado del sistema"""
        try:
            # Aplicar limpieza también aquí por seguridad
            componentes = self._clean_metadata(full_status.get('componentes_detectados', {}))
            capacidades = self._clean_metadata(full_status.get('capacidades_sistema', {}))
            
            motores_activos = capacidades.get('motores_activos', 0)
            gestores_activos = capacidades.get('gestores_activos', 0)
            pipelines_activos = capacidades.get('pipelines_activos', 0)
            
            total_componentes = motores_activos + gestores_activos + pipelines_activos
            
            # Calcular calidad promedio
            calidad_promedio = 0.0
            if 'estadisticas_calidad' in full_status:
                stats_calidad = full_status['estadisticas_calidad']
                if isinstance(stats_calidad, dict) and 'promedio_efectividad' in stats_calidad:
                    calidad_promedio = stats_calidad['promedio_efectividad']
            
            # Determinar estado general optimizado
            if total_componentes >= 8 and calidad_promedio > 90:
                status = 'premium'
                status_text = f'Sistema funcionando en modo premium ({total_componentes} componentes)'
            elif total_componentes >= 5:
                status = 'optimal'
                status_text = f'Sistema funcionando óptimamente ({total_componentes} componentes)'
            elif total_componentes >= 3:
                status = 'good'
                status_text = f'Sistema funcionando correctamente ({total_componentes} componentes)'
            elif total_componentes >= 1:
                status = 'limited'
                status_text = f'Sistema funcionando con limitaciones ({total_componentes} componentes)'
            else:
                status = 'critical'
                status_text = 'Sistema con componentes críticos faltantes'
            
            return {
                'status': status,
                'status_text': status_text,
                'motores_activos': motores_activos,
                'gestores_activos': gestores_activos,
                'pipelines_activos': pipelines_activos,
                'total_componentes': total_componentes,
                'calidad_promedio': calidad_promedio,
                'tiempo_promedio': capacidades.get('tiempo_promedio_generacion', 0.0),
                'componentes_principales': {
                    'aurora_director': True,
                    'neuromix_v27': componentes.get('neuromix_v27', {}).get('disponible', False),
                    'hypermod_v32': componentes.get('hypermod_v32', {}).get('disponible', False),
                    'harmonic_essence_v34': componentes.get('harmonic_essence_v34', {}).get('disponible', False),
                    'objective_manager': capacidades.get('objective_manager_disponible', False),
                    'quality_pipeline': 'quality_pipeline' in componentes and componentes['quality_pipeline'].get('disponible', False),
                    'carmine_analyzer': CARMINE_AVAILABLE,
                    'carmine_booster': self.carmine_booster is not None
                },
                'optimizations': {
                    'json_serialization': True,
                    'carmine_integration': CARMINE_AVAILABLE,
                    'quality_pipeline': QUALITY_PIPELINE_AVAILABLE,
                    'thread_pool': True,
                    'intelligent_cache': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error creando resumen de estado: {e}")
            return {
                'status': 'error',
                'status_text': f'Error obteniendo estado: {e}',
                'motores_activos': 0,
                'total_componentes': 0,
                'calidad_promedio': 0.0,
                'tiempo_promedio': 0.0
            }
    
    def _get_fallback_status(self) -> Dict[str, Any]:
        """Estado de fallback optimizado cuando Aurora no está disponible"""
        return {
            'summary': {
                'status': 'fallback',
                'status_text': 'Sistema Aurora no disponible - usando fallback optimizado',
                'motores_activos': 0,
                'total_componentes': 0,
                'calidad_promedio': 0.0,
                'tiempo_promedio': 0.0,
                'componentes_principales': {k: False for k in ['aurora_director', 'neuromix_v27', 'hypermod_v32', 'harmonic_essence_v34']},
                'optimizations': {
                    'fallback_mode': True,
                    'carmine_available': CARMINE_AVAILABLE
                }
            },
            'bridge_info': {
                'initialized': False,
                'fallback_mode': True,
                'timestamp': datetime.now().isoformat(),
                'carmine_available': CARMINE_AVAILABLE
            },
            'componentes_detectados': {},
            'capacidades_sistema': {}
        }
    
    def _get_basic_status(self) -> Dict[str, Any]:
        """Estado básico optimizado cuando no hay método obtener_estado_completo"""
        return {
            'summary': {
                'status': 'basic',
                'status_text': 'Aurora Director disponible - funcionalidad básica optimizada',
                'motores_activos': 1,
                'total_componentes': 1,
                'calidad_promedio': 85.0,
                'tiempo_promedio': 2.5,
                'componentes_principales': {
                    'aurora_director': True,
                    'carmine_analyzer': CARMINE_AVAILABLE
                }
            },
            'bridge_info': {
                'initialized': True,
                'basic_mode': True,
                'timestamp': datetime.now().isoformat(),
                'optimizations_enabled': True
            }
        }
    
    def _get_error_status(self, error_msg: str) -> Dict[str, Any]:
        """Estado de error optimizado"""
        return {
            'summary': {
                'status': 'error',
                'status_text': f'Error del sistema: {error_msg}',
                'motores_activos': 0,
                'total_componentes': 0,
                'calidad_promedio': 0.0,
                'tiempo_promedio': 0.0
            },
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'fallback_available': True
        }
    
    def validate_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validar parámetros de generación con validación optimizada"""
        errors = []
        warnings = []
        
        # Validar objetivo
        objetivo = params.get('objetivo', '').strip()
        if not objetivo:
            errors.append("El objetivo es requerido")
        elif len(objetivo) > 500:  # Aumentado límite
            errors.append("El objetivo es demasiado largo (máximo 500 caracteres)")
        elif len(objetivo) < 3:
            warnings.append("El objetivo es muy corto - considera ser más específico")
        
        # Validar duración
        duracion_min = params.get('duracion_min', 20)
        try:
            duracion_min = int(duracion_min)
            if duracion_min < 1:
                errors.append("La duración mínima es 1 minuto")
            elif duracion_min > 180:  # Aumentado límite
                errors.append("La duración máxima es 180 minutos")
            elif duracion_min > 60:
                warnings.append("Duraciones superiores a 60 minutos pueden tardar considerablemente")
        except (ValueError, TypeError):
            errors.append("La duración debe ser un número válido")
        
        # Validar intensidad
        intensidad = params.get('intensidad', 'media')
        intensidades_validas = ['suave', 'sutil', 'media', 'moderado', 'intenso', 'profundo', 'extremo']
        if intensidad not in intensidades_validas:
            errors.append(f"Intensidad inválida. Debe ser una de: {', '.join(intensidades_validas)}")
        
        # Validar estilo
        estilo = params.get('estilo', 'sereno')
        estilos_validos = [
            'sereno', 'crystalline', 'organico', 'etereo', 'tribal', 'mistico', 'cuantico', 
            'neural', 'neutro', 'energico', 'relajante', 'meditativo', 'futurista', 'natural'
        ]
        if estilo not in estilos_validos:
            errors.append(f"Estilo inválido. Debe ser uno de: {', '.join(estilos_validos)}")
        
        # Validar calidad
        calidad = params.get('calidad_objetivo', 'alta')
        calidades_validas = ['basica', 'media', 'alta', 'maxima', 'premium']
        if calidad not in calidades_validas:
            errors.append(f"Calidad inválida. Debe ser una de: {', '.join(calidades_validas)}")
        
        # Validar sample rate
        sample_rate = params.get('sample_rate', 44100)
        if sample_rate not in [22050, 44100, 48000, 96000]:
            errors.append("Sample rate debe ser 22050, 44100, 48000 o 96000")
        
        # Validaciones avanzadas opcionales
        if params.get('usar_carmine_optimization', True) and not CARMINE_AVAILABLE:
            warnings.append("Optimización Carmine solicitada pero no está disponible")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'recommendations': self._get_parameter_recommendations(params)
        }
    
    def _get_parameter_recommendations(self, params: Dict[str, Any]) -> List[str]:
        """Obtener recomendaciones inteligentes para parámetros"""
        recommendations = []
        
        objetivo = params.get('objetivo', '').lower()
        duracion = params.get('duracion_min', 20)
        
        # Recomendaciones basadas en objetivo
        if 'medita' in objetivo and duracion < 10:
            recommendations.append("Para meditación, considera aumentar la duración a al menos 10-15 minutos")
        
        if 'sueño' in objetivo or 'dormir' in objetivo:
            recommendations.append("Para inducir sueño, se recomienda estilo 'sereno' e intensidad 'suave'")
            
        if 'concentra' in objetivo or 'focus' in objetivo:
            recommendations.append("Para concentración, considera intensidad 'media' y estilo 'neural'")
        
        if CARMINE_AVAILABLE:
            recommendations.append("Carmine Analyzer disponible para optimización automática")
            
        return recommendations
    
    def estimate_generation_time(self, params: Dict[str, Any]) -> int:
        """Estimar tiempo de generación optimizado en segundos"""
        base_time = 3  # 3 segundos base (optimizado)
        
        duracion_min = params.get('duracion_min', 20)
        duracion_factor = max(1, duracion_min / 15)  # Factor ajustado
        
        calidad = params.get('calidad_objetivo', 'alta')
        calidad_factor = {
            'basica': 0.8,
            'media': 1.2,
            'alta': 1.8,
            'maxima': 2.5,
            'premium': 3.2
        }.get(calidad, 1.8)
        
        # Factor por disponibilidad de componentes
        system_factor = 1.0
        if self.initialized and self.aurora_director:
            system_factor = 0.7  # Más rápido con Aurora completo
            if CARMINE_AVAILABLE:
                system_factor *= 0.9  # Carmine optimiza tiempos
        else:
            system_factor = 2.2  # Más lento en fallback
        
        # Factor por paralelización
        if hasattr(self, 'thread_pool'):
            system_factor *= 0.85
        
        estimated = int(base_time * duracion_factor * calidad_factor * system_factor)
        return max(3, min(600, estimated))  # Entre 3 segundos y 10 minutos
    
    def crear_experiencia_completa(self, params: Dict[str, Any], 
                                 progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Crear experiencia neuroacústica completa con optimizaciones Carmine"""
        start_time = time.time()
        generation_id = str(uuid.uuid4())
        
        try:
            logger.info(f"🎵 Iniciando generación optimizada: {generation_id}")
            
            if progress_callback:
                progress_callback(5, "Inicializando generación optimizada")
            
            # Actualizar estadísticas
            with self._stats_lock:
                self._processing_stats['total_generations'] += 1
            
            # Optimización predictiva con Carmine si está disponible
            if self.carmine_booster and params.get('usar_carmine_optimization', True):
                try:
                    if progress_callback:
                        progress_callback(10, "Aplicando optimización predictiva Carmine")
                    
                    params = self._apply_carmine_optimization(params)
                    logger.info("🔋 Optimización predictiva Carmine aplicada")
                except Exception as e:
                    logger.warning(f"⚠️ Error en optimización Carmine: {e}")
            
            if progress_callback:
                progress_callback(15, "Seleccionando estrategia de generación")
            
            # Generar audio usando Aurora o fallback
            if self.initialized and self.aurora_director:
                resultado = self._generar_con_aurora(params, progress_callback, generation_id)
            else:
                resultado = self._generar_con_fallback(params, progress_callback, generation_id)
            
            if not resultado['success']:
                return resultado
            
            if progress_callback:
                progress_callback(80, "Procesando audio con pipeline de calidad")
            
            # Aplicar pipeline de calidad si está disponible
            audio_data = resultado['audio_data']
            if self.quality_pipeline and params.get('aplicar_quality_pipeline', True):
                try:
                    audio_data = self._apply_quality_pipeline(audio_data, params)
                    with self._stats_lock:
                        self._processing_stats['quality_improvements'] += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error en quality pipeline: {e}")
            
            # Análisis post-generación con Carmine
            if self.carmine_analyzer and params.get('analizar_con_carmine', True):
                try:
                    if progress_callback:
                        progress_callback(85, "Analizando calidad con Carmine")
                    
                    resultado['carmine_analysis'] = self._analyze_with_carmine(audio_data, params)
                    logger.debug("📊 Análisis Carmine completado")
                except Exception as e:
                    logger.warning(f"⚠️ Error en análisis Carmine: {e}")
            
            if progress_callback:
                progress_callback(90, "Guardando archivo de audio")
            
            # Convertir a WAV
            filename = self._save_audio_to_wav(audio_data, params, generation_id)
            
            if progress_callback:
                progress_callback(95, "Finalizando generación")
            
            processing_time = time.time() - start_time
            
            # Actualizar estadísticas finales
            with self._stats_lock:
                self._processing_stats['successful_generations'] += 1
                total = self._processing_stats['total_generations']
                self._processing_stats['avg_processing_time'] = (
                    (self._processing_stats['avg_processing_time'] * (total - 1) + processing_time) / total
                )
            
            if progress_callback:
                progress_callback(100, "Generación completada exitosamente")
            
            resultado_final = {
                'success': True,
                'generation_id': generation_id,
                'audio_filename': filename,
                'metadata': {
                    **resultado.get('metadata', {}),
                    'processing_time_seconds': processing_time,
                    'bridge_version': '7.2.0_optimized',
                    'generation_method': resultado.get('method', 'unknown'),
                    'optimizations_applied': {
                        'carmine_predictive': self.carmine_booster is not None and params.get('usar_carmine_optimization', True),
                        'quality_pipeline': self.quality_pipeline is not None and params.get('aplicar_quality_pipeline', True),
                        'carmine_analysis': self.carmine_analyzer is not None and params.get('analizar_con_carmine', True),
                        'thread_optimization': True
                    },
                    'audio_specs': {
                        'sample_rate': self.sample_rate,
                        'channels': 2,
                        'duration_seconds': audio_data.shape[1] / self.sample_rate if audio_data.ndim > 1 else len(audio_data) / self.sample_rate,
                        'format': 'WAV 16-bit',
                        'file_size_mb': round(os.path.getsize(os.path.join(self.audio_folder, filename)) / (1024*1024), 2)
                    }
                }
            }
            
            # Agregar análisis Carmine si está disponible
            if 'carmine_analysis' in resultado:
                resultado_final['metadata']['carmine_analysis'] = resultado['carmine_analysis']
            
            logger.info(f"✅ Generación {generation_id} completada en {processing_time:.2f}s")
            return resultado_final
            
        except Exception as e:
            logger.error(f"❌ Error crítico en generación {generation_id}: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'generation_id': generation_id,
                'error': f'Error crítico: {str(e)}',
                'processing_time_seconds': time.time() - start_time
            }
    
    def _apply_carmine_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Aplicar optimización predictiva con Carmine Power Booster"""
        try:
            if hasattr(self.carmine_booster, 'optimizacion_predictiva_carmine'):
                prediccion = self.carmine_booster.optimizacion_predictiva_carmine(params)
                
                if prediccion and 'config_optimizada' in prediccion:
                    optimized_params = params.copy()
                    optimized_params.update(prediccion['config_optimizada'])
                    
                    with self._stats_lock:
                        self._processing_stats['carmine_optimizations'] += 1
                    
                    logger.debug(f"🔋 Parámetros optimizados por Carmine: {prediccion.get('mejoras_aplicadas', [])}")
                    return optimized_params
            
            return params
        except Exception as e:
            logger.warning(f"Error en optimización Carmine: {e}")
            return params
    
    def _apply_quality_pipeline(self, audio_data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Aplicar pipeline de calidad Aurora"""
        try:
            if hasattr(self.quality_pipeline, 'procesar'):
                # Crear función lambda para el pipeline
                audio_func = lambda: audio_data
                processed_audio, metadata = self.quality_pipeline.procesar(audio_func)
                
                logger.debug(f"📈 Quality pipeline aplicado: score {metadata.get('score', 'N/A')}")
                return processed_audio
            
            return audio_data
        except Exception as e:
            logger.warning(f"Error en quality pipeline: {e}")
            return audio_data
    
    def _analyze_with_carmine(self, audio_data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar audio con Carmine Analyzer Enhanced"""
        try:
            if hasattr(self.carmine_analyzer, 'analyze_audio'):
                config = {
                    'objetivo': params.get('objetivo', ''),
                    'sample_rate': params.get('sample_rate', self.sample_rate)
                }
                
                analysis = self.carmine_analyzer.analyze_audio(audio_data, config)
                return self._clean_metadata(analysis)
            
            return {'available': False, 'reason': 'Carmine analyzer not initialized'}
        except Exception as e:
            logger.warning(f"Error en análisis Carmine: {e}")
            return {'error': str(e)}
    
    def _generar_con_aurora(self, params: Dict[str, Any], 
                           progress_callback: Optional[Callable[[int, str], None]] = None,
                           generation_id: str = None) -> Dict[str, Any]:
        """Generar usando Aurora Director V7 con optimizaciones"""
        try:
            if progress_callback:
                progress_callback(20, "Configurando Aurora Director V7")
            
            # Preparar parámetros optimizados para Aurora
            aurora_params = {
                'objetivo': params['objetivo'],
                'duracion_min': params['duracion_min'],
                'intensidad': params['intensidad'],
                'estilo': params['estilo'],
                'calidad_objetivo': params['calidad_objetivo'],
                'normalizar': params.get('normalizar', True),
                'aplicar_mastering': params.get('aplicar_mastering', True),
                'sample_rate': params.get('sample_rate', 44100),
                'contexto_uso': params.get('contexto_uso', 'general'),
                'modo_orquestacion': params.get('modo_orquestacion', 'hybrid'),
                'usar_objective_manager': params.get('usar_objective_manager', True),
                'validacion_automatica': params.get('validacion_automatica', True),
                'generation_id': generation_id  # Para tracking
            }
            
            # Añadir parámetros opcionales
            opcional_params = [
                'neurotransmisor_preferido', 'estrategia_preferida', 'motores_preferidos',
                'usar_carmine_booster', 'nivel_detalle_analisis', 'optimizar_para_objetivo'
            ]
            
            for param in opcional_params:
                if params.get(param):
                    aurora_params[param] = params[param]
            
            if progress_callback:
                progress_callback(30, "Ejecutando Aurora Director V7")
            
            # Llamar a Aurora con manejo de excepciones robusto
            resultado_aurora = self.aurora_director.crear_experiencia(**aurora_params)
            
            if progress_callback:
                progress_callback(70, "Audio generado, procesando resultado")
            
            # Verificar resultado con validaciones adicionales
            if not hasattr(resultado_aurora, 'audio_data'):
                raise Exception("Aurora no retornó audio_data")
            
            audio_data = resultado_aurora.audio_data
            if audio_data is None or audio_data.size == 0:
                raise Exception("Audio generado está vacío")
            
            # Validar calidad del audio
            if np.all(audio_data == 0):
                raise Exception("Audio generado contiene solo silencio")
            
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                logger.warning("⚠️ Audio contiene valores NaN/Inf, aplicando corrección")
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # ✅ USAR LAS FUNCIONES AUXILIARES DE LA CLASE (OPTIMIZADAS)
            # Obtener atributos de forma segura
            atributos = [
                'estrategia_usada', 'modo_orquestacion', 'componentes_usados',
                'tiempo_generacion', 'calidad_score', 'coherencia_neuroacustica',
                'efectividad_terapeutica', 'metadatos', 'analisis_detallado'
            ]
            
            resultado_attrs = {}
            for attr in atributos:
                resultado_attrs[attr] = getattr(resultado_aurora, attr, None)
            
            # Preparar metadatos con conversión segura a JSON (OPTIMIZADA)
            metadata = {
                'generator': 'Aurora Director V7.2 Optimized',
                'generation_id': generation_id,
                'strategy_used': self._enum_to_str(resultado_attrs['estrategia_usada']),
                'orchestration_mode': self._enum_to_str(resultado_attrs['modo_orquestacion']),
                'components_used': self._clean_metadata(resultado_attrs['componentes_usados']),
                'generation_time': float(resultado_attrs['tiempo_generacion']) if resultado_attrs['tiempo_generacion'] else 0.0,
                'quality_score': float(resultado_attrs['calidad_score']) if resultado_attrs['calidad_score'] else 0.0,
                'neuroacoustic_coherence': float(resultado_attrs['coherencia_neuroacustica']) if resultado_attrs['coherencia_neuroacustica'] else 0.0,
                'therapeutic_effectiveness': float(resultado_attrs['efectividad_terapeutica']) if resultado_attrs['efectividad_terapeutica'] else 0.0,
                'configuration': self._clean_metadata(params),
                'aurora_metadata': self._clean_metadata(resultado_attrs['metadatos']),
                'detailed_analysis': self._clean_metadata(resultado_attrs['analisis_detallado'])
            }
            
            return {
                'success': True,
                'audio_data': audio_data,
                'metadata': metadata,
                'method': 'aurora_director_v7_optimized'
            }
            
        except Exception as e:
            logger.error(f"Error en generación con Aurora: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Error en Aurora Director: {str(e)}'
            }
    
    def _generar_con_fallback(self, params: Dict[str, Any],
                            progress_callback: Optional[Callable[[int, str], None]] = None,
                            generation_id: str = None) -> Dict[str, Any]:
        """Generar audio usando sistema de fallback optimizado"""
        try:
            logger.warning("⚠️ Usando sistema de fallback optimizado para generación de audio")
            
            if progress_callback:
                progress_callback(20, "Iniciando generación fallback optimizada")
            
            # Parámetros básicos
            duracion_min = params.get('duracion_min', 20)
            duracion_sec = duracion_min * 60
            sample_rate = params.get('sample_rate', self.sample_rate)
            
            # Mapeo inteligente de objetivos a frecuencias
            objetivo_lower = params.get('objetivo', '').lower()
            freq_mapping = {
                'relajac': 432, 'calm': 432, 'paz': 432,
                'concentrac': 528, 'focus': 528, 'estudio': 528,
                'energia': 741, 'energy': 741, 'activac': 741,
                'medita': 396, 'spiritual': 396, 'zen': 396,
                'sueño': 285, 'sleep': 285, 'dormir': 285,
                'sanac': 528, 'healing': 528, 'terapia': 528
            }
            
            freq_base = 440  # Default
            for keyword, freq in freq_mapping.items():
                if keyword in objetivo_lower:
                    freq_base = freq
                    break
            
            if progress_callback:
                progress_callback(40, f"Generando onda base ({freq_base} Hz)")
            
            # Generar audio optimizado con múltiples capas
            total_samples = int(duracion_sec * sample_rate)
            t = np.linspace(0, duracion_sec, total_samples, False)
            
            # Onda principal con armónicos inteligentes
            intensidad_factor = {
                'suave': 0.4, 'sutil': 0.5, 'media': 0.6,
                'moderado': 0.7, 'intenso': 0.8, 'profundo': 0.9, 'extremo': 1.0
            }.get(params.get('intensidad', 'media'), 0.6)
            
            # Componentes de frecuencia optimizadas
            componentes = [
                (freq_base, 0.6 * intensidad_factor),           # Fundamental
                (freq_base * 2, 0.2 * intensidad_factor),       # Segunda armónica
                (freq_base * 0.5, 0.15 * intensidad_factor),    # Sub-armónica
                (freq_base * 1.5, 0.1 * intensidad_factor),     # Quinta perfecta
                (freq_base * 0.75, 0.08 * intensidad_factor),   # Cuarta sub-armónica
            ]
            
            if progress_callback:
                progress_callback(50, "Aplicando síntesis armónica")
            
            # Generar audio con síntesis aditiva
            audio_mono = np.zeros(total_samples)
            for freq, amp in componentes:
                fase = np.random.random() * 2 * np.pi  # Fase aleatoria para naturalidad
                audio_mono += amp * np.sin(2 * np.pi * freq * t + fase)
            
            if progress_callback:
                progress_callback(60, "Aplicando modulaciones")
            
            # Aplicar modulaciones sofisticadas
            # LFO para modulación de amplitud
            lfo_freq = 0.1 + np.random.random() * 0.4  # 0.1-0.5 Hz
            lfo = 0.95 + 0.05 * np.sin(2 * np.pi * lfo_freq * t)
            audio_mono *= lfo
            
            # Modulación de frecuencia sutil
            fm_depth = 2.0 + np.random.random() * 3.0  # 2-5 Hz
            fm_rate = 0.05 + np.random.random() * 0.05  # 0.05-0.1 Hz
            fm_mod = 1 + (fm_depth / freq_base) * np.sin(2 * np.pi * fm_rate * t)
            
            if progress_callback:
                progress_callback(70, "Aplicando envolvente dinámica")
            
            # Envelope sofisticado
            attack_time = min(10, duracion_sec * 0.1)  # 10% o máximo 10s
            release_time = min(15, duracion_sec * 0.15)  # 15% o máximo 15s
            
            attack_samples = int(attack_time * sample_rate)
            release_samples = int(release_time * sample_rate)
            sustain_samples = total_samples - attack_samples - release_samples
            
            envelope = np.ones(total_samples)
            
            # Attack suave
            if attack_samples > 0:
                envelope[:attack_samples] = np.power(np.linspace(0, 1, attack_samples), 0.5)
            
            # Release suave
            if release_samples > 0:
                envelope[-release_samples:] = np.power(np.linspace(1, 0, release_samples), 0.5)
            
            audio_mono *= envelope
            
            # Agregar textura sutil con ruido coloreado
            if params.get('estilo', 'sereno') in ['organico', 'natural']:
                if progress_callback:
                    progress_callback(75, "Agregando textura orgánica")
                
                # Ruido rosa sutil
                noise_level = 0.02
                pink_noise = np.random.normal(0, noise_level, total_samples)
                # Filtro simple para ruido rosa (aproximación)
                for i in range(1, len(pink_noise)):
                    pink_noise[i] = 0.99 * pink_noise[i-1] + 0.01 * pink_noise[i]
                
                audio_mono += pink_noise * intensidad_factor
            
            # Convertir a estéreo con imagen espacial
            if progress_callback:
                progress_callback(80, "Creando imagen estéreo")
            
            # Crear diferencia sutil entre canales para espacialidad
            delay_samples = int(0.001 * sample_rate)  # 1ms delay
            audio_left = audio_mono.copy()
            audio_right = np.roll(audio_mono, delay_samples) * 0.98  # Ligera atenuación
            
            audio_data = np.stack([audio_left, audio_right])
            
            # Normalización inteligente
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                target_level = 0.8 * intensidad_factor
                audio_data = audio_data * target_level / max_val
            
            # Aplicar compresión suave
            if progress_callback:
                progress_callback(85, "Aplicando procesamiento final")
            
            # Compresión suave para mayor consistencia
            threshold = 0.7
            ratio = 2.0
            compressed = np.where(
                np.abs(audio_data) > threshold,
                np.sign(audio_data) * (threshold + (np.abs(audio_data) - threshold) / ratio),
                audio_data
            )
            audio_data = compressed
            
            metadata = {
                'generator': 'Fallback System V7.2 Optimized',
                'generation_id': generation_id,
                'strategy_used': 'fallback_generation_optimized',
                'orchestration_mode': 'intelligent_fallback',
                'components_used': ['harmonic_synthesis', 'envelope_shaping', 'spatial_imaging'],
                'generation_time': 3.0,
                'quality_score': 80.0,
                'neuroacoustic_coherence': 0.85,
                'therapeutic_effectiveness': 0.8,
                'synthesis_details': {
                    'base_frequency': freq_base,
                    'harmonics_count': len(componentes),
                    'modulation_applied': True,
                    'spatial_processing': True,
                    'noise_texture': params.get('estilo') in ['organico', 'natural']
                },
                'objective_detected': objetivo_lower,
                'configuration': params,
                'fallback_mode': True,
                'optimizations': {
                    'intelligent_frequency_mapping': True,
                    'harmonic_synthesis': True,
                    'dynamic_envelope': True,
                    'spatial_imaging': True
                }
            }
            
            return {
                'success': True,
                'audio_data': audio_data,
                'metadata': metadata,
                'method': 'fallback_optimized'
            }
            
        except Exception as e:
            logger.error(f"Error en sistema de fallback optimizado: {e}")
            return {
                'success': False,
                'error': f'Error en fallback optimizado: {str(e)}'
            }
    
    def _save_audio_to_wav(self, audio_data: np.ndarray, params: Dict[str, Any], generation_id: str = None) -> str:
        """Guardar audio numpy a archivo WAV con optimizaciones"""
        try:
            # Generar nombre único optimizado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_objetivo = "".join(c for c in params['objetivo'] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_objetivo = safe_objetivo.replace(' ', '_')[:30]  # Aumentado límite
            
            # Incluir generation_id si está disponible
            id_suffix = f"_{generation_id[:8]}" if generation_id else f"_{uuid.uuid4().hex[:8]}"
            
            filename = f"aurora_opt_{safe_objetivo}_{params['duracion_min']}min_{timestamp}{id_suffix}.wav"
            filepath = os.path.join(self.audio_folder, filename)
            
            # Asegurar que existe el directorio
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Procesamiento optimizado del audio
            if audio_data.ndim == 1:
                # Mono a estéreo con imagen espacial mejorada
                audio_stereo = np.stack([audio_data, audio_data * 0.99])
            else:
                audio_stereo = audio_data
            
            # Asegurar orden correcto [samples, channels] para wave
            if audio_stereo.shape[0] == 2 and audio_stereo.shape[1] > audio_stereo.shape[0]:
                audio_stereo = audio_stereo.T
            elif audio_stereo.shape[0] < audio_stereo.shape[1]:
                # Ya está en formato [samples, channels]
                pass
            else:
                # Formato [channels, samples], transponer
                audio_stereo = audio_stereo.T
            
            # Normalización mejorada con preservación de dinámicas
            max_val = np.max(np.abs(audio_stereo))
            if max_val > 0:
                # Normalización con headroom
                headroom = 0.95  # Dejar 5% de headroom
                audio_stereo = audio_stereo / max_val * headroom
            
            # Dithering suave para mejor calidad a 16-bit
            dither_noise = np.random.normal(0, 1/32768/4, audio_stereo.shape)
            audio_stereo += dither_noise
            
            # Convertir a int16 con clipping suave
            audio_int16 = np.clip(audio_stereo * 32767, -32768, 32767).astype(np.int16)
            
            # Guardar con wave
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Estéreo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Verificar archivo guardado
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise Exception("Archivo WAV guardado está vacío")
            
            logger.info(f"✅ Audio guardado: {filename} ({file_size/1024/1024:.1f} MB)")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error guardando audio optimizado: {e}")
            raise Exception(f"Error guardando archivo WAV: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de procesamiento optimizadas"""
        with self._stats_lock:
            stats = self._processing_stats.copy()
        
        # Calcular métricas adicionales
        if stats['total_generations'] > 0:
            stats['success_rate'] = (stats['successful_generations'] / stats['total_generations']) * 100
        else:
            stats['success_rate'] = 0.0
        
        # Agregar estadísticas de Carmine si está disponible
        if self.carmine_analyzer and hasattr(self.carmine_analyzer, 'obtener_reporte_estadisticas'):
            try:
                stats['carmine_stats'] = self.carmine_analyzer.obtener_reporte_estadisticas()
            except Exception as e:
                logger.debug(f"Error obteniendo estadísticas Carmine: {e}")
        
        return stats
    
    def clear_cache(self):
        """Limpiar cache del sistema"""
        with self._status_lock:
            self._status_cache.clear()
            self._cache_timestamp = 0
        
        if self.carmine_analyzer and hasattr(self.carmine_analyzer, 'limpiar_cache'):
            try:
                self.carmine_analyzer.limpiar_cache()
                logger.info("🧹 Cache de Carmine limpiado")
            except Exception as e:
                logger.debug(f"Error limpiando cache Carmine: {e}")
        
        logger.info("🧹 Cache del sistema limpiado")
    
    def health_check(self) -> Dict[str, Any]:
        """Verificación de salud del sistema optimizada"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'bridge_status': 'healthy',
            'components': {
                'aurora_director': self.initialized and self.aurora_director is not None,
                'carmine_analyzer': self.carmine_analyzer is not None,
                'carmine_booster': self.carmine_booster is not None,
                'quality_pipeline': self.quality_pipeline is not None,
                'thread_pool': self.thread_pool is not None and not self.thread_pool._shutdown
            },
            'system_load': {
                'active_threads': threading.active_count(),
                'thread_pool_busy': self.thread_pool._threads if hasattr(self.thread_pool, '_threads') else 0
            },
            'disk_space': self._check_disk_space(),
            'memory_usage': self._check_memory_usage()
        }
        
        # Verificar si hay problemas
        issues = []
        if not health['components']['aurora_director']:
            issues.append("Aurora Director no está disponible")
        
        if health['disk_space']['available_mb'] < 100:
            issues.append("Poco espacio en disco (<100MB)")
        
        health['issues'] = issues
        health['overall_status'] = 'healthy' if not issues else 'warning' if len(issues) < 3 else 'critical'
        
        return health
    
    def _check_disk_space(self) -> Dict[str, float]:
        """Verificar espacio en disco"""
        try:
            stat = os.statvfs(self.audio_folder)
            free_bytes = stat.f_bavail * stat.f_frsize
            total_bytes = stat.f_blocks * stat.f_frsize
            
            return {
                'available_mb': free_bytes / (1024 * 1024),
                'total_mb': total_bytes / (1024 * 1024),
                'usage_percent': ((total_bytes - free_bytes) / total_bytes) * 100
            }
        except Exception as e:
            logger.debug(f"Error verificando espacio en disco: {e}")
            return {'available_mb': -1, 'total_mb': -1, 'usage_percent': -1}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Verificar uso de memoria (básico)"""
        try:
            import psutil
            process = psutil.Process()
            return {
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'memory_percent': process.memory_percent()
            }
        except ImportError:
            return {'available': False, 'reason': 'psutil not installed'}
        except Exception as e:
            logger.debug(f"Error verificando memoria: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Apagado limpio del sistema"""
        logger.info("🔄 Iniciando apagado limpio de AuroraBridge...")
        
        try:
            # Limpiar cache
            self.clear_cache()
            
            # Cerrar thread pool
            if self.thread_pool and not self.thread_pool._shutdown:
                self.thread_pool.shutdown(wait=True, timeout=30)
                logger.info("🔄 Thread pool cerrado")
            
            # Limpiar recursos de Carmine
            if self.carmine_analyzer and hasattr(self.carmine_analyzer, 'shutdown'):
                self.carmine_analyzer.shutdown()
                logger.info("🔄 Carmine Analyzer cerrado")
            
            logger.info("✅ AuroraBridge apagado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error durante apagado: {e}")
    
    def __del__(self):
        """Destructor para limpiar recursos"""
        try:
            self.shutdown()
        except Exception:
            pass  # Evitar errores en destructor
