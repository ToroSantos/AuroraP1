#!/usr/bin/env python3
"""
Aurora Bridge V7 - Integración Completa con Sistema Aurora
Puente entre Flask Backend y Aurora V7 Real

OPTIMIZADO: Error de serialización JSON corregido + Carmine integrado + Todas las funcionalidades
✅ CORREGIDO: Error de firma de método crear_experiencia() - Compatibilidad con ConfiguracionAuroraUnificada
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
    ✅ CORREGIDO: Compatibilidad de firmas de métodos con AuroraDirectorV7Refactored
    
    Funcionalidades:
    - Inicialización robusta con arranque controlado
    - Generación de audio con Aurora Director V7
    - Sistema de fallback inteligente
    - Integración con Carmine Analyzer y Quality Pipeline
    - Validación y metadatos completos
    """
    
    def __init__(self, audio_folder: str = "generated_audio", sample_rate: int = 44100):
        """Inicializar Aurora Bridge optimizado"""
        logger.info("🌟 Inicializando Aurora V7 con arranque controlado...")
        
        # Configuración básica
        self.audio_folder = Path(audio_folder)
        self.audio_folder.mkdir(exist_ok=True)
        self.sample_rate = sample_rate
        self.initialized = False
        self.aurora_director = None
        self.system_status = {}
        
        # Componentes opcionales
        self.carmine_analyzer = None
        self.carmine_booster = None
        self.quality_pipeline = None
        
        # Threading y estadísticas optimizadas
        self._stats_lock = threading.RLock()
        self._processing_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_generation_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # Inicializar sistemas
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Inicializar todos los sistemas de forma ordenada"""
        self._initialize_aurora_system()
        self._initialize_carmine_system()
        self._initialize_quality_pipeline()
        self._verify_system_components()
    
    def _initialize_aurora_system(self):
        """Inicializar Aurora Director V7 con arranque controlado"""
        try:
            # Usar Aurora Startup Fix si está disponible
            try:
                from aurora_startup_fix import init_aurora_system_controlled
                logger.info("🚀 Usando arranque controlado Aurora V7")
                
                result = init_aurora_system_controlled()
                if result and hasattr(result, 'success') and result.success:
                    director = result.director
                    logger.info(f"✅ Aurora Director obtenido del arranque controlado: {type(director).__name__}")
                    
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
                self.quality_pipeline = AuroraQualityPipeline()
                logger.info("✅ Aurora Quality Pipeline inicializado")
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando Quality Pipeline: {e}")
    
    def _initialize_fallback_system(self):
        """Configurar sistema fallback"""
        self.initialized = False
        self.aurora_director = None
        logger.info("🔄 Sistema configurado en modo fallback")
    
    def _verify_system_components(self):
        """Verificar estado de todos los componentes"""
        try:
            if self.initialized and self.aurora_director:
                self.system_status = self.aurora_director.obtener_estado_completo()
                logger.info("📊 Estado del sistema verificado")
            else:
                self.system_status = self._get_fallback_status()
        except Exception as e:
            logger.warning(f"⚠️ Error verificando componentes: {e}")
            self.system_status = self._get_error_status(str(e))
    
    def get_complete_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema optimizado"""
        try:
            if not self.system_status:
                self._verify_system_components()
            
            # Agregar estadísticas del bridge
            bridge_stats = {
                'bridge_initialized': self.initialized,
                'aurora_director_available': self.aurora_director is not None,
                'carmine_available': self.carmine_analyzer is not None,
                'quality_pipeline_available': self.quality_pipeline is not None,
                'audio_folder': str(self.audio_folder),
                'sample_rate': self.sample_rate,
                'processing_stats': self._processing_stats.copy()
            }
            
            if isinstance(self.system_status, dict):
                self.system_status['bridge'] = bridge_stats
                return self.system_status
            else:
                return {
                    'bridge': bridge_stats,
                    'status': 'basic',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado del sistema: {e}")
            return self._get_error_status(str(e))
    
    def _get_fallback_status(self) -> Dict[str, Any]:
        """Estado básico en modo fallback"""
        return {
            'summary': {
                'status': 'fallback',
                'status_text': 'Sistema en modo fallback - funcionalidad limitada',
                'motores_activos': 0,
                'total_componentes': 0,
                'calidad_promedio': 60.0,
                'tiempo_promedio': 3.0,
                'componentes_principales': {
                    'aurora_director': False,
                    'carmine_analyzer': CARMINE_AVAILABLE
                }
            },
            'bridge_info': {
                'initialized': False,
                'fallback_mode': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _get_basic_status(self) -> Dict[str, Any]:
        """Estado básico con Aurora funcional"""
        return {
            'summary': {
                'status': 'basic',
                'status_text': 'Sistema Aurora básico operativo',
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
        sample_rate = params.get('sample_rate', self.sample_rate)
        try:
            sample_rate = int(sample_rate)
            if sample_rate not in [22050, 44100, 48000, 96000]:
                warnings.append("Sample rate no estándar. Recomendado: 44100 Hz")
        except (ValueError, TypeError):
            warnings.append("Sample rate inválido, usando 44100 Hz por defecto")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'params_processed': {
                'objetivo': objetivo,
                'duracion_min': duracion_min,
                'intensidad': intensidad,
                'estilo': estilo,
                'calidad_objetivo': calidad,
                'sample_rate': sample_rate
            }
        }
    
    def estimate_generation_time(self, params: Dict[str, Any]) -> int:
        """Estimar tiempo de generación optimizado (en segundos)"""
        duracion_min = params.get('duracion_min', 20)
        calidad = params.get('calidad_objetivo', 'alta')
        
        # Factor base por duración
        base_time = max(3, min(duracion_min * 0.15, 60))  # 0.15s por minuto, máximo 60s
        
        # Factor por duración (no lineal)
        if duracion_min <= 10:
            duracion_factor = 1.0
        elif duracion_min <= 30:
            duracion_factor = 1.2
        elif duracion_min <= 60:
            duracion_factor = 1.5
        else:
            duracion_factor = 2.0
        
        # Factor por calidad
        calidad_factors = {
            'basica': 0.7,
            'media': 1.0,
            'alta': 1.3,
            'maxima': 1.8,
            'premium': 2.5
        }
        calidad_factor = calidad_factors.get(calidad, 1.0)
        
        # Factor del sistema
        if self.initialized and self.aurora_director:
            system_factor = 1.0  # Sistema optimizado
            # Bonus si Carmine está disponible
            if self.carmine_booster:
                system_factor *= 0.9  # 10% más rápido con Carmine
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
                    logger.info("✅ Quality Pipeline aplicado")
                except Exception as e:
                    logger.warning(f"⚠️ Error en Quality Pipeline: {e}")
            
            if progress_callback:
                progress_callback(90, "Guardando archivo de audio")
            
            # Guardar archivo WAV
            filename = f"aurora_{generation_id[:8]}_{int(time.time())}.wav"
            filepath = self.audio_folder / filename
            
            self._save_wav_file(audio_data, filepath)
            
            # Actualizar estadísticas de éxito
            generation_time = time.time() - start_time
            with self._stats_lock:
                stats = self._processing_stats
                stats['successful_generations'] += 1
                stats['total_processing_time'] += generation_time
                total_gens = stats['successful_generations']
                stats['average_generation_time'] = stats['total_processing_time'] / total_gens
            
            if progress_callback:
                progress_callback(100, "Generación completada exitosamente")
            
            logger.info(f"✅ Experiencia creada exitosamente: {filename} ({generation_time:.1f}s)")
            
            # Preparar resultado final
            resultado_final = resultado.copy()
            resultado_final.update({
                'filename': filename,
                'filepath': str(filepath),
                'generation_time': generation_time,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0
            })
            
            return resultado_final
            
        except Exception as e:
            # Actualizar estadísticas de error
            with self._stats_lock:
                self._processing_stats['failed_generations'] += 1
            
            error_msg = f"Error en generación completa: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': error_msg,
                'generation_id': generation_id,
                'generation_time': time.time() - start_time
            }
    
    def _generar_con_aurora(self, params: Dict[str, Any], 
                           progress_callback: Optional[Callable[[int, str], None]] = None,
                           generation_id: str = None) -> Dict[str, Any]:
        """
        ✅ MÉTODO CORREGIDO: Generar audio usando Aurora Director V7 
        SOLUCIÓN: Usar ConfiguracionAuroraUnificada en lugar de parámetros individuales
        """
        try:
            if not generation_id:
                generation_id = str(uuid.uuid4())
                
            logger.info(f"🎵 Iniciando generación con Aurora Director V7: {generation_id}")
            
            if progress_callback:
                progress_callback(20, "Configurando Aurora Director V7")
            
            # ===== SECCIÓN CORREGIDA - CRÍTICA =====
            # ✅ CREAR CONFIGURACIÓN UNIFICADA CORRECTAMENTE
            try:
                # Validar y extraer parámetros básicos requeridos
                objetivo = params.get('objetivo', 'bienestar_general')
                duracion_min = params.get('duracion_min', 20)
                intensidad = params.get('intensidad', 'media')
                
                # Convertir duración a segundos para Aurora Director
                duracion_segundos = float(duracion_min) * 60.0
                
                # Mapear intensidad a valor numérico si es necesario
                intensidad_valor = intensidad
                if isinstance(intensidad, str):
                    mapeo_intensidad = {
                        'suave': 0.3, 'sutil': 0.4, 'media': 0.5, 
                        'moderado': 0.6, 'intenso': 0.7, 'profundo': 0.8, 'extremo': 0.9
                    }
                    intensidad_valor = mapeo_intensidad.get(intensidad, 0.5)
                
                # ✅ CREAR CONFIGURACIÓN UNIFICADA CORRECTAMENTE
                configuracion_aurora = ConfiguracionAuroraUnificada(
                    objetivo=objetivo,
                    duracion=duracion_segundos,
                    intensidad=intensidad_valor,
                    estilo=params.get('estilo', 'sereno'),
                    calidad=params.get('calidad_objetivo', 'alta'),
                    sample_rate=params.get('sample_rate', self.sample_rate)
                )
                
                # Añadir parámetros opcionales como atributos adicionales si los soporta
                parametros_opcionales = [
                    'neurotransmisor_preferido', 'estrategia_preferida', 'motores_preferidos',
                    'usar_carmine_booster', 'nivel_detalle_analisis', 'optimizar_para_objetivo',
                    'aplicar_correcciones_automatica', 'generation_id'
                ]
                
                for param_key in parametros_opcionales:
                    if params.get(param_key) is not None:
                        # Si ConfiguracionAuroraUnificada soporta el parámetro, agregarlo
                        if hasattr(configuracion_aurora, param_key):
                            setattr(configuracion_aurora, param_key, params[param_key])
                        else:
                            # Almacenar en metadatos para compatibilidad futura
                            if not hasattr(configuracion_aurora, '_metadatos_extra'):
                                configuracion_aurora._metadatos_extra = {}
                            configuracion_aurora._metadatos_extra[param_key] = params[param_key]
                
                # Agregar generation_id para tracking
                if not hasattr(configuracion_aurora, 'generation_id'):
                    if hasattr(configuracion_aurora, '_metadatos_extra'):
                        configuracion_aurora._metadatos_extra['generation_id'] = generation_id
                    else:
                        configuracion_aurora._metadatos_extra = {'generation_id': generation_id}
                
                logger.info(f"✅ Configuración Aurora creada: objetivo='{objetivo}', duración={duracion_segundos:.1f}s, intensidad={intensidad_valor}")
                
            except Exception as e:
                logger.error(f"❌ Error creando ConfiguracionAuroraUnificada: {e}")
                raise Exception(f"Error en configuración Aurora: {e}")
            
            if progress_callback:
                progress_callback(30, "Ejecutando Aurora Director V7")
            
            # ✅ LLAMADA CORREGIDA - PASAR CONFIGURACIÓN EN LUGAR DE PARÁMETROS INDIVIDUALES
            try:
                resultado_aurora = self.aurora_director.crear_experiencia(configuracion_aurora)
                logger.info("✅ Aurora Director ejecutado exitosamente")
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    logger.error(f"❌ Error de compatibilidad en Aurora Director: {e}")
                    logger.error("💡 Sugerencia: Verificar que AuroraDirectorV7Refactored.crear_experiencia() acepta ConfiguracionAuroraUnificada")
                raise Exception(f"Error de compatibilidad Aurora Director: {e}")
            except Exception as e:
                logger.error(f"❌ Error ejecutando Aurora Director: {e}")
                raise Exception(f"Error en Aurora Director: {e}")
            
            # ===== FIN SECCIÓN CORREGIDA =====
            
            if progress_callback:
                progress_callback(70, "Audio generado, procesando resultado")
            
            # Verificar resultado con validaciones adicionales
            if not hasattr(resultado_aurora, 'audio_data'):
                raise Exception("Aurora no retornó audio_data")
            
            audio_data = resultado_aurora.audio_data
            if audio_data is None:
                raise Exception("Audio generado está vacío")
            elif isinstance(audio_data, dict):
                # Aurora Director devolvió un diccionario con estructura
                if not audio_data.get('success', True):
                    raise Exception(f"Error en generación Aurora: {audio_data.get('error', 'Error desconocido')}")
                actual_audio = audio_data.get('audio_data')
                if actual_audio is None:
                    raise Exception("Audio generado no contiene datos de audio")
                audio_data = actual_audio  # Usar el array real
            elif hasattr(audio_data, 'audio_data'):
                # Es un objeto ResultadoAuroraIntegrado
                audio_data = audio_data.audio_data

            # Ahora verificar el array numpy real
            if audio_data is None or (hasattr(audio_data, 'size') and audio_data.size == 0):
                raise Exception("Audio generado está vacío")

            # Validar calidad del audio
            if np.all(audio_data == 0):
                raise Exception("Audio generado contiene solo silencio")
            
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                logger.warning("⚠️ Audio contiene valores NaN/Inf, aplicando corrección")
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # ✅ USAR LAS FUNCIONES AUXILIARES DE LA CLASE (OPTIMIZADAS)|
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
                'components_used': resultado_attrs['componentes_usados'] or [],
                'generation_time': resultado_attrs['tiempo_generacion'] or 0.0,
                'quality_score': resultado_attrs['calidad_score'] or 0.0,
                'neuroacoustic_coherence': resultado_attrs['coherencia_neuroacustica'] or 0.0,
                'therapeutic_effectiveness': resultado_attrs['efectividad_terapeutica'] or 0.0,
                'sample_rate': self.sample_rate,
                'audio_length': len(audio_data) / self.sample_rate,
                'aurora_version': getattr(resultado_aurora, 'version', 'V7'),
                'timestamp': datetime.now().isoformat(),
                'configuration_used': {
                    'objetivo': objetivo,
                    'duracion_min': duracion_min, 
                    'intensidad': intensidad,
                    'estilo': params.get('estilo', 'sereno'),
                    'calidad_objetivo': params.get('calidad_objetivo', 'alta')
                }
            }
            
            # Agregar análisis detallado si está disponible
            if resultado_attrs['analisis_detallado']:
                try:
                    metadata['detailed_analysis'] = self._clean_metadata(resultado_attrs['analisis_detallado'])
                except Exception as e:
                    logger.warning(f"⚠️ Error procesando análisis detallado: {e}")
                    metadata['detailed_analysis'] = {"error": "No se pudo procesar"}
            
            # Agregar metadatos adicionales del resultado si están disponibles
            if resultado_attrs['metadatos']:
                try:
                    metadata['aurora_metadata'] = self._clean_metadata(resultado_attrs['metadatos'])
                except Exception as e:
                    logger.warning(f"⚠️ Error procesando metadatos Aurora: {e}")
            
            logger.info(f"✅ Generación completada exitosamente: {len(audio_data)} samples, {len(audio_data)/self.sample_rate:.1f}s")
            
            return {
                'success': True,
                'audio_data': audio_data,
                'metadata': metadata,
                'generation_id': generation_id,
                'generator': 'Aurora Director V7.2 Optimized'
            }
            
        except Exception as e:
            error_msg = f"Error en generación con Aurora: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback completo: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': error_msg,
                'metadata': {
                    'generator': 'Aurora Director V7.2 (Failed)',
                    'generation_id': generation_id,
                    'timestamp': datetime.now().isoformat(),
                    'error_details': str(e)
                },
                'generation_id': generation_id
            }
    
    def _generar_con_fallback(self, params: Dict[str, Any], 
                             progress_callback: Optional[Callable[[int, str], None]] = None,
                             generation_id: str = None) -> Dict[str, Any]:
        """Generar audio usando sistema fallback (senoidal simple)"""
        try:
            if not generation_id:
                generation_id = str(uuid.uuid4())
                
            logger.info(f"🔄 Generando con sistema fallback: {generation_id}")
            
            if progress_callback:
                progress_callback(20, "Configurando generación fallback")
            
            # Extraer parámetros
            duracion_min = params.get('duracion_min', 20)
            intensidad = params.get('intensidad', 'media')
            
            # Calcular duración en samples
            duracion_sec = duracion_min * 60
            samples = int(duracion_sec * self.sample_rate)
            
            if progress_callback:
                progress_callback(50, "Generando audio senoidal")
            
            # Generar audio senoidal simple
            t = np.linspace(0, duracion_sec, samples, False)
            
            # Frecuencias base
            freq_base = 40  # Hz - frecuencia base binaural
            freq_diferencial = 6  # Hz - diferencia binaural
            
            # Mapear intensidad
            intensidad_map = {'suave': 0.3, 'sutil': 0.4, 'media': 0.5, 'moderado': 0.6, 'intenso': 0.7, 'profundo': 0.8, 'extremo': 0.9}
            amplitud = intensidad_map.get(intensidad, 0.5)
            
            # Generar canales estéreo
            canal_izq = amplitud * np.sin(2 * np.pi * freq_base * t)
            canal_der = amplitud * np.sin(2 * np.pi * (freq_base + freq_diferencial) * t)
            
            # Combinar canales
            audio_data = np.column_stack((canal_izq, canal_der))
            
            if progress_callback:
                progress_callback(80, "Aplicando envolvente y normalización")
            
            # Aplicar envolvente suave
            fade_samples = int(0.1 * self.sample_rate)  # 0.1 segundo de fade
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            audio_data[:fade_samples, :] *= fade_in[:, np.newaxis]
            audio_data[-fade_samples:, :] *= fade_out[:, np.newaxis]
            
            # Normalizar
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.8  # Normalizar a 80%
            
            # Convertir a formato mono para compatibilidad
            audio_data = np.mean(audio_data, axis=1)
            
            if progress_callback:
                progress_callback(100, "Audio fallback generado")
            
            metadata = {
                'generator': 'Aurora Bridge Fallback System',
                'generation_id': generation_id,
                'method': 'senoidal_binaural',
                'frequency_base': freq_base,
                'frequency_differential': freq_diferencial,
                'amplitude': amplitud,
                'sample_rate': self.sample_rate,
                'audio_length': len(audio_data) / self.sample_rate,
                'timestamp': datetime.now().isoformat(),
                'configuration_used': {
                    'duracion_min': duracion_min,
                    'intensidad': intensidad
                }
            }
            
            logger.info(f"✅ Audio fallback generado: {len(audio_data)} samples, {len(audio_data)/self.sample_rate:.1f}s")
            
            return {
                'success': True,
                'audio_data': audio_data,
                'metadata': metadata,
                'generation_id': generation_id,
                'generator': 'Aurora Bridge Fallback'
            }
            
        except Exception as e:
            error_msg = f"Error en generación fallback: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'generation_id': generation_id
            }
    
    def _apply_carmine_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Aplicar optimización predictiva Carmine"""
        if not self.carmine_booster:
            return params
        
        try:
            # Analizar parámetros y sugerir optimizaciones
            optimized_params = params.copy()
            
            # Ejemplo de optimización basada en objetivo
            objetivo = params.get('objetivo', '').lower()
            if 'enfoque' in objetivo or 'concentración' in objetivo:
                optimized_params['intensidad'] = 'moderado'
                optimized_params['estilo'] = 'crystalline'
            elif 'relajación' in objetivo or 'calma' in objetivo:
                optimized_params['intensidad'] = 'suave'
                optimized_params['estilo'] = 'sereno'
            
            return optimized_params
            
        except Exception as e:
            logger.warning(f"Error en optimización Carmine: {e}")
            return params
    
    def _apply_quality_pipeline(self, audio_data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Aplicar pipeline de calidad al audio"""
        if not self.quality_pipeline:
            return audio_data
        
        try:
            # Aplicar pipeline de calidad
            processed_audio = self.quality_pipeline.process_audio(audio_data, self.sample_rate)
            return processed_audio
        except Exception as e:
            logger.warning(f"Error en quality pipeline: {e}")
            return audio_data
    
    def _save_wav_file(self, audio_data: np.ndarray, filepath: Path):
        """Guardar audio como archivo WAV"""
        try:
            # Normalizar y convertir a int16
            if audio_data.dtype != np.int16:
                # Normalizar a rango [-1, 1]
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                # Convertir a int16
                audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
                
            logger.info(f"✅ Archivo WAV guardado: {filepath.name}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando WAV: {e}")
            raise
    
    def _enum_to_str(self, enum_value) -> str:
        """Convertir enum a string de forma segura"""
        if enum_value is None:
            return "unknown"
        if hasattr(enum_value, 'value'):
            return str(enum_value.value)
        elif hasattr(enum_value, 'name'):
            return str(enum_value.name)
        else:
            return str(enum_value)
    
    def _clean_metadata(self, metadata: Any) -> Any:
        """Limpiar metadatos para serialización JSON"""
        if metadata is None:
            return None
        
        if isinstance(metadata, dict):
            cleaned = {}
            for key, value in metadata.items():
                cleaned[str(key)] = self._clean_metadata(value)
            return cleaned
        elif isinstance(metadata, list):
            return [self._clean_metadata(item) for item in metadata]
        elif hasattr(metadata, '__dict__'):
            return self._clean_metadata(metadata.__dict__)
        elif hasattr(metadata, 'value'):
            return metadata.value
        elif hasattr(metadata, 'name'):
            return metadata.name
        else:
            try:
                json.dumps(metadata)  # Test if serializable
                return metadata
            except (TypeError, ValueError):
                return str(metadata)

# Función de conveniencia para crear instancia global
def create_aurora_bridge(**kwargs) -> AuroraBridge:
    """Crear instancia de Aurora Bridge"""
    return AuroraBridge(**kwargs)

# Instancia global por defecto
aurora_bridge_instance = None

def get_aurora_bridge() -> AuroraBridge:
    """Obtener instancia global de Aurora Bridge"""
    global aurora_bridge_instance
    if aurora_bridge_instance is None:
        aurora_bridge_instance = create_aurora_bridge()
    return aurora_bridge_instance
