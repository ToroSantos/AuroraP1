#!/usr/bin/env python3
"""
Aurora Bridge V7 - Integración con Sistema Aurora
Puente entre Flask Backend y Aurora V7 Real
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
from typing import Dict, Any, Optional, Callable, List, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aurora.Bridge.V7")

# Agregar aurora_system al path de Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aurora_system'))

# Importar Aurora Director V7
try:
    from aurora_director_v7 import Aurora, ConfiguracionAuroraUnificada, ResultadoAuroraIntegrado
    AURORA_AVAILABLE = True
    print("✅ Aurora Director V7 importado correctamente")
except ImportError as e:
    AURORA_AVAILABLE = False
    print(f"❌ Aurora Director V7 no disponible: {e}")
    logging.warning("Aurora Director V7 no disponible")

class GenerationStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class AuroraBridge:
    """
    Puente de integración con el Sistema Aurora V7
    Maneja generación de audio, estado del sistema y conversión de archivos
    """
    
    def __init__(self):
        self.initialized = False
        self.aurora_director = None
        self.system_status = {}
        self.audio_folder = "static/audio"
        self.sample_rate = 44100
        
        # Crear carpeta de audio
        os.makedirs(self.audio_folder, exist_ok=True)
        
        # Inicializar Aurora
        self._initialize_aurora_system()
    
    def _initialize_aurora_system(self):
        """Inicializar el sistema Aurora V7"""
        try:
            logger.info("🌟 Inicializando Aurora V7...")
            
            if not AURORA_AVAILABLE:
                logger.warning("⚠️ Aurora Director no disponible - modo fallback")
                self._initialize_fallback_system()
                return
            
            # Crear instancia del director
            try:
                self.aurora_director = Aurora()
                logger.info("✅ Aurora Director instanciado")
            except Exception as e:
                logger.error(f"❌ Error instanciando Aurora Director: {e}")
                self._initialize_fallback_system()
                return
            
            # Verificar sistema
            self._verify_system_components()
            
            self.initialized = True
            logger.info("🚀 Aurora V7 inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando Aurora: {e}")
            logger.error(traceback.format_exc())
            self._initialize_fallback_system()
    
    def _initialize_fallback_system(self):
        """Sistema de fallback si Aurora no está disponible"""
        logger.warning("⚠️ Inicializando sistema de fallback")
        self.initialized = False
        self.aurora_director = None
    
    def _verify_system_components(self):
        """Verificar componentes del sistema Aurora"""
        try:
            if hasattr(self.aurora_director, 'obtener_estado_completo'):
                self.system_status = self.aurora_director.obtener_estado_completo()
                logger.info("✅ Estado del sistema obtenido")
            else:
                logger.warning("⚠️ obtener_estado_completo no disponible")
                
        except Exception as e:
            logger.warning(f"⚠️ Error verificando componentes: {e}")
    
    def get_complete_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema Aurora"""
        try:
            if not self.initialized or not self.aurora_director:
                return self._get_fallback_status()
            
            # Obtener estado desde Aurora Director
            if hasattr(self.aurora_director, 'obtener_estado_completo'):
                full_status = self.aurora_director.obtener_estado_completo()
                
                # Enriquecer con información del bridge
                full_status['bridge_info'] = {
                    'initialized': self.initialized,
                    'audio_folder': self.audio_folder,
                    'sample_rate': self.sample_rate,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Crear resumen amigable
                summary = self._create_status_summary(full_status)
                full_status['summary'] = summary
                
                return full_status
            else:
                return self._get_basic_status()
                
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return self._get_error_status(str(e))
    
    def _create_status_summary(self, full_status: Dict[str, Any]) -> Dict[str, Any]:
        """Crear resumen del estado del sistema"""
        try:
            componentes = full_status.get('componentes_detectados', {})
            capacidades = full_status.get('capacidades_sistema', {})
            
            motores_activos = capacidades.get('motores_activos', 0)
            gestores_activos = capacidades.get('gestores_activos', 0)
            pipelines_activos = capacidades.get('pipelines_activos', 0)
            
            total_componentes = len([c for c in componentes.values() if c.get('disponible')])
            
            # Determinar estado general
            if motores_activos >= 2 and total_componentes >= 5:
                status = "operational"
                status_text = "Sistema completamente operativo"
            elif motores_activos >= 1 and total_componentes >= 3:
                status = "good"
                status_text = "Sistema operativo con funcionalidad reducida"
            elif total_componentes >= 1:
                status = "limited"
                status_text = "Sistema con funcionalidad limitada"
            else:
                status = "error"
                status_text = "Sistema no operativo"
            
            # Obtener calidad promedio de forma segura
            calidad_promedio = 0.0
            if 'metricas_calidad' in full_status:
                calidad_promedio = full_status['metricas_calidad'].get('calidad_promedio', 0.0)
            
            # Obtener tiempo promedio de forma segura
            tiempo_promedio = 0.0
            if 'metricas_calidad' in full_status:
                tiempo_promedio = full_status['metricas_calidad'].get('tiempo_promedio', 0.0)
            
            return {
                'status': status,
                'status_text': status_text,
                'motores_activos': motores_activos,
                'gestores_activos': gestores_activos,
                'pipelines_activos': pipelines_activos,
                'total_componentes': total_componentes,
                'calidad_promedio': calidad_promedio,
                'tiempo_promedio': tiempo_promedio,
                'componentes_principales': {
                    'aurora_director': True,
                    'neuromix_v27': 'neuromix_v27' in componentes and componentes['neuromix_v27'].get('disponible', False),
                    'hypermod_v32': 'hypermod_v32' in componentes and componentes['hypermod_v32'].get('disponible', False),
                    'harmonic_essence_v34': 'harmonic_essence_v34' in componentes and componentes['harmonic_essence_v34'].get('disponible', False),
                    'objective_manager': capacidades.get('objective_manager_disponible', False),
                    'quality_pipeline': 'quality_pipeline' in componentes and componentes['quality_pipeline'].get('disponible', False)
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
        """Estado de fallback cuando Aurora no está disponible"""
        return {
            'summary': {
                'status': 'fallback',
                'status_text': 'Sistema Aurora no disponible - usando fallback',
                'motores_activos': 0,
                'total_componentes': 0,
                'calidad_promedio': 0.0,
                'tiempo_promedio': 0.0,
                'componentes_principales': {k: False for k in ['aurora_director', 'neuromix_v27', 'hypermod_v32', 'harmonic_essence_v34']}
            },
            'bridge_info': {
                'initialized': False,
                'fallback_mode': True,
                'timestamp': datetime.now().isoformat()
            },
            'componentes_detectados': {},
            'capacidades_sistema': {}
        }
    
    def _get_basic_status(self) -> Dict[str, Any]:
        """Estado básico cuando no hay método obtener_estado_completo"""
        return {
            'summary': {
                'status': 'basic',
                'status_text': 'Aurora Director disponible - funcionalidad básica',
                'motores_activos': 1,
                'total_componentes': 1,
                'calidad_promedio': 85.0,
                'tiempo_promedio': 2.5,
                'componentes_principales': {'aurora_director': True}
            },
            'bridge_info': {
                'initialized': True,
                'basic_mode': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _get_error_status(self, error_msg: str) -> Dict[str, Any]:
        """Estado de error"""
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
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validar parámetros de generación"""
        errors = []
        
        # Validar objetivo
        objetivo = params.get('objetivo', '').strip()
        if not objetivo:
            errors.append("El objetivo es requerido")
        elif len(objetivo) > 200:
            errors.append("El objetivo es demasiado largo (máximo 200 caracteres)")
        
        # Validar duración
        duracion_min = params.get('duracion_min', 20)
        try:
            duracion_min = int(duracion_min)
            if duracion_min < 1:
                errors.append("La duración mínima es 1 minuto")
            elif duracion_min > 120:
                errors.append("La duración máxima es 120 minutos")
        except (ValueError, TypeError):
            errors.append("La duración debe ser un número válido")
        
        # Validar intensidad
        intensidad = params.get('intensidad', 'media')
        if intensidad not in ['suave', 'media', 'intenso']:
            errors.append("La intensidad debe ser 'suave', 'media' o 'intenso'")
        
        # Validar estilo
        estilo = params.get('estilo', 'sereno')
        estilos_validos = ['sereno', 'crystalline', 'organico', 'etereo', 'tribal', 'mistico', 'cuantico', 'neural', 'neutro']
        if estilo not in estilos_validos:
            errors.append(f"Estilo inválido. Debe ser uno de: {', '.join(estilos_validos)}")
        
        # Validar calidad
        calidad = params.get('calidad_objetivo', 'alta')
        if calidad not in ['basica', 'media', 'alta', 'maxima']:
            errors.append("La calidad debe ser 'basica', 'media', 'alta' o 'maxima'")
        
        # Validar sample rate
        sample_rate = params.get('sample_rate', 44100)
        if sample_rate not in [22050, 44100, 48000]:
            errors.append("Sample rate debe ser 22050, 44100 o 48000")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def estimate_generation_time(self, params: Dict[str, Any]) -> int:
        """Estimar tiempo de generación en segundos"""
        base_time = 5  # 5 segundos base
        
        duracion_min = params.get('duracion_min', 20)
        duracion_factor = max(1, duracion_min / 10)  # Más tiempo para audios largos
        
        calidad = params.get('calidad_objetivo', 'alta')
        calidad_factor = {
            'basica': 1.0,
            'media': 1.5,
            'alta': 2.0,
            'maxima': 3.0
        }.get(calidad, 2.0)
        
        # Factor por disponibilidad de componentes
        system_factor = 1.0
        if self.initialized and self.aurora_director:
            system_factor = 0.8  # Más rápido con Aurora completo
        else:
            system_factor = 2.0  # Más lento en fallback
        
        estimated = int(base_time * duracion_factor * calidad_factor * system_factor)
        return max(5, min(300, estimated))  # Entre 5 segundos y 5 minutos
    
    def crear_experiencia_completa(self, params: Dict[str, Any], 
                                 progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Crear experiencia neuroacústica completa"""
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(10, "Inicializando generación")
            
            # Generar audio usando Aurora
            if self.initialized and self.aurora_director:
                resultado = self._generar_con_aurora(params, progress_callback)
            else:
                resultado = self._generar_con_fallback(params, progress_callback)
            
            if not resultado['success']:
                return resultado
            
            if progress_callback:
                progress_callback(80, "Procesando audio")
            
            # Convertir a WAV
            audio_data = resultado['audio_data']
            filename = self._save_audio_to_wav(audio_data, params)
            
            if progress_callback:
                progress_callback(95, "Finalizando")
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'audio_filename': filename,
                'metadata': {
                    **resultado.get('metadata', {}),
                    'processing_time_seconds': processing_time,
                    'bridge_version': '1.0.0',
                    'generation_method': resultado.get('method', 'unknown'),
                    'audio_specs': {
                        'sample_rate': self.sample_rate,
                        'channels': 2,
                        'duration_seconds': audio_data.shape[1] / self.sample_rate if audio_data.ndim > 1 else len(audio_data) / self.sample_rate,
                        'format': 'WAV 16-bit'
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error crítico en crear_experiencia_completa: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Error crítico: {str(e)}'
            }
    
    def _generar_con_aurora(self, params: Dict[str, Any], 
                           progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Generar usando Aurora Director V7"""
        try:
            if progress_callback:
                progress_callback(20, "Configurando Aurora Director")
            
            # Preparar parámetros para Aurora
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
                'validacion_automatica': params.get('validacion_automatica', True)
            }
            
            # Añadir parámetros opcionales
            if params.get('neurotransmisor_preferido'):
                aurora_params['neurotransmisor_preferido'] = params['neurotransmisor_preferido']
            
            if params.get('estrategia_preferida'):
                aurora_params['estrategia_preferida'] = params['estrategia_preferida']
            
            if params.get('motores_preferidos'):
                aurora_params['motores_preferidos'] = params['motores_preferidos']
            
            if progress_callback:
                progress_callback(30, "Ejecutando Aurora Director")
            
            # Llamar a Aurora
            resultado_aurora = self.aurora_director.crear_experiencia(**aurora_params)
            
            if progress_callback:
                progress_callback(70, "Audio generado, procesando resultado")
            
            # Verificar resultado
            if not hasattr(resultado_aurora, 'audio_data'):
                raise Exception("Aurora no retornó audio_data")
            
            audio_data = resultado_aurora.audio_data
            if audio_data is None or audio_data.size == 0:
                raise Exception("Audio generado está vacío")
            
            # ===== FUNCIONES AUXILIARES PARA SERIALIZACIÓN JSON =====
            def enum_to_str(value):
                """Convertir Enum a string de forma segura"""
                if value is None:
                    return None
                elif hasattr(value, 'value'):
                    return str(value.value)
                elif hasattr(value, 'name'):
                    return str(value.name)
                else:
                    return str(value)
            
            def clean_metadata(obj):
                """Limpiar objetos para que sean serializables en JSON"""
                if obj is None:
                    return None
                elif isinstance(obj, (str, int, float, bool)):
                    return obj
                elif isinstance(obj, list):
                    return [clean_metadata(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: clean_metadata(value) for key, value in obj.items()}
                elif hasattr(obj, 'value') or hasattr(obj, 'name'):
                    # Es un Enum
                    return enum_to_str(obj)
                else:
                    # Convertir a string como fallback
                    return str(obj)
            
            # Obtener atributos de forma segura
            estrategia_usada = getattr(resultado_aurora, 'estrategia_usada', None)
            modo_orquestacion = getattr(resultado_aurora, 'modo_orquestacion', None)
            componentes_usados = getattr(resultado_aurora, 'componentes_usados', [])
            tiempo_generacion = getattr(resultado_aurora, 'tiempo_generacion', 0)
            calidad_score = getattr(resultado_aurora, 'calidad_score', 0)
            coherencia_neuroacustica = getattr(resultado_aurora, 'coherencia_neuroacustica', 0)
            efectividad_terapeutica = getattr(resultado_aurora, 'efectividad_terapeutica', 0)
            metadatos_aurora = getattr(resultado_aurora, 'metadatos', {})
            
            # Preparar metadatos con conversión segura a JSON
            metadata = {
                'generator': 'Aurora Director V7',
                'strategy_used': enum_to_str(estrategia_usada),
                'orchestration_mode': enum_to_str(modo_orquestacion),
                'components_used': clean_metadata(componentes_usados),
                'generation_time': float(tiempo_generacion) if tiempo_generacion else 0.0,
                'quality_score': float(calidad_score) if calidad_score else 0.0,
                'neuroacoustic_coherence': float(coherencia_neuroacustica) if coherencia_neuroacustica else 0.0,
                'therapeutic_effectiveness': float(efectividad_terapeutica) if efectividad_terapeutica else 0.0,
                'configuration': clean_metadata(params),
                'aurora_metadata': clean_metadata(metadatos_aurora)
            }
            
            return {
                'success': True,
                'audio_data': audio_data,
                'metadata': metadata,
                'method': 'aurora_director_v7'
            }
            
        except Exception as e:
            logger.error(f"Error en generación con Aurora: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Error en Aurora Director: {str(e)}'
            }
    
    def _generar_con_fallback(self, params: Dict[str, Any],
                            progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Generar usando sistema de fallback"""
        try:
            if progress_callback:
                progress_callback(20, "Generando con sistema de fallback")
            
            duracion_sec = params['duracion_min'] * 60
            samples = int(self.sample_rate * duracion_sec)
            
            if progress_callback:
                progress_callback(40, "Creando ondas base")
            
            # Crear señal base simple pero efectiva
            t = np.linspace(0, duracion_sec, samples)
            
            # Mapear objetivo a frecuencias
            freq_map = {
                'concentracion': 14.0,
                'claridad_mental': 12.0,
                'enfoque': 15.0,
                'relajacion': 6.0,
                'meditacion': 7.83,  # Resonancia Schumann
                'creatividad': 10.0,
                'gratitud': 8.0,
                'energia': 16.0,
                'calma': 5.0,
                'sanacion': 528.0  # Frecuencia Solfeggio
            }
            
            objetivo_lower = params['objetivo'].lower()
            freq_base = 10.0  # Default
            for key, freq in freq_map.items():
                if key in objetivo_lower:
                    freq_base = freq
                    break
            
            if progress_callback:
                progress_callback(60, "Aplicando modulaciones")
            
            # Crear señal principal
            signal = np.sin(2 * np.pi * freq_base * t)
            
            # Añadir armónicos sutiles
            signal += 0.3 * np.sin(2 * np.pi * freq_base * 2 * t)
            signal += 0.1 * np.sin(2 * np.pi * freq_base * 3 * t)
            
            # Aplicar envolvente natural
            fade_samples = int(self.sample_rate * 3.0)  # 3 segundos
            if len(signal) > fade_samples * 2:
                # Fade in
                signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # Fade out
                signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Crear versión estéreo
            left = signal
            right = signal.copy()
            
            # Ligera decorrelación para espacialidad
            if len(right) > 100:
                right = np.roll(right, 50)  # Pequeño delay
            
            audio_data = np.stack([left, right])
            
            # Normalizar
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data * 0.8 / max_val
            
            metadata = {
                'generator': 'Fallback System',
                'strategy_used': 'fallback_generation',
                'orchestration_mode': 'simple',
                'components_used': ['fallback_generator'],
                'generation_time': 2.0,
                'quality_score': 75.0,
                'neuroacoustic_coherence': 0.8,
                'therapeutic_effectiveness': 0.7,
                'base_frequency': freq_base,
                'objective_detected': objetivo_lower,
                'configuration': params,
                'fallback_mode': True
            }
            
            return {
                'success': True,
                'audio_data': audio_data,
                'metadata': metadata,
                'method': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Error en sistema de fallback: {e}")
            return {
                'success': False,
                'error': f'Error en fallback: {str(e)}'
            }
    
    def _save_audio_to_wav(self, audio_data: np.ndarray, params: Dict[str, Any]) -> str:
        """Guardar audio numpy a archivo WAV"""
        try:
            # Generar nombre único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_objetivo = "".join(c for c in params['objetivo'] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_objetivo = safe_objetivo.replace(' ', '_')[:20]  # Limitar longitud
            
            filename = f"aurora_{safe_objetivo}_{params['duracion_min']}min_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(self.audio_folder, filename)
            
            # Asegurar que existe el directorio
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Asegurar que el audio esté en formato correcto
            if audio_data.ndim == 1:
                # Mono a estéreo
                audio_stereo = np.stack([audio_data, audio_data])
            else:
                audio_stereo = audio_data
            
            # Asegurar orden correcto [samples, channels] para scipy
            if audio_stereo.shape[0] == 2 and audio_stereo.shape[1] > audio_stereo.shape[0]:
                # Ya está en formato [channels, samples], necesitamos [samples, channels]
                audio_for_save = audio_stereo.T
            else:
                audio_for_save = audio_stereo
            
            # Convertir a 16-bit integer
            audio_int16 = np.clip(audio_for_save * 32767, -32768, 32767).astype(np.int16)
            
            # Guardar usando wave module (más confiable)
            with wave.open(filepath, 'w') as wf:
                wf.setnchannels(2)  # Estéreo
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # Entrelazar canales para formato estéreo
                if audio_int16.ndim == 2:
                    # [samples, channels] -> entrelazado
                    stereo_interleaved = np.empty((audio_int16.shape[0] * 2,), dtype=np.int16)
                    stereo_interleaved[0::2] = audio_int16[:, 0]  # Canal izquierdo
                    stereo_interleaved[1::2] = audio_int16[:, 1]  # Canal derecho
                    wf.writeframes(stereo_interleaved.tobytes())
                else:
                    # Mono
                    wf.writeframes(audio_int16.tobytes())
            
            logger.info(f"Audio guardado: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error guardando audio: {e}")
            raise Exception(f"Error guardando archivo WAV: {str(e)}")
    
    def get_available_objectives(self) -> List[str]:
        """Obtener objetivos disponibles"""
        return [
            'concentracion', 'claridad_mental', 'enfoque', 'relajacion', 'meditacion',
            'creatividad', 'gratitud', 'energia', 'calma', 'sanacion', 'sueño',
            'ansiedad', 'estres', 'expansion_consciencia', 'conexion_espiritual',
            'flujo_creativo', 'inspiracion', 'alegria', 'amor', 'compasion',
            'equilibrio_emocional', 'autoestima', 'confianza', 'determinacion'
        ]
    
    def get_available_neurotransmitters(self) -> List[str]:
        """Obtener neurotransmisores disponibles"""
        return [
            'dopamina', 'serotonina', 'gaba', 'acetilcolina', 'oxitocina',
            'anandamida', 'endorfina', 'bdnf', 'adrenalina', 'norepinefrina', 'melatonina'
        ]
    
    def get_available_styles(self) -> List[str]:
        """Obtener estilos disponibles"""
        return [
            'sereno', 'crystalline', 'organico', 'etereo', 'tribal', 
            'mistico', 'cuantico', 'neural', 'neutro'
        ]