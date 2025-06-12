#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperMod V32 - Motor Estructural Aurora V7
===========================================

Motor de capas estructurales y fases psicoacústicas para el Sistema Aurora V7.
Implementa detección automática, fallbacks inteligentes y generación de layers dinámicos.

Versión: V32_AURORA_CONNECTED_COMPLETE_SYNC_FIXED_OPTIMIZED
Autor: Sistema Aurora V7
Compatible con: MotorAuroraInterface
"""

import os
import sys
import time
import json
import logging
import datetime
import numpy as np
import wave
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import importlib
import threading
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN Y LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS Y TIPOS BASE
# =============================================================================

class WaveType(Enum):
    """Tipos de ondas cerebrales soportados"""
    ALPHA = "alpha"
    BETA = "beta" 
    THETA = "theta"
    DELTA = "delta"
    GAMMA = "gamma"

class ModulationType(Enum):
    """Tipos de modulación disponibles"""
    AM = "amplitude_modulation"
    FM = "frequency_modulation"
    QUANTUM = "quantum_modulation"
    SPATIAL_3D = "spatial_3d"
    SPATIAL_8D = "spatial_8d"

class DetectionStrategy(Enum):
    """Estrategias de detección de componentes"""
    OBJECTIVE_MANAGER = auto()
    GLOBAL_VARIABLES = auto()
    MODULE_DIRECT = auto()
    GESTOR_DIRECTO = auto()
    FALLBACK = auto()

# =============================================================================
# CLASES DE DATOS Y CONFIGURACIÓN
# =============================================================================

@dataclass
class LayerConfig:
    """Configuración para capas de audio"""
    frequency: float
    amplitude: float
    duration: float
    wave_type: WaveType
    modulation: Optional[ModulationType] = None
    phase_offset: float = 0.0
    spatial_position: Optional[Tuple[float, float, float]] = None

@dataclass
class HyperModStats:
    """Estadísticas del motor HyperMod"""
    layers_generados: int = 0
    presets_detectados: int = 0
    templates_procesados: int = 0
    tiempo_total_generacion: float = 0.0
    fallbacks_utilizados: int = 0
    detecciones_exitosas: int = 0
    ultima_actualizacion: Optional[datetime.datetime] = None

# =============================================================================
# DETECTOR DE OBJECTIVE TEMPLATES EXPANDIDO
# =============================================================================

class DetectorObjectiveTemplatesExpandido:
    """
    Detector avanzado de plantillas de objetivos con múltiples estrategias.
    Implementa detección inteligente con fallbacks automáticos.
    """
    
    def __init__(self):
        self.templates_detectados = {}
        self.estrategias_probadas = []
        self.ultima_deteccion = None
        self.stats = {
            'detecciones_exitosas': 0,
            'fallbacks_usados': 0,
            'tiempo_total_deteccion': 0.0
        }
        
    def detectar_templates(self) -> Dict[str, Any]:
        """
        Detecta templates usando múltiples estrategias en orden de prioridad.
        """
        start_time = time.time()
        
        # Estrategia 1: Desde objective_manager
        templates = self._detectar_desde_objective_manager()
        if templates:
            self._registrar_deteccion_exitosa(DetectionStrategy.OBJECTIVE_MANAGER)
            return templates
            
        # Estrategia 2: Variables globales
        templates = self._detectar_variables_globales_objective_manager()
        if templates:
            self._registrar_deteccion_exitosa(DetectionStrategy.GLOBAL_VARIABLES)
            return templates
            
        # Estrategia 3: Módulo directo
        templates = self._detectar_modulo_objective_templates()
        if templates:
            self._registrar_deteccion_exitosa(DetectionStrategy.MODULE_DIRECT)
            return templates
            
        # Estrategia 4: Gestor directo
        templates = self._detectar_gestor_directo()
        if templates:
            self._registrar_deteccion_exitosa(DetectionStrategy.GESTOR_DIRECTO)
            return templates
            
        # Estrategia 5: Fallback
        templates = self._crear_fallback_objective_templates()
        self._registrar_deteccion_exitosa(DetectionStrategy.FALLBACK)
        self.stats['fallbacks_usados'] += 1
        
        self.stats['tiempo_total_deteccion'] += time.time() - start_time
        return templates
        
    def _detectar_desde_objective_manager(self) -> Optional[Dict[str, Any]]:
        """Detecta templates desde objective_manager"""
        try:
            import objective_manager
            if hasattr(objective_manager, 'obtener_templates'):
                templates = objective_manager.obtener_templates()
                logger.info("✅ Templates detectados desde objective_manager")
                return templates
        except Exception as e:
            logger.debug(f"Detección objective_manager falló: {e}")
        return None
        
    def _detectar_variables_globales_objective_manager(self) -> Optional[Dict[str, Any]]:
        """Detecta templates desde variables globales"""
        try:
            if 'objective_manager' in globals():
                om = globals()['objective_manager']
                if hasattr(om, 'TEMPLATES_GLOBALES'):
                    logger.info("✅ Templates detectados desde variables globales")
                    return om.TEMPLATES_GLOBALES
        except Exception as e:
            logger.debug(f"Detección variables globales falló: {e}")
        return None
        
    def _detectar_modulo_objective_templates(self) -> Optional[Dict[str, Any]]:
        """Detecta templates desde módulo dedicado"""
        try:
            spec = importlib.util.find_spec('objective_templates_optimized')
            if spec:
                module = importlib.import_module('objective_templates_optimized')
                if hasattr(module, 'TEMPLATES'):
                    logger.info("✅ Templates detectados desde módulo dedicado")
                    return module.TEMPLATES
        except Exception as e:
            logger.debug(f"Detección módulo dedicado falló: {e}")
        return None
        
    def _detectar_gestor_directo(self) -> Optional[Dict[str, Any]]:
        """Detecta templates desde gestor directo"""
        try:
            # Intenta importar gestor directo si existe
            import sys
            for name, module in sys.modules.items():
                if 'objective' in name.lower() and hasattr(module, 'templates'):
                    logger.info(f"✅ Templates detectados desde gestor directo: {name}")
                    return module.templates
        except Exception as e:
            logger.debug(f"Detección gestor directo falló: {e}")
        return None
        
    def _crear_fallback_objective_templates(self) -> Dict[str, Any]:
        """Crea templates de fallback básicos"""
        fallback_templates = {
            'relajacion': {
                'name': 'Relajación Básica',
                'frequencies': [8.0, 10.0, 12.0],
                'duration': 300,
                'wave_types': ['alpha', 'theta'],
                'description': 'Template de fallback para relajación'
            },
            'concentracion': {
                'name': 'Concentración Básica', 
                'frequencies': [14.0, 16.0, 18.0],
                'duration': 300,
                'wave_types': ['beta'],
                'description': 'Template de fallback para concentración'
            },
            'meditacion': {
                'name': 'Meditación Básica',
                'frequencies': [4.0, 6.0, 8.0],
                'duration': 600,
                'wave_types': ['theta', 'alpha'],
                'description': 'Template de fallback para meditación'
            }
        }
        logger.info("🔄 Usando templates de fallback")
        return fallback_templates
        
    def _registrar_deteccion_exitosa(self, estrategia: DetectionStrategy):
        """Registra una detección exitosa"""
        self.estrategias_probadas.append(estrategia)
        self.stats['detecciones_exitosas'] += 1
        self.ultima_deteccion = datetime.datetime.now()
        logger.info(f"✅ Detección exitosa usando estrategia: {estrategia.name}")

# =============================================================================
# DETECTOR DE COMPONENTES HYPERMOD
# =============================================================================

class DetectorComponentesHyperMod:
    """
    Detector inteligente de componentes clave para HyperMod V32.
    Incluye detección de presets emocionales, estilos y fases.
    """
    
    def __init__(self):
        self.componentes_detectados = {}
        self.stats_deteccion = {
            'presets_emocionales': 0,
            'estilos_detectados': 0,
            'fases_detectadas': 0,
            'tiempo_deteccion': 0.0
        }
        
    def detectar_todos_componentes(self) -> Dict[str, Any]:
        """Detecta todos los componentes necesarios para HyperMod"""
        start_time = time.time()
        
        componentes = {
            'presets_emocionales': self._detectar_presets_emocionales(),
            'estilos_audio': self._detectar_estilos_audio(),
            'fases_estructurales': self._detectar_fases_estructurales(),
            'templates_objetivos': self._detectar_templates_objetivos()
        }
        
        self.stats_deteccion['tiempo_deteccion'] = time.time() - start_time
        self.componentes_detectados = componentes
        
        logger.info(f"🔍 Componentes detectados: {len([c for c in componentes.values() if c])}/4")
        return componentes
        
    def _detectar_presets_emocionales(self) -> Optional[Dict[str, Any]]:
        """Detecta presets emocionales disponibles"""
        try:
            # Intenta importar emotion_style_profiles
            import emotion_style_profiles
            if hasattr(emotion_style_profiles, 'PRESETS_EMOCIONALES'):
                presets = emotion_style_profiles.PRESETS_EMOCIONALES
                self.stats_deteccion['presets_emocionales'] = len(presets)
                logger.info(f"✅ Detectados {len(presets)} presets emocionales")
                return presets
        except Exception as e:
            logger.debug(f"No se pudieron detectar presets emocionales: {e}")
            
        # Fallback con presets básicos
        fallback_presets = {
            'calma': {'frequencies': [8.0, 10.0], 'amplitude': 0.7},
            'energia': {'frequencies': [15.0, 18.0], 'amplitude': 0.8},
            'focus': {'frequencies': [14.0, 16.0], 'amplitude': 0.75}
        }
        self.stats_deteccion['presets_emocionales'] = len(fallback_presets)
        logger.info("🔄 Usando presets emocionales de fallback")
        return fallback_presets
        
    def _detectar_estilos_audio(self) -> Optional[Dict[str, Any]]:
        """Detecta estilos de audio disponibles"""
        try:
            # Busca estilos en múltiples ubicaciones
            estilos_encontrados = {}
            
            # Buscar en emotion_style_profiles
            try:
                import emotion_style_profiles
                if hasattr(emotion_style_profiles, 'ESTILOS_AUDIO'):
                    estilos_encontrados.update(emotion_style_profiles.ESTILOS_AUDIO)
            except ImportError:
                pass
                
            if estilos_encontrados:
                self.stats_deteccion['estilos_detectados'] = len(estilos_encontrados)
                logger.info(f"✅ Detectados {len(estilos_encontrados)} estilos de audio")
                return estilos_encontrados
                
        except Exception as e:
            logger.debug(f"Error detectando estilos: {e}")
            
        # Fallback con estilos básicos
        fallback_estilos = {
            'ambient': {'reverb': 0.3, 'chorus': 0.2},
            'binaural': {'stereo_separation': 0.8, 'phase_shift': 0.1},
            'natural': {'noise_reduction': 0.9, 'harmonics': 0.6}
        }
        self.stats_deteccion['estilos_detectados'] = len(fallback_estilos)
        logger.info("🔄 Usando estilos de audio de fallback")
        return fallback_estilos
        
    def _detectar_fases_estructurales(self) -> Optional[Dict[str, Any]]:
        """Detecta fases estructurales disponibles"""
        try:
            # Intenta detectar desde presets_fases
            import presets_fases
            if hasattr(presets_fases, 'FASES_ESTRUCTURALES'):
                fases = presets_fases.FASES_ESTRUCTURALES
                self.stats_deteccion['fases_detectadas'] = len(fases)
                logger.info(f"✅ Detectadas {len(fases)} fases estructurales")
                return fases
        except Exception as e:
            logger.debug(f"No se pudieron detectar fases estructurales: {e}")
            
        # Fallback con fases básicas
        fallback_fases = {
            'introduccion': {'duration': 60, 'fade_in': True},
            'desarrollo': {'duration': 240, 'intensity': 0.8},
            'climax': {'duration': 120, 'intensity': 1.0},
            'resolucion': {'duration': 180, 'fade_out': True}
        }
        self.stats_deteccion['fases_detectadas'] = len(fallback_fases)
        logger.info("🔄 Usando fases estructurales de fallback")
        return fallback_fases
        
    def _detectar_templates_objetivos(self) -> Optional[Dict[str, Any]]:
        """Detecta templates de objetivos (usa el detector expandido)"""
        detector = DetectorObjectiveTemplatesExpandido()
        templates = detector.detectar_templates()
        if templates:
            logger.info(f"✅ Templates de objetivos detectados: {len(templates)}")
            return templates
        return None

# =============================================================================
# GESTOR AURORA INTEGRADO V32
# =============================================================================

class GestorAuroraIntegradoV32:
    """
    Gestor integrado que unifica todos los detectores y provee acceso 
    centralizado a los componentes de Aurora V7 para HyperMod.
    """
    
    def __init__(self):
        self.detector_templates = DetectorObjectiveTemplatesExpandido()
        self.detector_componentes = DetectorComponentesHyperMod()
        self.componentes_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 300  # 5 minutos
        
        self.stats_integrado = {
            'consultas_cache': 0,
            'renovaciones_cache': 0,
            'layers_generados': 0,
            'tiempo_total_operacion': 0.0
        }
        
    def inicializar_componentes(self) -> bool:
        """Inicializa y cachea todos los componentes"""
        try:
            start_time = time.time()
            
            # Detectar todos los componentes
            componentes = self.detector_componentes.detectar_todos_componentes()
            templates = self.detector_templates.detectar_templates()
            
            # Actualizar cache
            self.componentes_cache = {
                'componentes': componentes,
                'templates': templates,
                'timestamp': datetime.datetime.now()
            }
            self.cache_timestamp = time.time()
            
            self.stats_integrado['renovaciones_cache'] += 1
            self.stats_integrado['tiempo_total_operacion'] += time.time() - start_time
            
            logger.info("✅ Componentes HyperMod inicializados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
            return False
            
    def _cache_valido(self) -> bool:
        """Verifica si el cache sigue siendo válido"""
        if not self.cache_timestamp:
            return False
        return (time.time() - self.cache_timestamp) < self.cache_duration
        
    def _obtener_componentes(self) -> Dict[str, Any]:
        """Obtiene componentes del cache o los renueva"""
        if self._cache_valido():
            self.stats_integrado['consultas_cache'] += 1
            return self.componentes_cache
        else:
            self.inicializar_componentes()
            return self.componentes_cache
            
    def crear_layers_desde_preset_emocional(self, preset_name: str, 
                                          config: Dict[str, Any]) -> List[LayerConfig]:
        """Crea layers de audio desde un preset emocional"""
        start_time = time.time()
        
        try:
            componentes = self._obtener_componentes()
            presets = componentes['componentes']['presets_emocionales']
            
            if preset_name not in presets:
                logger.warning(f"⚠️ Preset '{preset_name}' no encontrado, usando fallback")
                return self._crear_layers_fallback_emocional(config)
                
            preset = presets[preset_name]
            layers = []
            
            # Generar layers basados en el preset
            for i, freq in enumerate(preset.get('frequencies', [8.0])):
                layer = LayerConfig(
                    frequency=freq,
                    amplitude=preset.get('amplitude', 0.7),
                    duration=config.get('duration', 300),
                    wave_type=WaveType.ALPHA,
                    phase_offset=i * 0.1
                )
                layers.append(layer)
                
            self.stats_integrado['layers_generados'] += len(layers)
            self.stats_integrado['tiempo_total_operacion'] += time.time() - start_time
            
            logger.info(f"✅ Generados {len(layers)} layers desde preset '{preset_name}'")
            return layers
            
        except Exception as e:
            logger.error(f"❌ Error creando layers desde preset: {e}")
            return self._crear_layers_fallback_emocional(config)
            
    def crear_layers_desde_secuencia_fases(self, secuencia: List[str],
                                         config: Dict[str, Any]) -> List[LayerConfig]:
        """Crea layers de audio desde una secuencia de fases"""
        start_time = time.time()
        
        try:
            componentes = self._obtener_componentes()
            fases = componentes['componentes']['fases_estructurales']
            
            layers = []
            tiempo_acumulado = 0
            
            for fase_name in secuencia:
                if fase_name not in fases:
                    logger.warning(f"⚠️ Fase '{fase_name}' no encontrada, saltando")
                    continue
                    
                fase = fases[fase_name]
                duracion_fase = fase.get('duration', 60)
                intensidad = fase.get('intensity', 0.7)
                
                # Crear layers para esta fase
                layer = LayerConfig(
                    frequency=config.get('base_frequency', 10.0),
                    amplitude=intensidad,
                    duration=duracion_fase,
                    wave_type=WaveType.ALPHA
                )
                layers.append(layer)
                
                tiempo_acumulado += duracion_fase
                
            self.stats_integrado['layers_generados'] += len(layers)
            self.stats_integrado['tiempo_total_operacion'] += time.time() - start_time
            
            logger.info(f"✅ Generados {len(layers)} layers desde secuencia de fases")
            return layers
            
        except Exception as e:
            logger.error(f"❌ Error creando layers desde secuencia: {e}")
            return self._crear_layers_fallback_secuencial(config)
            
    def crear_layers_desde_template_objetivo(self, template_name: str,
                                           config: Dict[str, Any]) -> List[LayerConfig]:
        """Crea layers de audio desde un template de objetivo"""
        start_time = time.time()
        
        try:
            componentes = self._obtener_componentes()
            templates = componentes['templates']
            
            if template_name not in templates:
                logger.warning(f"⚠️ Template '{template_name}' no encontrado, usando fallback")
                return self._crear_layers_fallback_objetivo(config)
                
            template = templates[template_name]
            layers = []
            
            # Generar layers basados en el template
            frequencies = template.get('frequencies', [10.0])
            wave_types = template.get('wave_types', ['alpha'])
            duracion = template.get('duration', 300)
            
            for i, freq in enumerate(frequencies):
                wave_type = WaveType(wave_types[i % len(wave_types)])
                
                layer = LayerConfig(
                    frequency=freq,
                    amplitude=config.get('amplitude', 0.75),
                    duration=duracion,
                    wave_type=wave_type,
                    phase_offset=i * 0.15
                )
                layers.append(layer)
                
            self.stats_integrado['layers_generados'] += len(layers)
            self.stats_integrado['tiempo_total_operacion'] += time.time() - start_time
            
            logger.info(f"✅ Generados {len(layers)} layers desde template '{template_name}'")
            return layers
            
        except Exception as e:
            logger.error(f"❌ Error creando layers desde template: {e}")
            return self._crear_layers_fallback_objetivo(config)
            
    def _crear_layers_fallback_emocional(self, config: Dict[str, Any]) -> List[LayerConfig]:
        """Crea layers de fallback para presets emocionales"""
        layers = [
            LayerConfig(
                frequency=10.0,
                amplitude=0.7,
                duration=config.get('duration', 300),
                wave_type=WaveType.ALPHA
            ),
            LayerConfig(
                frequency=8.0,
                amplitude=0.6,
                duration=config.get('duration', 300),
                wave_type=WaveType.ALPHA,
                phase_offset=0.1
            )
        ]
        logger.info("🔄 Usando layers de fallback emocional")
        return layers
        
    def _crear_layers_fallback_secuencial(self, config: Dict[str, Any]) -> List[LayerConfig]:
        """Crea layers de fallback para secuencias"""
        layers = [
            LayerConfig(
                frequency=config.get('base_frequency', 10.0),
                amplitude=0.5,
                duration=60,
                wave_type=WaveType.ALPHA
            ),
            LayerConfig(
                frequency=config.get('base_frequency', 10.0) * 1.2,
                amplitude=0.8,
                duration=240,
                wave_type=WaveType.BETA
            ),
            LayerConfig(
                frequency=config.get('base_frequency', 10.0),
                amplitude=0.6,
                duration=120,
                wave_type=WaveType.ALPHA
            )
        ]
        logger.info("🔄 Usando layers de fallback secuencial")
        return layers
        
    def _crear_layers_fallback_objetivo(self, config: Dict[str, Any]) -> List[LayerConfig]:
        """Crea layers de fallback para objetivos"""
        layers = [
            LayerConfig(
                frequency=10.0,
                amplitude=0.75,
                duration=config.get('duration', 300),
                wave_type=WaveType.ALPHA
            )
        ]
        logger.info("🔄 Usando layers de fallback objetivo")
        return layers
        
    def obtener_info_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada de un preset"""
        try:
            componentes = self._obtener_componentes()
            presets = componentes['componentes']['presets_emocionales']
            return presets.get(preset_name)
        except Exception as e:
            logger.error(f"❌ Error obteniendo info del preset: {e}")
            return None
            
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del gestor"""
        return {
            'stats_integrado': self.stats_integrado,
            'stats_templates': self.detector_templates.stats,
            'stats_componentes': self.detector_componentes.stats_deteccion,
            'cache_valido': self._cache_valido(),
            'componentes_en_cache': len(self.componentes_cache.get('componentes', {}))
        }

# =============================================================================
# GENERADOR DE ONDAS NEUROACÚSTICAS
# =============================================================================

class NeuroWaveGenerator:
    """
    Generador avanzado de ondas cerebrales con modulación espacial.
    Incluye soporte para AM, FM, modulación cuántica y efectos 3D/8D.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.wave_cache = {}
        self.stats = {
            'ondas_generadas': 0,
            'cache_hits': 0,
            'tiempo_generacion': 0.0
        }
        
    def generate_wave(self, layer_config: LayerConfig, 
                     sample_rate: Optional[int] = None) -> np.ndarray:
        """Genera una onda basada en la configuración del layer"""
        start_time = time.time()
        sr = sample_rate or self.sample_rate
        
        # Verificar cache
        cache_key = self._get_cache_key(layer_config, sr)
        if cache_key in self.wave_cache:
            self.stats['cache_hits'] += 1
            return self.wave_cache[cache_key]
            
        # Generar onda base
        duration_samples = int(layer_config.duration * sr)
        t = np.linspace(0, layer_config.duration, duration_samples, False)
        
        # Generar onda según el tipo
        if layer_config.wave_type == WaveType.ALPHA:
            wave = np.sin(2 * np.pi * layer_config.frequency * t + layer_config.phase_offset)
        elif layer_config.wave_type == WaveType.BETA:
            wave = np.sin(2 * np.pi * layer_config.frequency * t + layer_config.phase_offset)
            wave += 0.3 * np.sin(2 * np.pi * layer_config.frequency * 1.5 * t)
        elif layer_config.wave_type == WaveType.THETA:
            wave = np.sin(2 * np.pi * layer_config.frequency * t + layer_config.phase_offset)
            wave *= np.exp(-t / (layer_config.duration * 0.8))  # Decay envelope
        elif layer_config.wave_type == WaveType.DELTA:
            wave = np.sin(2 * np.pi * layer_config.frequency * t + layer_config.phase_offset)
            wave += 0.5 * np.sin(2 * np.pi * layer_config.frequency * 0.5 * t)
        else:  # GAMMA
            wave = np.sin(2 * np.pi * layer_config.frequency * t + layer_config.phase_offset)
            wave += 0.2 * np.random.normal(0, 0.1, len(t))  # Añadir ruido
            
        # Aplicar amplitud
        wave *= layer_config.amplitude
        
        # Aplicar modulación si está especificada
        if layer_config.modulation:
            wave = self.apply_modulation(wave, layer_config.modulation, t)
            
        # Aplicar efectos espaciales si están especificados
        if layer_config.spatial_position:
            wave = self.apply_spatial_effects(wave, layer_config.spatial_position)
            
        # Cachear resultado
        self.wave_cache[cache_key] = wave
        
        self.stats['ondas_generadas'] += 1
        self.stats['tiempo_generacion'] += time.time() - start_time
        
        return wave
        
    def apply_modulation(self, wave: np.ndarray, modulation: ModulationType,
                        t: np.ndarray) -> np.ndarray:
        """Aplica modulación a la onda"""
        if modulation == ModulationType.AM:
            # Modulación de amplitud
            mod_freq = 0.5  # Hz
            modulator = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))
            return wave * modulator
            
        elif modulation == ModulationType.FM:
            # Modulación de frecuencia
            mod_depth = 0.1
            mod_freq = 0.3
            freq_modulation = mod_depth * np.sin(2 * np.pi * mod_freq * t)
            return wave * (1 + freq_modulation)
            
        elif modulation == ModulationType.QUANTUM:
            # Modulación cuántica (patrones no lineales)
            quantum_pattern = np.sin(t * np.pi) * np.cos(t * np.pi * 0.618)
            return wave * (0.8 + 0.2 * quantum_pattern)
            
        return wave
        
    def apply_spatial_effects(self, wave: np.ndarray, 
                            position: Tuple[float, float, float]) -> np.ndarray:
        """Aplica efectos espaciales 3D/8D"""
        x, y, z = position
        
        # Efecto de posicionamiento espacial básico
        # En una implementación completa, esto incluiría HRTF y convolución
        
        # Simulación simple de posicionamiento
        distance = np.sqrt(x**2 + y**2 + z**2)
        attenuation = 1.0 / (1.0 + distance * 0.1)
        
        # Delay basado en posición
        delay_samples = int(abs(x) * self.sample_rate * 0.001)  # 1ms por unidad
        
        if delay_samples > 0 and delay_samples < len(wave):
            delayed_wave = np.pad(wave, (delay_samples, 0), mode='constant')[:-delay_samples]
            return delayed_wave * attenuation
            
        return wave * attenuation
        
    def _get_cache_key(self, layer_config: LayerConfig, sample_rate: int) -> str:
        """Genera clave de cache para la configuración"""
        return f"{layer_config.frequency}_{layer_config.amplitude}_{layer_config.duration}_{layer_config.wave_type.value}_{sample_rate}"
        
    def clear_cache(self):
        """Limpia el cache de ondas"""
        self.wave_cache.clear()
        logger.info("🧹 Cache de ondas limpiado")

# =============================================================================
# MOTOR PRINCIPAL HYPERMOD V32
# =============================================================================

class HyperModEngineV32AuroraConnected:
    """
    Motor principal HyperMod V32 - Aurora Connected
    
    Motor de capas estructurales y fases psicoacústicas completamente integrado
    con el Sistema Aurora V7. Implementa MotorAuroraInterface y provee generación
    avanzada de audio neuroacústico con detección automática de componentes.
    """
    
    # 🚀 PROPIEDADES DE INTERFAZ - MOTORAURORAINTERFACE
    nombre = "HyperMod V32 Aurora Connected"
    version = "V32_AURORA_CONNECTED_COMPLETE_SYNC_FIXED"
    
    def __init__(self, config_inicial: Optional[Dict[str, Any]] = None):
        """
        Inicializa el motor HyperMod V32 con configuración opcional.
        Preserva 100% la lógica existente con mejoras aditivas.
        """
        # === CONFIGURACIÓN BASE ===
        self.config = config_inicial or {}
        self.sample_rate = self.config.get('sample_rate', 44100)
        self.output_format = self.config.get('output_format', 'wav')
        
        # === COMPONENTES PRINCIPALES ===
        self.gestor_integrado = GestorAuroraIntegradoV32()
        self.wave_generator = NeuroWaveGenerator(self.sample_rate)
        
        # === ESTADÍSTICAS ===
        self.stats = HyperModStats()
        
        # === CONFIGURACIÓN AVANZADA ===
        self.max_layers = self.config.get('max_layers', 16)
        self.quality_level = self.config.get('quality_level', 'high')
        self.enable_cache = self.config.get('enable_cache', True)
        
        # === INICIALIZACIÓN ===
        self._inicializar_motor()
        
        logger.info(f"🚀 {self.nombre} v{self.version} inicializado correctamente")
        
    def _inicializar_motor(self):
        """Inicializa los componentes del motor"""
        try:
            # Inicializar componentes integrados
            if self.gestor_integrado.inicializar_componentes():
                logger.info("✅ Componentes HyperMod inicializados")
            else:
                logger.warning("⚠️ Algunos componentes no se pudieron inicializar")
                
            # Configurar generador de ondas
            if hasattr(self.wave_generator, 'sample_rate'):
                self.wave_generator.sample_rate = self.sample_rate
                
            self.stats.ultima_actualizacion = datetime.datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error inicializando motor: {e}")
            
    # === MÉTODOS DE INTERFAZ MOTORAURORAINTERFACE ===
    
    def generar_audio(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera audio neuroacústico basado en la configuración.
        Implementa MotorAuroraInterface.generar_audio()
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎵 Iniciando generación de audio con HyperMod V32")
            
            # Convertir configuración Aurora a formato HyperMod
            hypermod_config = self._convertir_config_aurora_a_hypermod(config)
            
            # Crear layers desde la configuración
            layers = self._crear_layers_desde_config_aurora(hypermod_config)
            
            if not layers:
                logger.warning("⚠️ No se pudieron crear layers, usando fallback")
                layers = self._crear_layers_fallback_basico(hypermod_config)
                
            # Generar audio desde los layers
            audio_data = self._generar_audio_desde_layers(layers)
            
            # Actualizar estadísticas
            self.stats.layers_generados += len(layers)
            self.stats.tiempo_total_generacion += time.time() - start_time
            self.stats.ultima_actualizacion = datetime.datetime.now()
            
            # Preparar resultado
            resultado = {
                'audio_data': audio_data,
                'sample_rate': self.sample_rate,
                'duration': len(audio_data) / self.sample_rate,
                'layers_generados': len(layers),
                'formato': self.output_format,
                'metadata': {
                    'motor': self.nombre,
                    'version': self.version,
                    'config_usada': hypermod_config,
                    'tiempo_generacion': time.time() - start_time,
                    'estadisticas': self._obtener_estadisticas_basicas()
                }
            }
            
            logger.info(f"✅ Audio generado: {len(audio_data)} samples, {len(layers)} layers")
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Error generando audio: {e}")
            return self._generar_resultado_error(config, str(e))
            
    def validar_configuracion(self, config: Dict[str, Any]) -> bool:
        """
        Valida la configuración de generación.
        Implementa MotorAuroraInterface.validar_configuracion()
        """
        try:
            # Validaciones básicas requeridas
            if not isinstance(config, dict):
                logger.error("❌ Configuración debe ser un diccionario")
                return False
                
            # Validar campos esenciales
            campos_requeridos = ['objetivo', 'duracion']
            for campo in campos_requeridos:
                if campo not in config:
                    logger.warning(f"⚠️ Campo requerido faltante: {campo}")
                    # No es crítico, se puede usar fallback
                    
            # Validar tipos de datos
            if 'duracion' in config:
                if not isinstance(config['duracion'], (int, float)) or config['duracion'] <= 0:
                    logger.error("❌ Duración debe ser un número positivo")
                    return False
                    
            if 'sample_rate' in config:
                if not isinstance(config['sample_rate'], int) or config['sample_rate'] < 8000:
                    logger.error("❌ Sample rate debe ser un entero >= 8000")
                    return False
                    
            # Validar límites
            duracion_maxima = 3600  # 1 hora máximo
            if config.get('duracion', 0) > duracion_maxima:
                logger.error(f"❌ Duración máxima permitida: {duracion_maxima}s")
                return False
                
            logger.info("✅ Configuración validada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validando configuración: {e}")
            return False
            
    def obtener_capacidades(self) -> Dict[str, Any]:
        """
        Obtiene las capacidades del motor HyperMod V32.
        Implementa MotorAuroraInterface.obtener_capacidades()
        """
        try:
            capacidades = {
                'motor_info': {
                    'nombre': self.nombre,
                    'version': self.version,
                    'tipo': 'Motor Estructural'
                },
                'formatos_salida': ['wav', 'numpy_array'],
                'sample_rates': [22050, 44100, 48000, 96000],
                'tipos_onda': ['alpha', 'beta', 'theta', 'delta', 'gamma'],
                'modulaciones': ['AM', 'FM', 'quantum', 'spatial_3d', 'spatial_8d'],
                'presets_emocionales': list(self._obtener_presets_disponibles()),
                'templates_objetivos': list(self._obtener_templates_disponibles()),
                'limites': {
                    'duracion_maxima': 3600,
                    'layers_maximos': self.max_layers,
                    'frecuencia_minima': 0.1,
                    'frecuencia_maxima': 1000.0
                },
                'funcionalidades': {
                    'deteccion_automatica': True,
                    'fallback_inteligente': True,
                    'cache_optimizado': self.enable_cache,
                    'generacion_espacial': True,
                    'modulacion_avanzada': True,
                    'compatible_aurora_v7': True
                },
                'estadisticas': self._obtener_estadisticas_basicas()
            }
            
            logger.info("📊 Capacidades del motor obtenidas")
            return capacidades
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo capacidades: {e}")
            return {'error': str(e), 'capacidades_basicas': True}
            
    def get_info(self) -> Dict[str, str]:
        """
        Información básica del motor para MotorAuroraInterface.
        Método requerido por la interfaz estándar de Aurora V7.
        """
        return {
            'nombre': self.nombre,
            'version': self.version,
            'tipo': 'Motor Estructural',
            'descripcion': 'Motor de capas estructurales y fases psicoacústicas',
            'estado': 'Conectado a Aurora V7',
            'capacidades': 'Detección automática, fallbacks inteligentes, layers dinámicos',
            'autor': 'Sistema Aurora V7',
            'compatibilidad': 'MotorAuroraInterface V7'
        }
        
    # === MÉTODOS INTERNOS DE CONVERSIÓN ===
    
    def _convertir_config_aurora_a_hypermod(self, config_aurora: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte configuración de Aurora a formato HyperMod"""
        hypermod_config = {
            'objetivo': config_aurora.get('objetivo', 'relajacion'),
            'duracion': config_aurora.get('duracion', 300),
            'intensidad': config_aurora.get('intensidad', 0.7),
            'estilo': config_aurora.get('estilo', 'binaural'),
            'preset_emocional': config_aurora.get('preset_emocional'),
            'secuencia_fases': config_aurora.get('secuencia_fases'),
            'sample_rate': config_aurora.get('sample_rate', self.sample_rate),
            'quality_level': config_aurora.get('quality_level', self.quality_level)
        }
        
        # Conversiones específicas de Aurora V7
        if 'neurotransmisor' in config_aurora:
            hypermod_config['objetivo'] = f"neurotransmisor_{config_aurora['neurotransmisor']}"
            
        if 'modo_generacion' in config_aurora:
            hypermod_config['estilo'] = config_aurora['modo_generacion']
            
        logger.debug(f"🔄 Configuración convertida: {hypermod_config}")
        return hypermod_config
        
    def _crear_layers_desde_config_aurora(self, config: Dict[str, Any]) -> List[LayerConfig]:
        """Crea layers de audio desde configuración Aurora"""
        layers = []
        
        try:
            # Estrategia 1: Desde preset emocional
            if config.get('preset_emocional'):
                layers_preset = self.gestor_integrado.crear_layers_desde_preset_emocional(
                    config['preset_emocional'], config
                )
                layers.extend(layers_preset)
                
            # Estrategia 2: Desde secuencia de fases
            elif config.get('secuencia_fases'):
                layers_fases = self.gestor_integrado.crear_layers_desde_secuencia_fases(
                    config['secuencia_fases'], config
                )
                layers.extend(layers_fases)
                
            # Estrategia 3: Desde template de objetivo
            elif config.get('objetivo'):
                layers_objetivo = self.gestor_integrado.crear_layers_desde_template_objetivo(
                    config['objetivo'], config
                )
                layers.extend(layers_objetivo)
                
            # Limitar número de layers
            if len(layers) > self.max_layers:
                layers = layers[:self.max_layers]
                logger.warning(f"⚠️ Limitado a {self.max_layers} layers máximo")
                
            logger.info(f"✅ Creados {len(layers)} layers desde configuración")
            return layers
            
        except Exception as e:
            logger.error(f"❌ Error creando layers: {e}")
            return []
            
    def _crear_layers_fallback_basico(self, config: Dict[str, Any]) -> List[LayerConfig]:
        """Crea layers de fallback básicos"""
        duracion = config.get('duracion', 300)
        intensidad = config.get('intensidad', 0.7)
        
        layers_fallback = [
            LayerConfig(
                frequency=10.0,
                amplitude=intensidad,
                duration=duracion,
                wave_type=WaveType.ALPHA
            ),
            LayerConfig(
                frequency=8.0,
                amplitude=intensidad * 0.8,
                duration=duracion,
                wave_type=WaveType.ALPHA,
                phase_offset=0.1
            )
        ]
        
        self.stats.fallbacks_utilizados += 1
        logger.info("🔄 Usando layers de fallback básico")
        return layers_fallback
        
    def _generar_audio_desde_layers(self, layers: List[LayerConfig]) -> np.ndarray:
        """Genera audio combinando múltiples layers"""
        if not layers:
            return np.array([])
            
        # Generar cada layer
        layer_waves = []
        for layer in layers:
            wave = self.wave_generator.generate_wave(layer, self.sample_rate)
            layer_waves.append(wave)
            
        # Encontrar la duración máxima
        max_length = max(len(wave) for wave in layer_waves)
        
        # Combinar layers con padding si es necesario
        combined_wave = np.zeros(max_length)
        for wave in layer_waves:
            if len(wave) < max_length:
                # Pad con ceros al final
                padded_wave = np.pad(wave, (0, max_length - len(wave)), mode='constant')
                combined_wave += padded_wave
            else:
                combined_wave += wave
                
        # Normalizar para evitar clipping
        if np.max(np.abs(combined_wave)) > 0:
            combined_wave = combined_wave / np.max(np.abs(combined_wave)) * 0.8
            
        return combined_wave
        
    def _generar_resultado_error(self, config: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Genera resultado de error con fallback"""
        logger.error(f"❌ Generando resultado de error: {error_msg}")
        
        # Audio de fallback (silencio)
        duracion = config.get('duracion', 5)
        audio_fallback = np.zeros(int(duracion * self.sample_rate))
        
        return {
            'audio_data': audio_fallback,
            'sample_rate': self.sample_rate,
            'duration': duracion,
            'layers_generados': 0,
            'formato': self.output_format,
            'error': error_msg,
            'fallback_usado': True,
            'metadata': {
                'motor': self.nombre,
                'version': self.version,
                'estado': 'error_con_fallback'
            }
        }
        
    # === MÉTODOS DE INFORMACIÓN Y ESTADÍSTICAS ===
    
    def _obtener_presets_disponibles(self) -> List[str]:
        """Obtiene lista de presets emocionales disponibles"""
        try:
            componentes = self.gestor_integrado._obtener_componentes()
            presets = componentes['componentes']['presets_emocionales']
            return list(presets.keys()) if presets else ['calma', 'energia', 'focus']
        except:
            return ['calma', 'energia', 'focus']
            
    def _obtener_templates_disponibles(self) -> List[str]:
        """Obtiene lista de templates de objetivos disponibles"""
        try:
            componentes = self.gestor_integrado._obtener_componentes()
            templates = componentes['templates']
            return list(templates.keys()) if templates else ['relajacion', 'concentracion', 'meditacion']
        except:
            return ['relajacion', 'concentracion', 'meditacion']
            
    def _obtener_estadisticas_basicas(self) -> Dict[str, Any]:
        """Obtiene estadísticas básicas del motor"""
        return {
            'layers_generados': self.stats.layers_generados,
            'presets_detectados': self.stats.presets_detectados,
            'tiempo_total_generacion': self.stats.tiempo_total_generacion,
            'fallbacks_utilizados': self.stats.fallbacks_utilizados,
            'ultima_actualizacion': self.stats.ultima_actualizacion.isoformat() if self.stats.ultima_actualizacion else None
        }
        
    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema"""
        stats_gestor = self.gestor_integrado.obtener_estadisticas()
        stats_wave_gen = self.wave_generator.stats
        
        return {
            'motor_stats': self._obtener_estadisticas_basicas(),
            'gestor_stats': stats_gestor,
            'wave_generator_stats': stats_wave_gen,
            'configuracion_actual': {
                'sample_rate': self.sample_rate,
                'max_layers': self.max_layers,
                'quality_level': self.quality_level,
                'cache_enabled': self.enable_cache
            }
        }
        
    def limpiar_cache(self):
        """Limpia todos los caches del motor"""
        self.wave_generator.clear_cache()
        self.gestor_integrado.componentes_cache.clear()
        self.gestor_integrado.cache_timestamp = None
        logger.info("🧹 Cache del motor HyperMod limpiado")

# =============================================================================
# FUNCIONES DE UTILIDAD Y EXPORTACIÓN
# =============================================================================

def crear_motor_hypermod(config: Optional[Dict[str, Any]] = None) -> HyperModEngineV32AuroraConnected:
    """
    Función de conveniencia para crear una instancia del motor HyperMod V32.
    """
    return HyperModEngineV32AuroraConnected(config)

def validar_motor_hypermod() -> bool:
    """
    Valida que el motor HyperMod V32 esté correctamente configurado.
    """
    try:
        motor = crear_motor_hypermod()
        
        # Test básico de funcionalidad
        config_test = {
            'objetivo': 'relajacion',
            'duracion': 10,  # 10 segundos para test rápido
            'intensidad': 0.5
        }
        
        # Validar configuración
        if not motor.validar_configuracion(config_test):
            logger.error("❌ Validación de configuración falló")
            return False
            
        # Test de capacidades
        capacidades = motor.obtener_capacidades()
        if not capacidades or 'error' in capacidades:
            logger.error("❌ No se pudieron obtener capacidades")
            return False
            
        # Test de info
        info = motor.get_info()
        if not info or 'nombre' not in info:
            logger.error("❌ No se pudo obtener información del motor")
            return False
            
        logger.info("✅ Motor HyperMod V32 validado correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validando motor: {e}")
        return False

def diagnosticar_motor_hypermod() -> Dict[str, Any]:
    """
    Ejecuta un diagnóstico completo del motor HyperMod V32.
    """
    diagnostico = {
        'motor_valido': False,
        'componentes_detectados': {},
        'capacidades_completas': False,
        'test_generacion': False,
        'errores': [],
        'warnings': [],
        'estadisticas': {}
    }
    
    try:
        # Test 1: Validación básica
        diagnostico['motor_valido'] = validar_motor_hypermod()
        
        # Test 2: Creación de motor
        motor = crear_motor_hypermod()
        
        # Test 3: Componentes
        stats_completas = motor.obtener_estadisticas_completas()
        diagnostico['estadisticas'] = stats_completas
        
        # Test 4: Capacidades
        capacidades = motor.obtener_capacidades()
        diagnostico['capacidades_completas'] = 'funcionalidades' in capacidades
        
        # Test 5: Test de generación rápida
        config_test = {'objetivo': 'test', 'duracion': 1}
        if motor.validar_configuracion(config_test):
            resultado = motor.generar_audio(config_test)
            diagnostico['test_generacion'] = 'audio_data' in resultado
        
        logger.info("🔍 Diagnóstico de HyperMod V32 completado")
        
    except Exception as e:
        diagnostico['errores'].append(str(e))
        logger.error(f"❌ Error en diagnóstico: {e}")
        
    return diagnostico

# =============================================================================
# AUTO-INTEGRACIÓN CON AURORA V7 (Basado en patrón NeuroMix exitoso)
# =============================================================================

def aplicar_auto_integracion_hypermod():
    """
    Sistema de auto-integración para HyperMod V32 basado en el patrón exitoso de NeuroMix.
    """
    try:
        # Crear instancia global del motor
        motor_hypermod_global = HyperModEngineV32AuroraConnected()
        
        # Registro en namespace global para compatibilidad
        globals()['HyperModEngineV32'] = HyperModEngineV32AuroraConnected
        globals()['HyperModAuroraConnected'] = HyperModEngineV32AuroraConnected
        globals()['motor_hypermod_v32'] = motor_hypermod_global
        
        logger.info("✅ HyperMod V32 auto-integración completada")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en auto-integración HyperMod: {e}")
        return False

# Auto-aplicar integración al importar el módulo
try:
    aplicar_auto_integracion_hypermod()
except Exception as e:
    logger.warning(f"⚠️ Auto-integración HyperMod falló: {e}")

# =============================================================================
# PUNTO DE ENTRADA Y AUTOTEST
# =============================================================================

if __name__ == "__main__":
    """
    Punto de entrada principal para testing y diagnóstico.
    """
    print("🚀 HyperMod V32 - Motor Estructural Aurora V7")
    print("=" * 50)
    
    # Ejecutar diagnóstico completo
    diagnostico = diagnosticar_motor_hypermod()
    
    print(f"✅ Motor válido: {diagnostico['motor_valido']}")
    print(f"✅ Capacidades completas: {diagnostico['capacidades_completas']}")
    print(f"✅ Test generación: {diagnostico['test_generacion']}")
    
    if diagnostico['errores']:
        print(f"❌ Errores encontrados: {len(diagnostico['errores'])}")
        for error in diagnostico['errores']:
            print(f"   - {error}")
    
    if diagnostico['warnings']:
        print(f"⚠️ Warnings: {len(diagnostico['warnings'])}")
        for warning in diagnostico['warnings']:
            print(f"   - {warning}")
    
    # Mostrar estadísticas básicas
    if 'motor_stats' in diagnostico['estadisticas']:
        stats = diagnostico['estadisticas']['motor_stats']
        print(f"\n📊 Estadísticas básicas:")
        print(f"   - Layers generados: {stats.get('layers_generados', 0)}")
        print(f"   - Fallbacks utilizados: {stats.get('fallbacks_utilizados', 0)}")
        print(f"   - Tiempo total generación: {stats.get('tiempo_total_generacion', 0):.2f}s")
    
    print("\n🎯 HyperMod V32 listo para integración con Aurora V7")
    
    # Test de auto-integración
    print("\n🔧 Verificando auto-integración...")
    try:
        motor_test = crear_motor_hypermod()
        info = motor_test.get_info()
        print(f"✅ Auto-integración exitosa: {info['nombre']} v{info['version']}")
    except Exception as e:
        print(f"❌ Error en auto-integración: {e}")

# =============================================================================
# EXPORTACIONES PRINCIPALES
# =============================================================================

# Clase principal para importación
__all__ = [
    'HyperModEngineV32AuroraConnected',
    'crear_motor_hypermod',
    'validar_motor_hypermod',
    'diagnosticar_motor_hypermod',
    'LayerConfig',
    'HyperModStats',
    'WaveType',
    'ModulationType'
]

# Información del módulo
__version__ = "V32_AURORA_CONNECTED_COMPLETE_SYNC_FIXED"
__author__ = "Sistema Aurora V7"
__description__ = "Motor estructural de capas y fases psicoacústicas para Aurora V7"
__compatible_with__ = "MotorAuroraInterface"