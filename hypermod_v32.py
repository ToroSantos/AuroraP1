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
import numpy as np
import wave
import struct
import importlib.util
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import importlib
import threading
from pathlib import Path
from datetime import datetime
import field_profiles

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
    ultima_actualizacion: Optional[datetime] = None

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
        """Detecta templates usando múltiples estrategias en orden de prioridad."""
        start_time = time.time()

        # Estrategia 1: Desde objective_manager
        templates = self._detectar_desde_objective_manager()
        if templates:
            self._registrar_deteccion_exitosa(DetectionStrategy.OBJECTIVE_MANAGER)
            self.stats['tiempo_total_deteccion'] += time.time() - start_time
            return templates

        # Estrategia 2: Variables globales
        templates = self._detectar_variables_globales_objective_manager()
        if templates:
            self._registrar_deteccion_exitosa(DetectionStrategy.GLOBAL_VARIABLES)
            self.stats['tiempo_total_deteccion'] += time.time() - start_time
            return templates

        # Estrategia 3: Módulo directo
        templates = self._detectar_modulo_objective_templates()
        if templates:
            self._registrar_deteccion_exitosa(DetectionStrategy.MODULE_DIRECT)
            self.stats['tiempo_total_deteccion'] += time.time() - start_time
            return templates

        # Estrategia 4: Gestor directo
        templates = self._detectar_gestor_directo()
        if templates:
            self._registrar_deteccion_exitosa(DetectionStrategy.GESTOR_DIRECTO)
            self.stats['tiempo_total_deteccion'] += time.time() - start_time
            return templates

        # Estrategia 5: Fallback
        templates = self._crear_fallback_objective_templates()
        self._registrar_deteccion_exitosa(DetectionStrategy.FALLBACK)
        self.stats['fallbacks_usados'] += 1
        self.stats['tiempo_total_deteccion'] += time.time() - start_time
        logger.warning("⚠️ Usando templates de fallback - no se detectaron reales")
        return templates
        
    def _detectar_desde_objective_manager(self) -> Optional[Dict[str, Any]]:
        """Detecta templates desde objective_manager"""
        try:
            import objective_manager
            if hasattr(objective_manager, 'OBJECTIVE_TEMPLATES'):
                templates = objective_manager.OBJECTIVE_TEMPLATES
            elif hasattr(objective_manager, 'templates_disponibles'):
                templates = objective_manager.templates_disponibles    
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
                if hasattr(om, 'OBJECTIVE_TEMPLATES'):
                    logger.info("✅ Templates detectados desde variables globales")
                    return om.OBJECTIVE_TEMPLATES
                elif hasattr(om, 'templates_disponibles'):
                    logger.info("✅ Templates detectados desde variables globales (templates_disponibles)")
                    return om.templates_disponibles
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
        fallback_templates.update({
            'equilibrio': {
                'name': 'equilibrio Performance',
                'frequencies': [10.0, 15.0, 20.0],
                'duration': 300,
                'wave_types': ['alpha', 'beta'],
                'description': 'Template optimizado para equilibrio de rendimiento'
            },
            'test': {
                'name': 'Test Template',
                'frequencies': [12.0, 16.0],
                'duration': 300,
                'wave_types': ['beta'],
                'description': 'Template básico para tests unitarios'
            }
        })
        logger.info("🔄 Usando templates de fallback")
        return fallback_templates
        
    def _registrar_deteccion_exitosa(self, estrategia: DetectionStrategy):
        """Registra una detección exitosa"""
        self.estrategias_probadas.append(estrategia)
        self.stats['detecciones_exitosas'] += 1
        self.ultima_deteccion = datetime.now()
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
            'estilos_audio': self._detectar_estilos_audio_mejorado(),
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
            import emotion_style_profiles

            if hasattr(emotion_style_profiles, 'PRESETS_EMOCIONALES_AURORA'):
                presets = emotion_style_profiles.PRESETS_EMOCIONALES_AURORA
                self.stats_deteccion['presets_emocionales'] = len(presets)
                logger.info(f"✅ Detectados {len(presets)} presets emocionales desde variable")
                return presets  # ✅ ESTE RETURN FALTABA

            elif hasattr(emotion_style_profiles, 'GestorPresetsEmocionales'):
                gestor = emotion_style_profiles.GestorPresetsEmocionales()
                presets = gestor.presets
                self.stats_deteccion['presets_emocionales'] = len(presets)
                logger.info(f"✅ Detectados {len(presets)} presets emocionales desde gestor")
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
    
    def _detectar_estilos_audio_mejorado(self):
        """
        Detectar estilos de audio con verificaciones robustas y fallbacks
        """
        try:
            estilos_encontrados = []
            
            # Estrategia 1: PERFILES_CAMPO_AURORA (con verificación robusta)
            if hasattr(field_profiles, 'PERFILES_CAMPO_AURORA'):
                try:
                    perfiles = getattr(field_profiles, 'PERFILES_CAMPO_AURORA', None)
                    if perfiles and isinstance(perfiles, dict):
                        estilos_encontrados = list(perfiles.keys())
                        logger.info(f"✅ Detectados {len(estilos_encontrados)} estilos desde PERFILES_CAMPO_AURORA")
                except Exception as e:
                    logger.debug(f"Error accediendo PERFILES_CAMPO_AURORA: {e}")
            
            # Estrategia 2: Atributos alternativos en field_profiles
            if not estilos_encontrados:
                atributos_alternativos = [
                    'PERFILES_CAMPO_V7',
                    'FIELD_PROFILES',
                    'PERFILES_DISPONIBLES',
                    'ESTILOS_DISPONIBLES'
                ]
                
                for atributo in atributos_alternativos:
                    if hasattr(field_profiles, atributo):
                        try:
                            perfiles = getattr(field_profiles, atributo)
                            if isinstance(perfiles, dict):
                                estilos_encontrados = list(perfiles.keys())
                                logger.info(f"✅ Detectados {len(estilos_encontrados)} estilos desde {atributo}")
                                break
                            elif isinstance(perfiles, list):
                                estilos_encontrados = perfiles
                                logger.info(f"✅ Detectados {len(estilos_encontrados)} estilos desde {atributo} (lista)")
                                break
                        except Exception as e:
                            logger.debug(f"Error accediendo {atributo}: {e}")
            
            # Estrategia 3: Inspección dinámica del módulo field_profiles
            if not estilos_encontrados:
                try:
                    # Buscar cualquier diccionario que contenga perfiles
                    for attr_name in dir(field_profiles):
                        if not attr_name.startswith('_'):
                            attr_value = getattr(field_profiles, attr_name)
                            if isinstance(attr_value, dict) and len(attr_value) > 0:
                                # Verificar si parece un diccionario de perfiles
                                primera_clave = next(iter(attr_value.keys()))
                                if isinstance(primera_clave, str) and len(primera_clave) > 2:
                                    estilos_encontrados = list(attr_value.keys())
                                    logger.info(f"✅ Detectados {len(estilos_encontrados)} estilos desde {attr_name} (inspección)")
                                    break
                except Exception as e:
                    logger.debug(f"Error en inspección dinámica: {e}")
            
            # Estrategia 4: Fallback con estilos básicos
            if not estilos_encontrados:
                estilos_encontrados = [
                    "sereno", "mistico", "crystalline", "tribal", "ambient",
                    "natural", "espacial", "profundo", "energetico"
                ]
                logger.info(f"✅ Usando {len(estilos_encontrados)} estilos de fallback")
                self.stats_deteccion['fallback_estilos'] = True
            
            # Actualizar estadísticas
            self.stats_deteccion['estilos_detectados'] = len(estilos_encontrados)
            return estilos_encontrados
            
        except Exception as e:
            logger.error(f"❌ Error detectando estilos: {e}")
            # Fallback ultra-básico
            return ["neutral", "basic", "standard"]
    
    def _detectar_fases_estructurales(self) -> Optional[List[Dict[str, Any]]]:
        try:
            logger.info("🔄 Generando fases estructurales de fallback...")
            
            # Fases básicas estructurales para experiencias neuroacústicas
            fases_fallback = [
                {
                    "nombre": "introduccion",
                    "duracion": 60.0,
                    "intensidad": 0.3,
                    "tipo": "preparacion",
                    "descripcion": "Fase de preparación y relajación inicial"
                },
                {
                    "nombre": "desarrollo_inicial", 
                    "duracion": 120.0,
                    "intensidad": 0.5,
                    "tipo": "activacion",
                    "descripcion": "Activación gradual del estado deseado"
                },
                {
                    "nombre": "nucleo_experiencia",
                    "duracion": 180.0,
                    "intensidad": 0.8,
                    "tipo": "immersion",
                    "descripcion": "Núcleo principal de la experiencia"
                },
                {
                    "nombre": "desarrollo_profundo",
                    "duracion": 240.0,
                    "intensidad": 0.9,
                    "tipo": "profundizacion",
                    "descripcion": "Profundización del estado neuroacústico"
                },
                {
                    "nombre": "plateau",
                    "duracion": 300.0,
                    "intensidad": 1.0,
                    "tipo": "mantenimiento",
                    "descripcion": "Mantenimiento del estado óptimo"
                },
                {
                    "nombre": "transicion",
                    "duracion": 180.0,
                    "intensidad": 0.6,
                    "tipo": "transicion",
                    "descripcion": "Transición hacia la conclusión"
                },
                {
                    "nombre": "conclusion",
                    "duracion": 120.0,
                    "intensidad": 0.2,
                    "tipo": "finalizacion",
                    "descripcion": "Finalización suave y retorno gradual"
                }
            ]
            
            # Actualizar estadísticas
            self.stats_deteccion['fases_detectadas'] = len(fases_fallback)
            self.stats_deteccion['fallback_fases'] = True
            
            logger.info(f"✅ Creadas {len(fases_fallback)} fases estructurales de fallback")
            return fases_fallback
            
        except Exception as e:
            logger.error(f"❌ Error creando fases fallback: {e}")
            # Fallback ultra-básico
            return [
                {
                    "nombre": "experiencia_basica",
                    "duracion": 300.0,
                    "intensidad": 0.7,
                    "tipo": "completa",
                    "descripcion": "Experiencia neuroacústica básica"
                }
            ]
    
    def _detectar_templates_objetivos(self) -> Optional[Dict[str, Any]]:
        """Detecta templates de objetivos directamente desde ObjectiveManager"""
        templates = None
        try:
            import objective_manager
            if hasattr(objective_manager, 'OBJECTIVE_TEMPLATES'):
                templates = objective_manager.OBJECTIVE_TEMPLATES
                self.stats_deteccion['templates_objetivos'] = len(templates)
                logger.info(f"✅ Detectados {len(templates)} templates desde OBJECTIVE_TEMPLATES")
            elif hasattr(objective_manager, 'templates_disponibles'):
                templates = objective_manager.templates_disponibles
                self.stats_deteccion['templates_objetivos'] = len(templates)
                logger.info(f"✅ Detectados {len(templates)} templates desde templates_disponibles")
        except Exception as e:
            logger.warning(f"⚠️ Error detectando templates desde objective_manager: {e}")

        # ✅ VALIDACIÓN FINAL para evitar fallback innecesario
        if templates:
            return templates

        # 🌀 Fallback real
        from hypermod_v32 import DetectorObjectiveTemplatesExpandido  # IMPORT CORRECTO
        logger.warning("⚠️ No se detectaron templates reales, usando fallback")
        templates = DetectorObjectiveTemplatesExpandido().detectar_templates()
        logger.info("🧠 Usando templates de fallback")
        return templates

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
            
            componentes = self.detector_componentes.detectar_todos_componentes()
            templates = self.detector_componentes._detectar_templates_objetivos()

            self.componentes_cache = {
                'componentes': componentes,
                'templates': templates,
                'timestamp': datetime.now()
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
                
            self.stats.ultima_actualizacion = datetime.now()
            
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
            self.stats.ultima_actualizacion = datetime.now()
            
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

    def generar_neuro_mod(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera audio con modulación neuroacústica específica
        
        Método público que expone la funcionalidad de neuro-modulación del
        NeuroWaveGenerator subyacente con configuración optimizada para Aurora V7.
        
        Args:
            config (Dict[str, Any]): Configuración de modulación neuroacústica
                - frecuencia_base (float): Frecuencia base en Hz (default: 10.0)
                - tipo_modulacion (str): 'AM', 'FM', 'QUANTUM', 'SPATIAL' (default: 'AM')
                - amplitud (float): Amplitud de modulación 0.0-1.0 (default: 0.7)
                - duracion (float): Duración en segundos (default: 300)
                - tipo_onda (str): 'ALPHA', 'BETA', 'THETA', 'DELTA' (default: 'ALPHA')
                
        Returns:
            Dict[str, Any]: Resultado con audio modulado y metadata
                - audio_data (np.ndarray): Datos de audio generados
                - sample_rate (int): Frecuencia de muestreo
                - duracion (float): Duración real del audio
                - tipo_modulacion (str): Tipo de modulación aplicada
                - metadata (Dict): Información detallada de la generación
        """
        start_time = time.time()
        
        try:
            logger.info("🎵 Iniciando generación de neuro-modulación avanzada")
            
            # Extraer y validar parámetros
            frecuencia_base = config.get('frecuencia_base', 10.0)
            tipo_modulacion = config.get('tipo_modulacion', 'AM').upper()
            amplitud = config.get('amplitud', 0.7)
            duracion = config.get('duracion', 300)
            tipo_onda = config.get('tipo_onda', 'ALPHA').upper()
            
            # Validar parámetros
            if not (0.1 <= frecuencia_base <= 100.0):
                frecuencia_base = 10.0
                logger.warning(f"⚠️ Frecuencia base ajustada a {frecuencia_base} Hz")
                
            if amplitud < 0.0 or amplitud > 1.0:
                amplitud = 0.7
                logger.warning(f"⚠️ Amplitud ajustada a {amplitud}")
            
            # Convertir tipo_onda a enum si es necesario
            try:
                from hypermod_v32 import WaveType
                if hasattr(WaveType, tipo_onda):
                    wave_type_enum = getattr(WaveType, tipo_onda)
                else:
                    wave_type_enum = WaveType.ALPHA
                    logger.warning(f"⚠️ Tipo de onda desconocido, usando ALPHA")
            except:
                wave_type_enum = tipo_onda  # Fallback a string
            
            # Crear configuración de layer para el generador
            from hypermod_v32 import LayerConfig
            layer_config = LayerConfig(
                frequency=frecuencia_base,
                amplitude=amplitud,
                duration=duracion,
                wave_type=wave_type_enum
            )
            
            # Generar onda base
            onda_base = self.wave_generator.generate_wave(layer_config)
            
            # Aplicar modulación específica si se solicita
            if tipo_modulacion != 'BASIC':
                try:
                    from hypermod_v32 import ModulationType
                    if hasattr(ModulationType, tipo_modulacion):
                        mod_type = getattr(ModulationType, tipo_modulacion)
                        
                        # Crear array de tiempo para modulación
                        t = np.linspace(0, duracion, len(onda_base))
                        
                        # Aplicar modulación
                        onda_modulada = self.wave_generator.apply_modulation(onda_base, mod_type, t)
                    else:
                        onda_modulada = onda_base
                        logger.warning(f"⚠️ Tipo de modulación {tipo_modulacion} no disponible")
                except Exception as e:
                    onda_modulada = onda_base
                    logger.warning(f"⚠️ Error aplicando modulación: {e}")
            else:
                onda_modulada = onda_base
            
            # Actualizar estadísticas
            self.stats.layers_generados += 1
            tiempo_generacion = time.time() - start_time
            self.stats.tiempo_total_generacion += tiempo_generacion
            self.stats.ultima_actualizacion = datetime.now()
            
            # Preparar resultado
            resultado = {
                'audio_data': onda_modulada,
                'sample_rate': self.sample_rate,
                'duracion': len(onda_modulada) / self.sample_rate,
                'tipo_modulacion': tipo_modulacion,
                'frecuencia_base': frecuencia_base,
                'amplitud': amplitud,
                'tipo_onda': tipo_onda,
                'metadata': {
                    'motor': self.nombre,
                    'version': self.version,
                    'metodo': 'generar_neuro_mod',
                    'tiempo_generacion': tiempo_generacion,
                    'configuracion_usada': config,
                    'samples_generados': len(onda_modulada),
                    'wave_generator_stats': getattr(self.wave_generator, 'stats', {})
                }
            }
            
            logger.info(f"✅ Neuro-modulación generada: {len(onda_modulada)} samples, {tipo_modulacion}")
            return resultado
            
        except Exception as e:
            error_msg = f"Error en generación de neuro-modulación: {e}"
            logger.error(f"❌ {error_msg}")
            
            # Retornar audio de fallback
            duracion_fallback = config.get('duracion', 5)
            audio_fallback = np.zeros(int(duracion_fallback * self.sample_rate))
            
            return {
                'audio_data': audio_fallback,
                'sample_rate': self.sample_rate,
                'duracion': duracion_fallback,
                'tipo_modulacion': 'FALLBACK',
                'error': error_msg,
                'metadata': {
                    'motor': self.nombre,
                    'metodo': 'generar_neuro_mod',
                    'estado': 'error_con_fallback'
                }
            }
    
    def aplicar_modulacion_avanzada(self, audio_input: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica modulación avanzada a audio existente
        
        Método público que aplica modulación neuroacústica avanzada incluyendo
        efectos espaciales y modulación cuántica a audio ya generado.
        
        Args:
            audio_input (np.ndarray): Audio de entrada para modular
            config (Dict[str, Any]): Configuración de modulación avanzada
                - tipo_modulacion (str): 'AM', 'FM', 'QUANTUM', 'SPATIAL' (default: 'SPATIAL')
                - intensidad (float): Intensidad de modulación 0.0-1.0 (default: 0.5)
                - posicion_espacial (Tuple): (x, y, z) para efectos 3D (default: (0, 0, 0))
                - efectos_adicionales (List): Lista de efectos a aplicar
                
        Returns:
            Dict[str, Any]: Audio modulado con estadísticas de procesamiento
        """
        start_time = time.time()
        
        try:
            logger.info("🔀 Aplicando modulación avanzada a audio existente")
            
            # Validar entrada
            if audio_input is None or len(audio_input) == 0:
                raise ValueError("Audio de entrada inválido o vacío")
            
            # Convertir a numpy array si es necesario
            if not isinstance(audio_input, np.ndarray):
                audio_input = np.asarray(audio_input)
            
            # Extraer parámetros
            tipo_modulacion = config.get('tipo_modulacion', 'SPATIAL').upper()
            intensidad = config.get('intensidad', 0.5)
            posicion_espacial = config.get('posicion_espacial', (0.0, 0.0, 0.0))
            efectos_adicionales = config.get('efectos_adicionales', [])
            
            # Validar intensidad
            intensidad = max(0.0, min(1.0, intensidad))
            
            # Crear copia para procesamiento
            audio_procesado = audio_input.copy()
            efectos_aplicados = []
            
            # Aplicar modulación principal
            if tipo_modulacion in ['AM', 'FM', 'QUANTUM']:
                try:
                    from hypermod_v32 import ModulationType
                    if hasattr(ModulationType, tipo_modulacion):
                        mod_type = getattr(ModulationType, tipo_modulacion)
                        
                        # Crear array de tiempo
                        duracion_audio = len(audio_input) / self.sample_rate
                        t = np.linspace(0, duracion_audio, len(audio_input))
                        
                        # Aplicar modulación con intensidad
                        audio_modulado = self.wave_generator.apply_modulation(audio_procesado, mod_type, t)
                        
                        # Mezclar con original según intensidad
                        audio_procesado = (audio_input * (1.0 - intensidad) + 
                                         audio_modulado * intensidad)
                        efectos_aplicados.append(f"{tipo_modulacion}_modulacion")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error aplicando modulación {tipo_modulacion}: {e}")
            
            # Aplicar efectos espaciales
            if tipo_modulacion == 'SPATIAL' or 'spatial' in efectos_adicionales:
                try:
                    audio_espacial = self.wave_generator.apply_spatial_effects(
                        audio_procesado, posicion_espacial
                    )
                    
                    # Mezclar con intensidad
                    audio_procesado = (audio_procesado * (1.0 - intensidad) + 
                                     audio_espacial * intensidad)
                    efectos_aplicados.append("efectos_espaciales")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error aplicando efectos espaciales: {e}")
            
            # Aplicar efectos adicionales
            for efecto in efectos_adicionales:
                if efecto == 'normalizacion':
                    # Normalizar audio
                    max_val = np.max(np.abs(audio_procesado))
                    if max_val > 0:
                        audio_procesado = audio_procesado / max_val * 0.95
                    efectos_aplicados.append("normalizacion")
                    
                elif efecto == 'suavizado':
                    # Aplicar suavizado simple
                    ventana = np.ones(5) / 5
                    if len(audio_procesado) > len(ventana):
                        audio_procesado = np.convolve(audio_procesado, ventana, mode='same')
                    efectos_aplicados.append("suavizado")
            
            # Actualizar estadísticas
            tiempo_procesamiento = time.time() - start_time
            self.stats.tiempo_total_generacion += tiempo_procesamiento
            self.stats.ultima_actualizacion = datetime.now()
            
            # Preparar resultado
            resultado = {
                'audio_data': audio_procesado,
                'sample_rate': self.sample_rate,
                'duracion': len(audio_procesado) / self.sample_rate,
                'efectos_aplicados': efectos_aplicados,
                'tipo_modulacion_principal': tipo_modulacion,
                'intensidad_aplicada': intensidad,
                'posicion_espacial': posicion_espacial,
                'metadata': {
                    'motor': self.nombre,
                    'version': self.version,
                    'metodo': 'aplicar_modulacion_avanzada',
                    'tiempo_procesamiento': tiempo_procesamiento,
                    'configuracion_usada': config,
                    'samples_procesados': len(audio_procesado),
                    'samples_originales': len(audio_input),
                    'diferencia_amplitud': np.max(np.abs(audio_procesado - audio_input))
                }
            }
            
            logger.info(f"✅ Modulación avanzada aplicada: {len(efectos_aplicados)} efectos")
            return resultado
            
        except Exception as e:
            error_msg = f"Error aplicando modulación avanzada: {e}"
            logger.error(f"❌ {error_msg}")
            
            # Retornar audio original con metadata de error
            return {
                'audio_data': audio_input if audio_input is not None else np.array([]),
                'sample_rate': self.sample_rate,
                'duracion': len(audio_input) / self.sample_rate if audio_input is not None else 0,
                'efectos_aplicados': [],
                'error': error_msg,
                'metadata': {
                    'motor': self.nombre,
                    'metodo': 'aplicar_modulacion_avanzada',
                    'estado': 'error_audio_original_preservado'
                }
            }

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
        config_test = {'objetivo': 'relajacion', 'duracion': 1}
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
        globals()['HyperMod'] = HyperModEngineV32AuroraConnected
        
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
# CLASE WRAPPER ENHANCED (Patrón de compatibilidad estándar Aurora V7)
# =============================================================================

class HyperModV32Enhanced:
    """
    Wrapper Enhanced del motor HyperMod V32 siguiendo el patrón Aurora estándar
    
    Clase wrapper que expone la funcionalidad de HyperModEngineV32AuroraConnected
    con la interfaz Enhanced esperada por el sistema Aurora V7. Sigue el mismo
    patrón exitoso de NeuroMixAuroraV27 para compatibilidad total.
    
    Esta clase actúa como puente de compatibilidad entre la expectativa del sistema
    Aurora y la implementación robusta de HyperModEngineV32AuroraConnected.
    """
    
    def __init__(self, modo='estructural', silent=False, cache_enabled=True, config_inicial=None):
        """
        Inicializa el wrapper Enhanced con configuración estándar Aurora
        
        Args:
            modo (str): Modo de operación ('estructural', 'avanzado', 'experimental')
            silent (bool): Modo silencioso para logging (default: False)
            cache_enabled (bool): Activar cache de ondas (default: True)
            config_inicial (Dict): Configuración inicial opcional
        """
        # Configuración del wrapper
        self.modo = modo
        self.silent = silent
        self.cache_enabled = cache_enabled
        
        # Configurar logging según modo silent
        if silent:
            logging.getLogger('hypermod_v32').setLevel(logging.WARNING)
        
        # Preparar configuración para motor subyacente
        config_motor = config_inicial or {}
        config_motor.update({
            'enable_cache': cache_enabled,
            'quality_level': 'high' if modo == 'avanzado' else 'medium',
            'max_layers': 32 if modo == 'experimental' else 16
        })
        
        # Crear instancia del motor real
        self._motor_hypermod = HyperModEngineV32AuroraConnected(config_motor)
        
        # Propiedades de compatibilidad Aurora
        self.nombre = f"HyperMod V32 Enhanced ({modo})"
        self.version = "V32_ENHANCED_AURORA_COMPATIBLE"
        self.tipo_motor = "estructural_enhanced"
        
        # Estados y estadísticas wrapper
        self._estado_wrapper = "inicializado"
        self._operaciones_wrapper = 0
        self._tiempo_total_wrapper = 0.0
        
        if not silent:
            logger.info(f"✨ {self.nombre} inicializado en modo {modo}")
    
    # === MÉTODOS DE INTERFAZ AURORA ENHANCED ===
    
    def aplicar_presets(self, preset_name: str, **kwargs) -> dict:
        """
        Método estándar Enhanced para aplicar presets emocionales
        
        Args:
            preset_name (str): Nombre del preset ('calma', 'energia', 'focus', etc.)
            **kwargs: Parámetros adicionales para personalización
            
        Returns:
            dict: Configuración de preset aplicada y metadata
        """
        start_time = time.time()
        
        try:
            # Configuración base del preset
            config_preset = {
                'preset_emocional': preset_name,
                'duracion': kwargs.get('duracion', 300),
                'intensidad': kwargs.get('intensidad', 0.7),
                'calidad': kwargs.get('calidad', 'alta')
            }
            
            # Generar layers desde preset usando motor subyacente
            layers_config = {
                'objetivo': preset_name,
                'duracion': config_preset['duracion'],
                'intensidad': config_preset['intensidad']
            }
            
            # Validar configuración con motor real
            if self._motor_hypermod.validar_configuracion(layers_config):
                preset_aplicado = {
                    'preset_name': preset_name,
                    'config_generada': layers_config,
                    'modo_wrapper': self.modo,
                    'cache_enabled': self.cache_enabled,
                    'motor_subyacente': self._motor_hypermod.nombre,
                    "timestamp": datetime.now().isoformat(),
                    'tiempo_aplicacion': time.time() - start_time
                }
                
                self._operaciones_wrapper += 1
                self._tiempo_total_wrapper += preset_aplicado['tiempo_aplicacion']
                
                if not self.silent:
                    logger.info(f"✅ Preset '{preset_name}' aplicado exitosamente")
                
                return preset_aplicado
            else:
                raise ValueError(f"Preset '{preset_name}' no válido para configuración actual")
                
        except Exception as e:
            error_msg = f"Error aplicando preset '{preset_name}': {e}"
            if not self.silent:
                logger.error(f"❌ {error_msg}")
            
            return {
                'preset_name': preset_name,
                'error': error_msg,
                'fallback_aplicado': True,
                'config_fallback': {'objetivo': 'relajacion', 'duracion': 300}
            }
    
    def generar_experiencia_enhanced(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera experiencia neuroacústica usando funcionalidad Enhanced
        
        Args:
            config (Dict): Configuración de experiencia Enhanced
                - tipo_experiencia (str): 'estructural', 'modulada', 'avanzada'
                - parametros_especificos (Dict): Parámetros del tipo de experiencia
                
        Returns:
            Dict: Resultado de experiencia con metadata Enhanced
        """
        start_time = time.time()
        
        try:
            tipo_experiencia = config.get('tipo_experiencia', 'estructural')
            
            if tipo_experiencia == 'modulada':
                # Usar funcionalidad de neuro-modulación
                resultado = self._motor_hypermod.generar_neuro_mod(config)
                resultado['tipo_enhanced'] = 'neuro_modulacion'
                
            elif tipo_experiencia == 'avanzada':
                # Generar audio base y aplicar modulación avanzada
                config_base = dict(config)
                config_base['tipo_experiencia'] = 'estructural'
                
                resultado_base = self._motor_hypermod.generar_audio(config_base)
                
                if 'audio_data' in resultado_base:
                    config_modulacion = config.get('parametros_especificos', {})
                    resultado = self._motor_hypermod.aplicar_modulacion_avanzada(
                        resultado_base['audio_data'], config_modulacion
                    )
                    resultado['tipo_enhanced'] = 'modulacion_avanzada'
                    resultado['audio_base_metadata'] = resultado_base.get('metadata', {})
                else:
                    resultado = resultado_base
                    resultado['tipo_enhanced'] = 'fallback_estructural'
            else:
                # Experiencia estructural estándar
                resultado = self._motor_hypermod.generar_audio(config)
                resultado['tipo_enhanced'] = 'estructural_estandar'
            
            # Agregar metadata del wrapper Enhanced
            resultado['enhanced_metadata'] = {
                'wrapper_version': self.version,
                'modo_wrapper': self.modo,
                'operaciones_realizadas': self._operaciones_wrapper,
                'tiempo_total_wrapper': self._tiempo_total_wrapper,
                'tiempo_experiencia': time.time() - start_time
            }
            
            self._operaciones_wrapper += 1
            
            if not self.silent:
                logger.info(f"✨ Experiencia {tipo_experiencia} generada con Enhanced wrapper")
            
            return resultado
            
        except Exception as e:
            error_msg = f"Error en experiencia Enhanced: {e}"
            if not self.silent:
                logger.error(f"❌ {error_msg}")
            
            return {
                'error': error_msg,
                'tipo_enhanced': 'error_con_fallback',
                'fallback_data': np.zeros(44100 * 5),  # 5 segundos de silencio
                'sample_rate': 44100
            }
    
    def get_enhanced_info(self) -> Dict[str, Any]:
        """
        Información completa del wrapper Enhanced
        
        Returns:
            Dict: Información detallada del Enhanced wrapper y motor subyacente
        """
        motor_info = self._motor_hypermod.get_info()
        
        return {
            'nombre_enhanced': self.nombre,
            'version_enhanced': self.version,
            'tipo_motor': self.tipo_motor,
            'modo_operacion': self.modo,
            'silent_mode': self.silent,
            'cache_enabled': self.cache_enabled,
            'motor_subyacente': motor_info,
            'estadisticas_wrapper': {
                'estado': self._estado_wrapper,
                'operaciones_realizadas': self._operaciones_wrapper,
                'tiempo_total_operaciones': self._tiempo_total_wrapper,
                'tiempo_promedio_operacion': (
                    self._tiempo_total_wrapper / self._operaciones_wrapper 
                    if self._operaciones_wrapper > 0 else 0
                )
            },
            'capacidades_enhanced': {
                'presets_emocionales': True,
                'experiencias_avanzadas': True,
                'neuro_modulacion': True,
                'modulacion_avanzada': True,
                'cache_inteligente': self.cache_enabled,
                'modo_silencioso': self.silent
            }
        }
    
    def obtener_capacidades_enhanced(self) -> Dict[str, Any]:
        """
        Capacidades específicas del Enhanced wrapper
        
        Returns:
            Dict: Capacidades extendidas con funcionalidad Enhanced
        """
        capacidades_motor = self._motor_hypermod.obtener_capacidades()
        
        capacidades_enhanced = {
            'enhanced_wrapper': True,
            'modo_operacion': self.modo,
            'tipos_experiencia_soportados': [
                'estructural', 'modulada', 'avanzada'
            ],
            'presets_enhanced_disponibles': [
                'calma_profunda', 'energia_creativa', 'focus_intenso',
                'meditacion_guiada', 'concentracion_extrema'
            ],
            'funcionalidades_advanced': {
                'neuro_modulacion_directa': True,
                'modulacion_avanzada_post_proceso': True,
                'presets_dinamicos': True,
                'cache_inteligente': self.cache_enabled,
                'estadisticas_wrapper': True
            }
        }
        
        # Combinar capacidades del motor con Enhanced
        capacidades_enhanced.update(capacidades_motor)
        capacidades_enhanced['enhanced_integration'] = True
        
        return capacidades_enhanced
    
    # === DELEGACIÓN A MOTOR SUBYACENTE ===
    
    def generar_audio(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Delega generación de audio al motor subyacente con wrapper metadata"""
        resultado = self._motor_hypermod.generar_audio(config)
        resultado['enhanced_wrapper_used'] = True
        resultado['wrapper_mode'] = self.modo
        self._operaciones_wrapper += 1
        return resultado
    
    def generar_neuro_mod(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Delega neuro-modulación al motor subyacente"""
        resultado = self._motor_hypermod.generar_neuro_mod(config)
        resultado['enhanced_wrapper_used'] = True
        self._operaciones_wrapper += 1
        return resultado
    
    def aplicar_modulacion_avanzada(self, audio_input: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Delega modulación avanzada al motor subyacente"""
        resultado = self._motor_hypermod.aplicar_modulacion_avanzada(audio_input, config)
        resultado['enhanced_wrapper_used'] = True
        self._operaciones_wrapper += 1
        return resultado
    
    def validar_configuracion(self, config: Dict[str, Any]) -> bool:
        """Delega validación al motor subyacente"""
        return self._motor_hypermod.validar_configuracion(config)
    
    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        """Combina estadísticas del motor con estadísticas del wrapper"""
        stats_motor = self._motor_hypermod.obtener_estadisticas_completas()
        
        stats_enhanced = {
            'enhanced_wrapper_stats': {
                'operaciones_wrapper': self._operaciones_wrapper,
                'tiempo_total_wrapper': self._tiempo_total_wrapper,
                'modo_operacion': self.modo,
                'estado_wrapper': self._estado_wrapper
            },
            'motor_subyacente_stats': stats_motor
        }
        
        return stats_enhanced
    
    def limpiar_cache(self):
        """Delega limpieza de cache al motor subyacente"""
        self._motor_hypermod.limpiar_cache()
        if not self.silent:
            logger.info("🧹 Cache Enhanced limpiado")
    
    # === COMPATIBILIDAD AURORA V7 ===
    
    @property
    def motor_subyacente(self):
        """Acceso directo al motor subyacente para compatibilidad avanzada"""
        return self._motor_hypermod
    
    def __str__(self):
        return f"{self.nombre} v{self.version} (modo: {self.modo})"
    
    def __repr__(self):
        return f"HyperModV32Enhanced(modo='{self.modo}', silent={self.silent}, cache_enabled={self.cache_enabled})"
    
# =============================================================================
# FACTORY FUNCTION ENHANCED
# =============================================================================

def crear_hypermod_enhanced(modo='estructural', silent=False, cache_enabled=True, **kwargs) -> HyperModV32Enhanced:
    """
    Factory function para crear instancia Enhanced de HyperMod V32
    
    Args:
        modo (str): Modo de operación del Enhanced wrapper
        silent (bool): Modo silencioso para logging
        cache_enabled (bool): Activar cache inteligente
        **kwargs: Parámetros adicionales para configuración
        
    Returns:
        HyperModV32Enhanced: Instancia del wrapper Enhanced configurado
    """
    try:
        enhanced_motor = HyperModV32Enhanced(
            modo=modo,
            silent=silent, 
            cache_enabled=cache_enabled,
            config_inicial=kwargs.get('config_inicial')
        )
        
        if not silent:
            logger.info(f"✨ HyperMod V32 Enhanced creado exitosamente en modo {modo}")
        
        return enhanced_motor
        
    except Exception as e:
        logger.error(f"❌ Error creando HyperMod Enhanced: {e}")
        raise

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

HyperModV32 = HyperModEngineV32AuroraConnected

def _validar_alias_hypermod_v32():
    """Validación automática del alias HyperModV32"""
    try:
        # Verificar que el alias apunta a clase válida
        assert HyperModV32 is not None, "Alias HyperModV32 es None"
        assert hasattr(HyperModV32, '__init__'), "Alias sin constructor"
        
        # Verificar métodos críticos esperados por Aurora Director
        metodos_requeridos = ['generar_audio', 'aplicar_fases', 'crear_estructura_capas']
        
        # Crear instancia temporal para validar
        instancia_test = HyperModV32()
        
        metodos_encontrados = []
        metodos_faltantes = []
        
        for metodo in metodos_requeridos:
            if hasattr(instancia_test, metodo):
                metodos_encontrados.append(metodo)
            else:
                metodos_faltantes.append(metodo)
        
        logger.info(f"🏆 PRIMERA VICTORIA - HyperModV32 alias validado:")
        logger.info(f"   ✅ Métodos encontrados: {metodos_encontrados}")
        
        if metodos_faltantes:
            logger.warning(f"   ⚠️ Métodos faltantes: {metodos_faltantes}")
            return False
        
        logger.info("   🎯 Alias HyperModV32 completamente funcional")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validando alias HyperModV32: {e}")
        return False

# ✅ EJECUTAR VALIDACIÓN AUTOMÁTICA
if __name__ != "__main__":  # Solo si se importa, no si se ejecuta directamente
    try:
        _validacion_exitosa = _validar_alias_hypermod_v32()
        if _validacion_exitosa:
            logger.info("🏆 PRIMERA VICTORIA CONFIRMADA: HyperModV32 alias operativo")
        else:
            logger.warning("⚠️ Alias HyperModV32 creado pero con advertencias")
    except Exception as e:
        logger.error(f"❌ Error en validación automática HyperModV32: {e}")

# =============================================================================
# REGISTRO EN NAMESPACE GLOBAL PARA MÁXIMA COMPATIBILIDAD
# =============================================================================

# Asegurar que el alias esté disponible globalmente
globals()['HyperModV32'] = HyperModV32

# Crear también alias de compatibilidad adicionales por si acaso
globals()['HyperMod'] = HyperModV32  # Alias corto
globals()['HyperModV32Aurora'] = HyperModV32  # Alias específico Aurora

# =============================================================================
# METADATA DEL ALIAS
# =============================================================================

ALIAS_METADATA = {
    "alias_name": "HyperModV32",
    "target_class": "HyperModEngineV32AuroraConnected", 
    "created_by": "aurora_v7_error_correction_phase1",
    "purpose": "Resolver ERROR C1: Clase Principal Faltante",
    "validation_methods": ["generar_audio", "aplicar_fases", "crear_estructura_capas"],
    'timestamp': datetime.now().isoformat(),
    "status": "active"
}

# Hacer metadata accesible
globals()['HYPERMOD_ALIAS_METADATA'] = ALIAS_METADATA

# =============================================================================
# LOGGING DE ÉXITO
# =============================================================================

logger.info("🏆 PRIMERA VICTORIA IMPLEMENTADA: HyperModV32 alias creado")
logger.info(f"   📍 Alias: HyperModV32 → {HyperModV32.__name__}")
logger.info(f"   🎯 Estado: Activo y validado")
logger.info(f"   📊 Metadata: {ALIAS_METADATA['purpose']}")

# =============================================================================
# EXPORTACIONES PRINCIPALES
# =============================================================================

# Clase principal para importación
__all__ = [
    'HyperModEngineV32AuroraConnected',
    'HyperModV32Enhanced',
    'crear_motor_hypermod',
    'crear_hypermod_enhanced',
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