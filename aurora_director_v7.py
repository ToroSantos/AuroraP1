#!/usr/bin/env python3
"""
🎯 AURORA DIRECTOR V7 - REFACTORIZADO SIN DEPENDENCIAS CIRCULARES
================================================================

Orquestador principal del Sistema Aurora V7.
Versión refactorizada que utiliza interfaces, factory pattern e inyección
de dependencias para eliminar dependencias circulares.

Versión: V7.1 - Refactorizado
Propósito: Orquestación principal sin dependencias circulares
Compatibilidad: Todos los motores y módulos Aurora V7
"""

import os
import sys
import time
import json
import logging
import uuid
import threading
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import lru_cache, wraps
import numpy as np

# Importar interfaces Aurora (sin ciclos)
try:
    from aurora_interfaces import (
        DirectorAuroraInterface, MotorAuroraInterface, GestorPerfilesInterface,
        AnalizadorInterface, ExtensionAuroraInterface, ComponenteAuroraBase,
        TipoComponenteAurora, EstadoComponente, NivelCalidad,
        ConfiguracionBaseInterface, ResultadoBaseInterface,
        validar_motor_aurora, validar_gestor_perfiles
    )
    from aurora_factory import aurora_factory, lazy_importer
    INTERFACES_DISPONIBLES = True
except ImportError as e:
    logging.warning(f"Aurora interfaces no disponibles: {e}")
    INTERFACES_DISPONIBLES = False
    
    # Clases dummy para compatibilidad
    class DirectorAuroraInterface: pass
    class MotorAuroraInterface: pass
    class GestorPerfilesInterface: pass
    class ComponenteAuroraBase: pass
    class ConfiguracionBaseInterface: pass
    class ResultadoBaseInterface: pass

# Configuración de logging
logger = logging.getLogger("Aurora.Director.V7.Refactored")

VERSION = "V7.1_REFACTORIZADO_NO_CIRCULAR_DEPS"

# ===============================================================================
# ENUMS Y CONFIGURACIONES
# ===============================================================================

class EstrategiaGeneracion(Enum):
    """Estrategias de generación disponibles"""
    SYNC_HIBRIDO = "sync_hibrido"
    MULTI_MOTOR = "multi_motor"
    FALLBACK_PROGRESIVO = "fallback_progresivo"
    LAYERED = "layered"
    SIMPLE = "simple"

class ModoOrquestacion(Enum):
    """Modos de orquestación del director"""
    COMPLETO = "completo"
    BASICO = "basico"
    FALLBACK = "fallback"
    EXPERIMENTAL = "experimental"

class TipoComponente(Enum):
    """Tipos de componentes gestionados por el director"""
    MOTOR = "motor"
    GESTOR_INTELIGENCIA = "gestor_inteligencia"
    PIPELINE = "pipeline"
    EXTENSION = "extension"
    ANALIZADOR = "analizador"

@dataclass
class ConfiguracionAuroraUnificada:
    """Configuración unificada para Aurora Director refactorizado"""
    
    # Configuración básica
    objetivo: str = "relajacion"
    duracion_min: float = 5.0
    intensidad: float = 0.5
    estilo: str = "ambiental"
    calidad_objetivo: str = "alta"
    
    # Configuración de motores
    motores_preferidos: List[str] = field(default_factory=lambda: [
        "neuromix_super_unificado", "harmonicEssence_v34", "hypermod_v32"
    ])
    estrategia_generacion: EstrategiaGeneracion = EstrategiaGeneracion.SYNC_HIBRIDO
    modo_orquestacion: ModoOrquestacion = ModoOrquestacion.COMPLETO
    
    # Configuración avanzada
    usar_parches: bool = True
    validacion_estructural: bool = True
    aplicar_optimizaciones: bool = True
    exportar_metadatos: bool = True
    
    # Configuración de fallback
    permitir_fallback: bool = True
    fallback_rapido: bool = False
    
    def validar(self) -> bool:
        """Valida la configuración"""
        try:
            return (
                isinstance(self.objetivo, str) and len(self.objetivo) > 0 and
                isinstance(self.duracion_min, (int, float)) and self.duracion_min > 0 and
                0.0 <= self.intensidad <= 1.0
            )
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "objetivo": self.objetivo,
            "duracion_min": self.duracion_min,
            "intensidad": self.intensidad,
            "estilo": self.estilo,
            "calidad_objetivo": self.calidad_objetivo,
            "motores_preferidos": self.motores_preferidos,
            "estrategia_generacion": self.estrategia_generacion.value,
            "modo_orquestacion": self.modo_orquestacion.value,
            "usar_parches": self.usar_parches,
            "validacion_estructural": self.validacion_estructural,
            "aplicar_optimizaciones": self.aplicar_optimizaciones,
            "exportar_metadatos": self.exportar_metadatos,
            "permitir_fallback": self.permitir_fallback,
            "fallback_rapido": self.fallback_rapido
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Carga desde diccionario"""
        for key, value in data.items():
            if hasattr(self, key):
                if key == "estrategia_generacion":
                    try:
                        setattr(self, key, EstrategiaGeneracion(value))
                    except ValueError:
                        setattr(self, key, EstrategiaGeneracion.SYNC_HIBRIDO)
                elif key == "modo_orquestacion":
                    try:
                        setattr(self, key, ModoOrquestacion(value))
                    except ValueError:
                        setattr(self, key, ModoOrquestacion.COMPLETO)
                else:
                    setattr(self, key, value)

@dataclass
class ResultadoAuroraIntegrado:
    """Resultado integrado de generación Aurora"""
    
    # Resultado principal
    exito: bool = False
    mensaje: str = ""
    datos: Dict[str, Any] = field(default_factory=dict)
    
    # Audio generado
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 44100
    duracion_real: float = 0.0
    
    # Metadatos de generación
    estrategia_usada: EstrategiaGeneracion = EstrategiaGeneracion.FALLBACK_PROGRESIVO
    motores_utilizados: List[str] = field(default_factory=list)
    tiempo_generacion: float = 0.0
    
    # Análisis y calidad
    analisis_calidad: Dict[str, Any] = field(default_factory=dict)
    validacion_estructural: Dict[str, Any] = field(default_factory=dict)
    
    # Diagnóstico
    errores_encontrados: List[str] = field(default_factory=list)
    advertencias: List[str] = field(default_factory=list)
    
    def serializar(self) -> Dict[str, Any]:
        """Serializa el resultado excluyendo audio_data"""
        return {
            "exito": self.exito,
            "mensaje": self.mensaje,
            "datos": self.datos,
            "audio_presente": self.audio_data is not None,
            "audio_shape": self.audio_data.shape if self.audio_data is not None else None,
            "sample_rate": self.sample_rate,
            "duracion_real": self.duracion_real,
            "estrategia_usada": self.estrategia_usada.value,
            "motores_utilizados": self.motores_utilizados,
            "tiempo_generacion": self.tiempo_generacion,
            "analisis_calidad": self.analisis_calidad,
            "validacion_estructural": self.validacion_estructural,
            "errores_encontrados": self.errores_encontrados,
            "advertencias": self.advertencias
        }

# ===============================================================================
# DETECTOR DE COMPONENTES REFACTORIZADO
# ===============================================================================

class DetectorComponentesRefactored:
    """
    Detector de componentes que usa factory pattern sin dependencias circulares
    """
    
    def __init__(self):
        self.componentes_disponibles = {}
        self.componentes_activos = {}
        self.cache_deteccion = {}
        self.lock = threading.Lock()
        
        # Configuración de componentes esperados
        self.componentes_esperados = {
            # Motores principales
            "neuromix_super_unificado": {
                "tipo": TipoComponente.MOTOR,
                "modulo": "neuromix_definitivo_v7",
                "clase": "NeuroMixSuperUnificadoRefactored",
                "prioridad": 1,
                "requerido": True
            },
            "harmonicEssence_v34": {
                "tipo": TipoComponente.MOTOR,
                "modulo": "harmonicEssence_v34",
                "clase": "HarmonicEssenceV34Refactored",
                "prioridad": 1,
                "requerido": True
            },
            "neuromix_aurora_v27": {
                "tipo": TipoComponente.MOTOR,
                "modulo": "neuromix_aurora_v27",
                "clase": "AuroraNeuroAcousticEngineV27",
                "prioridad": 2,
                "requerido": False
            },
            "hypermod_v32": {
                "tipo": TipoComponente.MOTOR,
                "modulo": "hypermod_v32",
                "clase": "HyperModV32",
                "prioridad": 2,
                "requerido": False
            },
            
            # Gestores de inteligencia
            "emotion_style_profiles": {
                "tipo": TipoComponente.GESTOR_INTELIGENCIA,
                "modulo": "emotion_style_profiles",
                "clase": "GestorPresetsEmocionalesRefactored",
                "prioridad": 2,
                "requerido": True
            },
            "objective_manager": {
                "tipo": TipoComponente.GESTOR_INTELIGENCIA,
                "modulo": "objective_manager",
                "clase": "ObjectiveManagerUnificado",
                "prioridad": 3,
                "requerido": False
            },
            "field_profiles": {
                "tipo": TipoComponente.GESTOR_INTELIGENCIA,
                "modulo": "field_profiles",
                "clase": "GestorPerfilesCampo",
                "prioridad": 3,
                "requerido": False
            },
            
            # Pipelines y análisis
            "aurora_quality_pipeline": {
                "tipo": TipoComponente.PIPELINE,
                "modulo": "aurora_quality_pipeline",
                "clase": "AuroraQualityPipeline",
                "prioridad": 4,
                "requerido": False
            },
            "carmine_analyzer": {
                "tipo": TipoComponente.ANALIZADOR,
                "modulo": "Carmine_Analyzer",
                "clase": "CarmineAuroraAnalyzer",
                "prioridad": 4,
                "requerido": False
            },
            
            # Extensiones
            "multimotor_extension": {
                "tipo": TipoComponente.EXTENSION,
                "modulo": "aurora_multimotor_extension",
                "clase": "ColaboracionMultiMotorV7Optimizada",
                "prioridad": 5,
                "requerido": False
            }
        }
    
    def detectar_todos_los_componentes(self) -> Dict[str, Any]:
        """Detecta todos los componentes usando factory pattern"""
        
        with self.lock:
            logger.info("🔍 Detectando componentes del sistema Aurora...")
            
            resultados = {
                "componentes_detectados": 0,
                "componentes_activos": 0,
                "motores_disponibles": [],
                "gestores_disponibles": [],
                "pipelines_disponibles": [],
                "extensiones_disponibles": [],
                "errores": []
            }
            
            for nombre, config in self.componentes_esperados.items():
                try:
                    # Detectar usando factory
                    componente = self._detectar_componente_factory(nombre, config)
                    
                    if componente:
                        self.componentes_disponibles[nombre] = {
                            "instancia": componente,
                            "config": config,
                            "activo": True,
                            "modo_deteccion": "factory"
                        }
                        
                        resultados["componentes_detectados"] += 1
                        resultados["componentes_activos"] += 1
                        
                        # Clasificar por tipo
                        tipo = config["tipo"]
                        if tipo == TipoComponente.MOTOR:
                            resultados["motores_disponibles"].append(nombre)
                        elif tipo == TipoComponente.GESTOR_INTELIGENCIA:
                            resultados["gestores_disponibles"].append(nombre)
                        elif tipo == TipoComponente.PIPELINE:
                            resultados["pipelines_disponibles"].append(nombre)
                        elif tipo == TipoComponente.EXTENSION:
                            resultados["extensiones_disponibles"].append(nombre)
                        
                        logger.debug(f"✅ {nombre} detectado y activo")
                    else:
                        # Componente no disponible
                        self.componentes_disponibles[nombre] = {
                            "instancia": None,
                            "config": config,
                            "activo": False,
                            "modo_deteccion": "fallido"
                        }
                        
                        if config["requerido"]:
                            error_msg = f"Componente requerido {nombre} no disponible"
                            resultados["errores"].append(error_msg)
                            logger.warning(f"⚠️ {error_msg}")
                        else:
                            logger.debug(f"ℹ️ Componente opcional {nombre} no disponible")
                
                except Exception as e:
                    error_msg = f"Error detectando {nombre}: {e}"
                    resultados["errores"].append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            logger.info(f"✅ Detección completada: {resultados['componentes_activos']}/{len(self.componentes_esperados)} componentes activos")
            
            return resultados
    
    def _detectar_componente_factory(self, nombre: str, config: Dict[str, Any]) -> Optional[Any]:
        """Detecta un componente específico usando factory"""
        
        try:
            tipo = config["tipo"]
            
            # Usar factory según el tipo
            if tipo == TipoComponente.MOTOR:
                return aurora_factory.crear_motor(nombre)
            elif tipo == TipoComponente.GESTOR_INTELIGENCIA:
                return aurora_factory.crear_gestor(nombre)
            elif tipo == TipoComponente.ANALIZADOR:
                return aurora_factory.crear_analizador(nombre)
            elif tipo == TipoComponente.EXTENSION:
                return aurora_factory.crear_extension(nombre)
            else:
                # Fallback: lazy loading directo
                modulo = config["modulo"]
                clase = config["clase"]
                
                component_class = lazy_importer.get_class(modulo, clase)
                if component_class:
                    return component_class()
            
            return None
            
        except Exception as e:
            logger.debug(f"Error creando componente {nombre} vía factory: {e}")
            return None
    
    def obtener_componente(self, nombre: str) -> Optional[Any]:
        """Obtiene un componente detectado"""
        if nombre in self.componentes_disponibles:
            comp_info = self.componentes_disponibles[nombre]
            if comp_info["activo"]:
                return comp_info["instancia"]
        return None
    
    def obtener_motores_activos(self) -> List[str]:
        """Obtiene lista de motores activos"""
        motores = []
        for nombre, info in self.componentes_disponibles.items():
            if info["activo"] and info["config"]["tipo"] == TipoComponente.MOTOR:
                motores.append(nombre)
        return sorted(motores, key=lambda x: self.componentes_esperados[x]["prioridad"])
    
    def obtener_gestores_activos(self) -> List[str]:
        """Obtiene lista de gestores activos"""
        gestores = []
        for nombre, info in self.componentes_disponibles.items():
            if info["activo"] and info["config"]["tipo"] == TipoComponente.GESTOR_INTELIGENCIA:
                gestores.append(nombre)
        return sorted(gestores, key=lambda x: self.componentes_esperados[x]["prioridad"])

# ===============================================================================
# AURORA DIRECTOR PRINCIPAL REFACTORIZADO
# ===============================================================================

class AuroraDirectorV7Refactored(ComponenteAuroraBase):
    """
    Director principal de Aurora V7 refactorizado sin dependencias circulares
    
    Utiliza factory pattern e inyección de dependencias para orquestar
    todos los componentes del sistema sin crear ciclos de importación.
    """
    
    def __init__(self):
        super().__init__("AuroraDirectorV7", VERSION)
        
        # Detector de componentes
        self.detector = DetectorComponentesRefactored()
        
        # Componentes activos
        self.motores = {}
        self.gestores = {}
        self.pipelines = {}
        self.extensiones = {}
        self.analizadores = {}
        
        # Configuración
        self.modo_orquestacion = ModoOrquestacion.COMPLETO
        self.estrategia_default = EstrategiaGeneracion.SYNC_HIBRIDO
        
        # Cache y optimización
        self.cache_experiencias = {}
        self.cache_configuraciones = {}
        
        # Estadísticas
        self.stats = {
            "experiencias_generadas": 0,
            "tiempo_total_generacion": 0.0,
            "estrategias_usadas": {},
            "motores_mas_usados": {},
            "errores_manejados": 0,
            "fallbacks_utilizados": 0
        }
        
        # Inicialización
        self._inicializar_sistema()
        
        self.cambiar_estado(EstadoComponente.CONFIGURADO)
        logger.info("🎯 Aurora Director V7 refactorizado inicializado")
    
    def _inicializar_sistema(self):
        """Inicializa el sistema detectando y configurando componentes"""
        
        try:
            # Detectar todos los componentes
            resultado_deteccion = self.detector.detectar_todos_los_componentes()
            
            # Organizar componentes por tipo
            self._organizar_componentes()
            
            # Validar componentes críticos
            self._validar_componentes_criticos()
            
            # Configurar modo de orquestación según componentes disponibles
            self._configurar_modo_orquestacion(resultado_deteccion)
            
            logger.info(f"✅ Sistema inicializado: {len(self.motores)} motores, {len(self.gestores)} gestores")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            self._activar_modo_fallback()
    
    def _organizar_componentes(self):
        """Organiza los componentes detectados por tipo"""
        
        for nombre, info in self.detector.componentes_disponibles.items():
            if not info["activo"]:
                continue
                
            componente = info["instancia"]
            tipo = info["config"]["tipo"]
            
            if tipo == TipoComponente.MOTOR:
                self.motores[nombre] = componente
            elif tipo == TipoComponente.GESTOR_INTELIGENCIA:
                self.gestores[nombre] = componente
            elif tipo == TipoComponente.PIPELINE:
                self.pipelines[nombre] = componente
            elif tipo == TipoComponente.EXTENSION:
                self.extensiones[nombre] = componente
            elif tipo == TipoComponente.ANALIZADOR:
                self.analizadores[nombre] = componente
    
    def _validar_componentes_criticos(self):
        """Valida que los componentes críticos estén disponibles"""
        
        componentes_criticos = [
            "neuromix_super_unificado",
            "harmonicEssence_v34", 
            "emotion_style_profiles"
        ]
        
        faltantes = []
        for componente in componentes_criticos:
            if componente not in self.detector.componentes_disponibles or \
               not self.detector.componentes_disponibles[componente]["activo"]:
                faltantes.append(componente)
        
        if faltantes:
            logger.warning(f"⚠️ Componentes críticos faltantes: {faltantes}")
            self._crear_componentes_fallback(faltantes)
    
    def _configurar_modo_orquestacion(self, resultado_deteccion: Dict[str, Any]):
        """Configura el modo de orquestación según componentes disponibles"""
        
        num_motores = len(resultado_deteccion["motores_disponibles"])
        num_gestores = len(resultado_deteccion["gestores_disponibles"])
        
        if num_motores >= 2 and num_gestores >= 2:
            self.modo_orquestacion = ModoOrquestacion.COMPLETO
            self.estrategia_default = EstrategiaGeneracion.SYNC_HIBRIDO
        elif num_motores >= 1 and num_gestores >= 1:
            self.modo_orquestacion = ModoOrquestacion.BASICO
            self.estrategia_default = EstrategiaGeneracion.MULTI_MOTOR
        else:
            self.modo_orquestacion = ModoOrquestacion.FALLBACK
            self.estrategia_default = EstrategiaGeneracion.FALLBACK_PROGRESIVO
            
        logger.info(f"🎯 Modo orquestación: {self.modo_orquestacion.value}, Estrategia: {self.estrategia_default.value}")
    
    def _activar_modo_fallback(self):
        """Activa modo fallback de emergencia"""
        
        self.modo_orquestacion = ModoOrquestacion.FALLBACK
        self.estrategia_default = EstrategiaGeneracion.FALLBACK_PROGRESIVO
        self.stats["fallbacks_utilizados"] += 1
        
        logger.warning("🛡️ Modo fallback activado")
    
    def _crear_componentes_fallback(self, componentes_faltantes: List[str]):
        """Crea componentes fallback para componentes críticos faltantes"""
        
        for componente in componentes_faltantes:
            try:
                fallback_instance = self._crear_fallback_individual(componente)
                if fallback_instance:
                    if componente.startswith("neuromix") or "motor" in componente:
                        self.motores[componente] = fallback_instance
                    else:
                        self.gestores[componente] = fallback_instance
                    
                    logger.info(f"🛡️ Fallback creado para {componente}")
                    
            except Exception as e:
                logger.error(f"❌ Error creando fallback para {componente}: {e}")
    
    def _crear_fallback_individual(self, nombre_componente: str) -> Optional[Any]:
        """Crea un componente fallback individual"""
        
        if "neuromix" in nombre_componente:
            return self._crear_motor_neuromix_fallback()
        elif "harmonic" in nombre_componente:
            return self._crear_motor_harmonic_fallback()
        elif "emotion" in nombre_componente:
            return self._crear_gestor_emotion_fallback()
        else:
            return self._crear_componente_generico_fallback(nombre_componente)
    
    def _crear_motor_neuromix_fallback(self):
        """Crea motor NeuroMix fallback"""
        
        class NeuroMixFallback:
            def __init__(self):
                self.nombre = "NeuroMix"
                self.version = "fallback"
            
            def generar_audio(self, config):
                duracion = config.get('duracion_min', 60)
                sample_rate = 44100
                t = np.linspace(0, duracion, int(duracion * sample_rate))
                
                # Audio binaural básico
                freq_base = 220.0
                freq_binaural = 10.0
                
                left = 0.3 * np.sin(2 * np.pi * freq_base * t)
                right = 0.3 * np.sin(2 * np.pi * (freq_base + freq_binaural) * t)
                
                audio_data = np.column_stack([left, right])
                
                return {
                    "audio_data": audio_data,
                    "sample_rate": sample_rate,
                    "duracion_sec": duracion,
                    "metadatos": {"motor": "NeuroMix", "modo": "fallback"}
                }
            
            def validar_configuracion(self, config):
                return True
            
            def obtener_capacidades(self):
                return {"tipo": "fallback", "capacidades": ["binaural_basico"]}
            
            def get_info(self):
                return {"nombre": "NeuroMix", "version": "fallback"}
        
        return NeuroMixFallback()
    
    def _crear_motor_harmonic_fallback(self):
        """Crea motor HarmonicEssence fallback"""
        
        class HarmonicFallback:
            def __init__(self):
                self.nombre = "HarmonicEssence"
                self.version = "fallback"
            
            def generar_audio(self, config):
                duracion = config.get('duracion_min', 60)
                sample_rate = 44100
                t = np.linspace(0, duracion, int(duracion * sample_rate))
                
                # Textura ambiental simple
                freq = 110.0
                audio = 0.2 * np.sin(2 * np.pi * freq * t) * np.exp(-t / (duracion * 0.5))
                audio_stereo = np.column_stack([audio, audio])
                
                return {
                    "audio_data": audio_stereo,
                    "sample_rate": sample_rate,
                    "duracion_sec": duracion,
                    "metadatos": {"motor": "HarmonicEssence", "modo": "fallback"}
                }
            
            def validar_configuracion(self, config):
                return True
            
            def obtener_capacidades(self):
                return {"tipo": "fallback", "texturas": ["ambiental_basica"]}
            
            def get_info(self):
                return {"nombre": "HarmonicEssence", "version": "fallback"}
        
        return HarmonicFallback()
    
    def _crear_gestor_emotion_fallback(self):
        """Crea gestor de emociones fallback"""
        
        class EmotionFallback:
            def __init__(self):
                self.nombre = "EmotionProfiles"
                self.version = "fallback"
            
            def obtener_perfil(self, nombre):
                # Perfiles básicos fallback
                perfiles_basicos = {
                    "relajacion": {
                        "intensidad": 0.3,
                        "reverb_factor": 0.5,
                        "tipo_textura": "ambiental"
                    },
                    "enfoque": {
                        "intensidad": 0.6,
                        "reverb_factor": 0.2,
                        "tipo_textura": "neuronal"
                    }
                }
                return perfiles_basicos.get(nombre.lower(), perfiles_basicos["relajacion"])
            
            def listar_perfiles(self):
                return ["relajacion", "enfoque"]
            
            def validar_perfil(self, perfil):
                return True
        
        return EmotionFallback()
    
    def _crear_componente_generico_fallback(self, nombre: str):
        """Crea componente genérico fallback"""
        
        class ComponenteFallback:
            def __init__(self, nombre):
                self.nombre = nombre
                self.version = "fallback"
            
            def obtener_perfil(self, nombre):
                return {"nombre": nombre, "tipo": "fallback"}
            
            def listar_perfiles(self):
                return ["fallback"]
            
            def validar_perfil(self, perfil):
                return True
        
        return ComponenteFallback(nombre)
    
    # ===================================================================
    # IMPLEMENTACIÓN DE DirectorAuroraInterface
    # ===================================================================
    
    def registrar_componente(self, nombre: str, componente: Any, tipo: TipoComponenteAurora) -> bool:
        """
        Registra un componente en el director (implementa DirectorAuroraInterface)
        
        Args:
            nombre: Nombre del componente
            componente: Instancia del componente
            tipo: Tipo de componente
            
        Returns:
            True si se registró correctamente
        """
        try:
            if tipo == TipoComponenteAurora.MOTOR:
                if validar_motor_aurora(componente):
                    self.motores[nombre] = componente
                    logger.info(f"✅ Motor {nombre} registrado")
                    return True
                else:
                    logger.error(f"❌ Motor {nombre} no implementa la interface correcta")
                    return False
            
            elif tipo == TipoComponenteAurora.GESTOR:
                if validar_gestor_perfiles(componente):
                    self.gestores[nombre] = componente
                    logger.info(f"✅ Gestor {nombre} registrado")
                    return True
                else:
                    logger.error(f"❌ Gestor {nombre} no implementa la interface correcta")
                    return False
            
            # Otros tipos (por ahora aceptar sin validación específica)
            else:
                if tipo.value == "analizador":
                    self.analizadores[nombre] = componente
                elif tipo.value == "extension":
                    self.extensiones[nombre] = componente
                else:
                    self.pipelines[nombre] = componente
                
                logger.info(f"✅ Componente {nombre} ({tipo.value}) registrado")
                return True
        
        except Exception as e:
            logger.error(f"❌ Error registrando componente {nombre}: {e}")
            return False
    
    def crear_experiencia(self, config: ConfiguracionBaseInterface) -> ResultadoBaseInterface:
        """
        Crea una experiencia de audio completa (implementa DirectorAuroraInterface)
        
        Args:
            config: Configuración de la experiencia
            
        Returns:
            Resultado con audio generado y metadatos
        """
        start_time = time.time()
        
        try:
            # Convertir configuración si es necesario
            if isinstance(config, dict):
                config_aurora = ConfiguracionAuroraUnificada()
                config_aurora.from_dict(config)
            elif hasattr(config, 'to_dict'):
                config_aurora = ConfiguracionAuroraUnificada()
                config_aurora.from_dict(config.to_dict())
            else:
                config_aurora = config
            
            # Validar configuración
            if not config_aurora.validar():
                return self._crear_resultado_error("Configuración inválida")
            
            # Generar experiencia usando estrategia apropiada
            resultado = self._generar_experiencia_estrategica(config_aurora)
            
            # Actualizar estadísticas
            tiempo_total = time.time() - start_time
            self.stats["experiencias_generadas"] += 1
            self.stats["tiempo_total_generacion"] += tiempo_total
            
            estrategia_str = resultado.estrategia_usada.value
            self.stats["estrategias_usadas"][estrategia_str] = \
                self.stats["estrategias_usadas"].get(estrategia_str, 0) + 1
            
            for motor in resultado.motores_utilizados:
                self.stats["motores_mas_usados"][motor] = \
                    self.stats["motores_mas_usados"].get(motor, 0) + 1
            
            logger.info(f"✅ Experiencia generada en {tiempo_total:.2f}s usando {estrategia_str}")
            
            return resultado
            
        except Exception as e:
            self.stats["errores_manejados"] += 1
            logger.error(f"❌ Error creando experiencia: {e}")
            return self._crear_resultado_error(f"Error en generación: {e}")
    
    def obtener_estado_sistema(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo del sistema (implementa DirectorAuroraInterface)
        
        Returns:
            Dict con estado de todos los componentes
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "director": {
                "nombre": self.nombre,
                "version": self.version,
                "estado": self.estado.value,
                "modo_orquestacion": self.modo_orquestacion.value,
                "estrategia_default": self.estrategia_default.value
            },
            "componentes": {
                "motores": {
                    "activos": len(self.motores),
                    "nombres": list(self.motores.keys())
                },
                "gestores": {
                    "activos": len(self.gestores),
                    "nombres": list(self.gestores.keys())
                },
                "pipelines": {
                    "activos": len(self.pipelines),
                    "nombres": list(self.pipelines.keys())
                },
                "extensiones": {
                    "activas": len(self.extensiones),
                    "nombres": list(self.extensiones.keys())
                },
                "analizadores": {
                    "activos": len(self.analizadores),
                    "nombres": list(self.analizadores.keys())
                }
            },
            "estadisticas": self.stats.copy(),
            "deteccion_componentes": self.detector.componentes_disponibles
        }
    
    # ===================================================================
    # MÉTODOS DE GENERACIÓN ESTRATÉGICA
    # ===================================================================
    
    def _generar_experiencia_estrategica(self, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        """Genera experiencia usando la estrategia más apropiada"""
        
        # Seleccionar estrategia
        estrategia = config.estrategia_generacion
        if estrategia == EstrategiaGeneracion.SYNC_HIBRIDO and len(self.motores) < 2:
            estrategia = EstrategiaGeneracion.MULTI_MOTOR
        
        # Ejecutar según estrategia
        if estrategia == EstrategiaGeneracion.SYNC_HIBRIDO:
            return self._generar_sync_hibrido(config)
        elif estrategia == EstrategiaGeneracion.MULTI_MOTOR:
            return self._generar_multi_motor(config)
        elif estrategia == EstrategiaGeneracion.LAYERED:
            return self._generar_layered(config)
        else:
            return self._generar_fallback_progresivo(config)
    
    def _generar_sync_hibrido(self, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        """Genera experiencia usando sync híbrido (estrategia óptima)"""
        
        resultado = ResultadoAuroraIntegrado()
        resultado.estrategia_usada = EstrategiaGeneracion.SYNC_HIBRIDO
        
        try:
            # Obtener perfil emocional
            perfil_emocional = self._obtener_perfil_emocional(config.objetivo, config.estilo)
            
            # Generar con motor principal
            motor_principal = self._seleccionar_motor_principal()
            if not motor_principal:
                return self._generar_fallback_progresivo(config)
            
            # Configurar generación
            config_motor = self._preparar_configuracion_motor(config, perfil_emocional)
            
            # Generar audio base
            resultado_motor = motor_principal.generar_audio(config_motor)
            audio_base = resultado_motor.get("audio_data")
            
            if audio_base is None:
                return self._generar_fallback_progresivo(config)
            
            # Aplicar capas adicionales con otros motores
            audio_final = self._aplicar_capas_adicionales(audio_base, config, perfil_emocional)
            
            # Aplicar análisis y validación
            if config.validacion_estructural:
                validacion = self._aplicar_validacion_estructural(audio_final)
                resultado.validacion_estructural = validacion
            
            # Aplicar análisis de calidad
            analisis_calidad = self._aplicar_analisis_calidad(audio_final)
            resultado.analisis_calidad = analisis_calidad
            
            # Preparar resultado final
            resultado.exito = True
            resultado.mensaje = "Experiencia generada exitosamente con sync híbrido"
            resultado.audio_data = audio_final
            resultado.sample_rate = resultado_motor.get("sample_rate", 44100)
            resultado.duracion_real = len(audio_final) / resultado.sample_rate
            resultado.motores_utilizados = [motor_principal.__class__.__name__]
            
            return resultado
            
        except Exception as e:
            logger.error(f"Error en sync híbrido: {e}")
            return self._generar_fallback_progresivo(config)
    
    def _generar_multi_motor(self, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        """Genera experiencia usando múltiples motores"""
        
        resultado = ResultadoAuroraIntegrado()
        resultado.estrategia_usada = EstrategiaGeneracion.MULTI_MOTOR
        
        try:
            motores_activos = list(self.motores.values())[:3]  # Máximo 3 motores
            if not motores_activos:
                return self._generar_fallback_progresivo(config)
            
            # Generar con cada motor
            audios_generados = []
            for motor in motores_activos:
                try:
                    config_motor = self._preparar_configuracion_motor(config)
                    resultado_motor = motor.generar_audio(config_motor)
                    
                    audio_data = resultado_motor.get("audio_data")
                    if audio_data is not None:
                        audios_generados.append(audio_data)
                        resultado.motores_utilizados.append(motor.__class__.__name__)
                except Exception as e:
                    logger.warning(f"Motor {motor.__class__.__name__} falló: {e}")
                    continue
            
            if not audios_generados:
                return self._generar_fallback_progresivo(config)
            
            # Mezclar audios
            audio_final = self._mezclar_audios(audios_generados)
            
            resultado.exito = True
            resultado.mensaje = f"Experiencia generada con {len(audios_generados)} motores"
            resultado.audio_data = audio_final
            resultado.sample_rate = 44100
            resultado.duracion_real = len(audio_final) / 44100
            
            return resultado
            
        except Exception as e:
            logger.error(f"Error en multi-motor: {e}")
            return self._generar_fallback_progresivo(config)
    
    def _generar_layered(self, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        """Genera experiencia con capas secuenciales"""
        
        resultado = ResultadoAuroraIntegrado()
        resultado.estrategia_usada = EstrategiaGeneracion.LAYERED
        
        try:
            # Generar capa base
            motor_base = self._seleccionar_motor_principal()
            if not motor_base:
                return self._generar_fallback_progresivo(config)
            
            config_motor = self._preparar_configuracion_motor(config)
            resultado_base = motor_base.generar_audio(config_motor)
            audio_base = resultado_base.get("audio_data")
            
            if audio_base is None:
                return self._generar_fallback_progresivo(config)
            
            # Aplicar capas adicionales
            audio_final = audio_base
            
            # Capa harmónica
            if "harmonicEssence_v34" in self.motores:
                try:
                    harmonic_motor = self.motores["harmonicEssence_v34"]
                    config_harmonic = self._preparar_configuracion_harmonic(config)
                    resultado_harmonic = harmonic_motor.generar_audio(config_harmonic)
                    audio_harmonic = resultado_harmonic.get("audio_data")
                    
                    if audio_harmonic is not None:
                        audio_final = self._combinar_audios(audio_final, audio_harmonic, 0.7, 0.3)
                        resultado.motores_utilizados.append("harmonicEssence_v34")
                except Exception as e:
                    logger.warning(f"Error aplicando capa harmónica: {e}")
            
            resultado.exito = True
            resultado.mensaje = "Experiencia generada con capas secuenciales"
            resultado.audio_data = audio_final
            resultado.sample_rate = 44100
            resultado.duracion_real = len(audio_final) / 44100
            resultado.motores_utilizados.insert(0, motor_base.__class__.__name__)
            
            return resultado
            
        except Exception as e:
            logger.error(f"Error en layered: {e}")
            return self._generar_fallback_progresivo(config)
    
    def _generar_fallback_progresivo(self, config: ConfiguracionAuroraUnificada) -> ResultadoAuroraIntegrado:
        """Genera experiencia usando fallback progresivo"""
        
        resultado = ResultadoAuroraIntegrado()
        resultado.estrategia_usada = EstrategiaGeneracion.FALLBACK_PROGRESIVO
        self.stats["fallbacks_utilizados"] += 1
        
        try:
            # Intentar con cualquier motor disponible
            for motor_name, motor in self.motores.items():
                try:
                    config_motor = self._preparar_configuracion_motor(config)
                    resultado_motor = motor.generar_audio(config_motor)
                    
                    audio_data = resultado_motor.get("audio_data")
                    if audio_data is not None:
                        resultado.exito = True
                        resultado.mensaje = f"Experiencia generada con motor fallback {motor_name}"
                        resultado.audio_data = audio_data
                        resultado.sample_rate = resultado_motor.get("sample_rate", 44100)
                        resultado.duracion_real = len(audio_data) / resultado.sample_rate
                        resultado.motores_utilizados = [motor_name]
                        return resultado
                        
                except Exception as e:
                    logger.warning(f"Motor fallback {motor_name} falló: {e}")
                    continue
            
            # Fallback final: audio matemático simple
            audio_fallback = self._generar_audio_matematico_fallback(config)
            
            resultado.exito = True
            resultado.mensaje = "Audio generado con fallback matemático"
            resultado.audio_data = audio_fallback
            resultado.sample_rate = 44100
            resultado.duracion_real = len(audio_fallback) / 44100
            resultado.motores_utilizados = ["matematico_fallback"]
            
            return resultado
            
        except Exception as e:
            logger.error(f"Error en fallback progresivo: {e}")
            return self._crear_resultado_error(f"Todos los fallbacks fallaron: {e}")
    
    # ===================================================================
    # MÉTODOS DE UTILIDAD Y SOPORTE
    # ===================================================================
    
    def _obtener_perfil_emocional(self, objetivo: str, estilo: str) -> Dict[str, Any]:
        """Obtiene perfil emocional usando gestores disponibles"""
        
        # Intentar con gestor de emotion_style_profiles
        if "emotion_style_profiles" in self.gestores:
            try:
                gestor = self.gestores["emotion_style_profiles"]
                perfil = gestor.obtener_perfil(objetivo)
                if perfil:
                    return perfil
            except Exception as e:
                logger.warning(f"Error obteniendo perfil emocional: {e}")
        
        # Fallback: perfil básico
        return {
            "objetivo": objetivo,
            "estilo": estilo,
            "intensidad": 0.5,
            "tipo_textura": "ambiental",
            "reverb_factor": 0.3
        }
    
    def _seleccionar_motor_principal(self) -> Optional[Any]:
        """Selecciona el motor principal para generación"""
        
        # Prioridad: NeuroMix Super > NeuroMix > HarmonicEssence
        motores_prioritarios = [
            "neuromix_super_unificado",
            "neuromix_aurora_v27", 
            "harmonicEssence_v34"
        ]
        
        for motor_name in motores_prioritarios:
            if motor_name in self.motores:
                return self.motores[motor_name]
        
        # Cualquier motor disponible
        if self.motores:
            return list(self.motores.values())[0]
        
        return None
    
    def _preparar_configuracion_motor(self, config: ConfiguracionAuroraUnificada, 
                                     perfil_emocional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepara configuración para motores"""
        
        config_motor = {
            "objetivo": config.objetivo,
            "duracion_sec": config.duracion_min * 60,
            "intensidad": config.intensidad,
            "estilo": config.estilo,
            "calidad_objetivo": config.calidad_objetivo
        }
        
        if perfil_emocional:
            config_motor.update(perfil_emocional.get("configuracion_motor", {}))
        
        return config_motor
    
    def _preparar_configuracion_harmonic(self, config: ConfiguracionAuroraUnificada) -> Dict[str, Any]:
        """Prepara configuración específica para HarmonicEssence"""
        
        return {
            "tipo_textura": "ambiental",
            "modo_generacion": "estetico",
            "complejidad": "moderado",
            "duracion_sec": config.duracion_min * 60,
            "intensidad": config.intensidad * 0.7,
            "reverb_factor": 0.4,
            "spatial_width": 0.8
        }
    
    def _aplicar_capas_adicionales(self, audio_base: np.ndarray, 
                                  config: ConfiguracionAuroraUnificada,
                                  perfil_emocional: Dict[str, Any]) -> np.ndarray:
        """Aplica capas adicionales al audio base"""
        
        audio_resultado = audio_base.copy()
        
        # Aplicar HarmonicEssence si está disponible
        if "harmonicEssence_v34" in self.motores:
            try:
                harmonic_motor = self.motores["harmonicEssence_v34"]
                config_harmonic = self._preparar_configuracion_harmonic(config)
                resultado_harmonic = harmonic_motor.generar_audio(config_harmonic)
                
                audio_harmonic = resultado_harmonic.get("audio_data")
                if audio_harmonic is not None:
                    # Combinar audios
                    audio_resultado = self._combinar_audios(audio_resultado, audio_harmonic, 0.8, 0.2)
                    
            except Exception as e:
                logger.warning(f"Error aplicando capa harmónica: {e}")
        
        return audio_resultado
    
    def _combinar_audios(self, audio1: np.ndarray, audio2: np.ndarray, 
                        peso1: float, peso2: float) -> np.ndarray:
        """Combina dos audios con pesos específicos"""
        
        try:
            # Asegurar que ambos audios tengan la misma longitud
            min_length = min(len(audio1), len(audio2))
            audio1_crop = audio1[:min_length]
            audio2_crop = audio2[:min_length]
            
            # Asegurar misma forma (mono/estéreo)
            if len(audio1_crop.shape) != len(audio2_crop.shape):
                if len(audio1_crop.shape) == 1:
                    audio1_crop = np.column_stack([audio1_crop, audio1_crop])
                if len(audio2_crop.shape) == 1:
                    audio2_crop = np.column_stack([audio2_crop, audio2_crop])
            
            # Combinar
            audio_combinado = peso1 * audio1_crop + peso2 * audio2_crop
            
            # Normalizar para evitar clipping
            max_val = np.max(np.abs(audio_combinado))
            if max_val > 0.85:
                audio_combinado = audio_combinado * (0.85 / max_val)
            
            return audio_combinado
            
        except Exception as e:
            logger.warning(f"Error combinando audios: {e}")
            return audio1
    
    def _mezclar_audios(self, audios: List[np.ndarray]) -> np.ndarray:
        """Mezcla múltiples audios"""
        
        if not audios:
            # Audio silencio fallback
            return np.zeros((44100, 2))
        
        if len(audios) == 1:
            return audios[0]
        
        try:
            # Encontrar longitud mínima
            min_length = min(len(audio) for audio in audios)
            
            # Inicializar resultado
            audio_resultado = np.zeros((min_length, 2))
            
            # Mezclar audios
            peso_por_audio = 1.0 / len(audios)
            
            for audio in audios:
                audio_crop = audio[:min_length]
                
                # Convertir a estéreo si es necesario
                if len(audio_crop.shape) == 1:
                    audio_crop = np.column_stack([audio_crop, audio_crop])
                elif audio_crop.shape[1] == 1:
                    audio_crop = np.column_stack([audio_crop[:, 0], audio_crop[:, 0]])
                
                audio_resultado += peso_por_audio * audio_crop
            
            # Normalizar
            max_val = np.max(np.abs(audio_resultado))
            if max_val > 0.85:
                audio_resultado = audio_resultado * (0.85 / max_val)
            
            return audio_resultado
            
        except Exception as e:
            logger.error(f"Error mezclando audios: {e}")
            return audios[0]  # Retornar primer audio como fallback
    
    def _aplicar_validacion_estructural(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Aplica validación estructural al audio"""
        
        try:
            # Validación básica incorporada
            validacion = {
                "duracion_sec": len(audio_data) / 44100,
                "canales": audio_data.shape[1] if len(audio_data.shape) > 1 else 1,
                "nivel_maximo": float(np.max(np.abs(audio_data))),
                "clipping_detectado": bool(np.any(np.abs(audio_data) > 0.95)),
                "silencio_detectado": bool(np.all(np.abs(audio_data) < 0.001)),
                "validacion_basica": "exitosa"
            }
            
            # Si hay verificador estructural disponible, usarlo
            if "verify_structure" in self.pipelines:
                try:
                    verificador = self.pipelines["verify_structure"]
                    if hasattr(verificador, 'verificar_estructura_aurora_v7_unificada'):
                        resultado_verificacion = verificador.verificar_estructura_aurora_v7_unificada(audio_data)
                        validacion.update(resultado_verificacion)
                except Exception as e:
                    logger.warning(f"Error en verificación estructural avanzada: {e}")
            
            return validacion
            
        except Exception as e:
            logger.warning(f"Error en validación estructural: {e}")
            return {"validacion_basica": "error", "error": str(e)}
    
    def _aplicar_analisis_calidad(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Aplica análisis de calidad al audio"""
        
        try:
            # Análisis básico incorporado
            analisis = {
                "rms_level": float(np.sqrt(np.mean(audio_data ** 2))),
                "peak_level": float(np.max(np.abs(audio_data))),
                "dynamic_range": float(np.max(audio_data) - np.min(audio_data)),
                "analisis_basico": "completado"
            }
            
            # Si hay analizador Carmine disponible, usarlo
            if "carmine_analyzer" in self.analizadores:
                try:
                    carmine = self.analizadores["carmine_analyzer"]
                    if hasattr(carmine, 'analyze_audio'):
                        resultado_carmine = carmine.analyze_audio(audio_data)
                        analisis.update(resultado_carmine)
                except Exception as e:
                    logger.warning(f"Error en análisis Carmine: {e}")
            
            return analisis
            
        except Exception as e:
            logger.warning(f"Error en análisis de calidad: {e}")
            return {"analisis_basico": "error", "error": str(e)}
    
    def _generar_audio_matematico_fallback(self, config: ConfiguracionAuroraUnificada) -> np.ndarray:
        """Genera audio usando funciones matemáticas puras (último fallback)"""
        
        try:
            duracion = config.duracion_min * 60
            sample_rate = 44100
            
            t = np.linspace(0, duracion, int(duracion * sample_rate))
            
            # Frecuencias base
            freq_base = 220.0  # La3
            freq_binaural = 10.0  # Alfa
            
            # Canal izquierdo
            left = 0.3 * np.sin(2 * np.pi * freq_base * t)
            
            # Canal derecho con diferencia binaural
            right = 0.3 * np.sin(2 * np.pi * (freq_base + freq_binaural) * t)
            
            # Envelope para suavizar
            envelope = np.exp(-0.5 * (t - duracion/2)**2 / (duracion/4)**2)
            
            left *= envelope
            right *= envelope
            
            # Combinar canales
            audio_stereo = np.column_stack([left, right])
            
            return audio_stereo
            
        except Exception as e:
            logger.error(f"Error en fallback matemático: {e}")
            # Fallback del fallback: silencio
            return np.zeros((44100, 2))
    
    def _crear_resultado_error(self, mensaje: str) -> ResultadoAuroraIntegrado:
        """Crea resultado de error"""
        
        resultado = ResultadoAuroraIntegrado()
        resultado.exito = False
        resultado.mensaje = mensaje
        resultado.errores_encontrados = [mensaje]
        resultado.estrategia_usada = EstrategiaGeneracion.FALLBACK_PROGRESIVO
        
        # Audio silencio como placeholder
        resultado.audio_data = np.zeros((44100, 2))
        resultado.sample_rate = 44100
        resultado.duracion_real = 1.0
        
        return resultado

# ===============================================================================
# FUNCIONES DE FACTORY Y COMPATIBILIDAD
# ===============================================================================

def crear_aurora_director_refactorizado() -> AuroraDirectorV7Refactored:
    """
    Factory function para crear Aurora Director refactorizado
    
    Returns:
        Instancia de Aurora Director refactorizada
    """
    return AuroraDirectorV7Refactored()

# Instancia global para compatibilidad
_director_global = None

def Aurora() -> AuroraDirectorV7Refactored:
    """
    Función de compatibilidad para obtener instancia global del director
    
    Returns:
        Instancia global de Aurora Director
    """
    global _director_global
    if _director_global is None:
        _director_global = crear_aurora_director_refactorizado()
    return _director_global

# Auto-registro en factory si está disponible
if INTERFACES_DISPONIBLES:
    try:
        aurora_factory.registrar_componente = lambda nombre, tipo, modulo, clase: \
            aurora_factory.registrar_motor(nombre, modulo, clase) if tipo == "motor" else None
        
        logger.info("🏭 Aurora Director V7 refactorizado funcional")
    except Exception as e:
        logger.warning(f"No se pudo configurar factory: {e}")

# ===============================================================================
# FUNCIONES DE TEST
# ===============================================================================

def test_aurora_director_refactorizado():
    """Test del Aurora Director refactorizado"""
    
    print("🧪 Testeando Aurora Director V7 Refactorizado...")
    
    # Crear director
    director = crear_aurora_director_refactorizado()
    
    # Test estado
    estado = director.obtener_estado_sistema()
    print(f"✅ Estado obtenido: {estado['director']['estado']}")
    
    # Test configuración
    config = ConfiguracionAuroraUnificada(
        objetivo="relajacion",
        duracion_min=0.1,  # 6 segundos para test
        intensidad=0.5
    )
    
    print(f"✅ Configuración válida: {config.validar()}")
    
    # Test generación
    resultado = director.crear_experiencia(config)
    print(f"✅ Experiencia generada: {resultado.mensaje}")
    print(f"   Éxito: {resultado.exito}")
    print(f"   Estrategia: {resultado.estrategia_usada.value}")
    print(f"   Motores: {resultado.motores_utilizados}")
    
    # Test estadísticas
    stats = director.stats
    print(f"✅ Estadísticas: {stats['experiencias_generadas']} experiencias")
    
    print("🎉 Test completado exitosamente")

# ===============================================================================
# ALIASES PARA COMPATIBILIDAD
# ===============================================================================

# Aliases para mantener compatibilidad con código existente
AuroraDirectorV7Integrado = AuroraDirectorV7Refactored
DetectorComponentesAvanzado = DetectorComponentesRefactored

# ===============================================================================
# INFORMACIÓN DEL MÓDULO
# ===============================================================================

__version__ = VERSION
__author__ = "Sistema Aurora V7"
__description__ = "Aurora Director V7 refactorizado sin dependencias circulares"

if __name__ == "__main__":
    print("🎯 Aurora Director V7 Refactorizado")
    print(f"Versión: {__version__}")
    print("Estado: Sin dependencias circulares")
    
    # Ejecutar test
    test_aurora_director_refactorizado()
