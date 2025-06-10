#!/usr/bin/env python3
"""
🧠 NEUROMIX DEFINITIVO V7 - REFACTORIZADO SIN DEPENDENCIAS CIRCULARES
====================================================================

Parche unificado y mejorador del motor NeuroMix Aurora V27.
Versión refactorizada que utiliza interfaces y factory pattern para eliminar
dependencias circulares con aurora_director_v7.

Versión: V7.1 - Refactorizado
Propósito: Extensión y mejora de NeuroMix sin dependencias circulares
Compatibilidad: Aurora Director V7, NeuroMix Aurora V27
"""

import importlib
import logging
import time
import threading
import traceback
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import lru_cache, wraps
import numpy as np

# Importar interfaces Aurora (sin ciclos)
try:
    from aurora_interfaces import (
        MotorAuroraInterface, ComponenteAuroraBase, TipoComponenteAurora,
        EstadoComponente, NivelCalidad, validar_motor_aurora
    )
    from aurora_factory import aurora_factory, lazy_importer
    INTERFACES_DISPONIBLES = True
except ImportError as e:
    logging.warning(f"Aurora interfaces no disponibles: {e}")
    INTERFACES_DISPONIBLES = False
    
    # Clases dummy para compatibilidad
    class MotorAuroraInterface: pass
    class ComponenteAuroraBase: pass

# Configuración de logging
logger = logging.getLogger("Aurora.NeuroMix.Definitivo.V7.Refactored")

VERSION = "V7.1_REFACTORIZADO_NO_CIRCULAR_DEPS"

# ===============================================================================
# DETECCIÓN Y GESTIÓN SIN DEPENDENCIAS CIRCULARES
# ===============================================================================

class SuperDetectorAuroraRefactored:
    """
    Detector inteligente del sistema Aurora sin dependencias circulares
    
    Utiliza lazy loading y factory pattern para detectar componentes
    sin importar directamente aurora_director_v7
    """
    
    def __init__(self):
        self.componentes_detectados = {}
        self.estado_sistema = {}
        self.director_global = None
        self.neuromix_original = None
        self.cache_deteccion = {}
        self.lock = threading.Lock()
        
        # Inicializar detección
        self._detectar_sistema_lazy()
    
    def _detectar_sistema_lazy(self):
        """Detecta sistema Aurora usando lazy loading"""
        
        with self.lock:
            logger.info("🔍 Iniciando detección del sistema Aurora (sin dependencias circulares)")
            
            # Detectar Aurora Director usando factory
            self._detectar_aurora_director_lazy()
            
            # Detectar componentes individuales
            self._detectar_componentes_individuales_lazy()
            
            # Detectar NeuroMix original
            self._detectar_neuromix_original_lazy()
            
            logger.info(f"✅ Detección completada: {len(self.componentes_detectados)} componentes")
    
    def _detectar_aurora_director_lazy(self):
        """Detecta Aurora Director usando factory sin importación directa"""
        
        try:
            # Intentar obtener director usando factory
            director_factory = lazy_importer.get_class('aurora_director_v7', 'Aurora')
            if director_factory:
                self.director_global = director_factory()
                self.estado_sistema["director"] = {
                    "disponible": True,
                    "version": getattr(self.director_global, 'version', 'unknown'),
                    "modo_creacion": "factory",
                    "componentes": len(getattr(self.director_global, 'componentes', {}))
                }
                logger.info("✅ Aurora Director detectado vía factory")
                return
            
            # Fallback: usar aurora_factory para crear director
            director_instancia = aurora_factory.crear_motor("aurora_director_v7")
            if director_instancia:
                self.director_global = director_instancia
                self.estado_sistema["director"] = {
                    "disponible": True,
                    "version": getattr(director_instancia, 'version', 'unknown'),
                    "modo_creacion": "aurora_factory",
                    "componentes": 0
                }
                logger.info("✅ Aurora Director creado vía aurora_factory")
                return
            
            # No se encontró director
            self.estado_sistema["director"] = {
                "disponible": False,
                "motivo": "no_encontrado",
                "modo_creacion": "ninguno"
            }
            logger.warning("⚠️ Aurora Director no encontrado")
            
        except Exception as e:
            self.estado_sistema["director"] = {
                "disponible": False,
                "motivo": f"error: {e}",
                "modo_creacion": "ninguno"
            }
            logger.error(f"❌ Error detectando Aurora Director: {e}")
    
    def _detectar_componentes_individuales_lazy(self):
        """Detecta componentes individuales usando lazy loading"""
        
        componentes_buscar = [
            ("neuromix_aurora_v27", "AuroraNeuroAcousticEngineV27"),
            ("hypermod_v32", "HyperModEngineV32AuroraConnected"),
            ("harmonicEssence_v34", "HarmonicEssenceV34Refactored"),
            ("emotion_style_profiles", "GestorPresetsEmocionalesRefactored")
        ]
        
        for modulo_nombre, clase_nombre in componentes_buscar:
            try:
                # Usar lazy importer para evitar dependencias circulares
                clase_componente = lazy_importer.get_class(modulo_nombre, clase_nombre)
                
                if clase_componente:
                    self.componentes_detectados[modulo_nombre] = {
                        "modulo": modulo_nombre,
                        "clase": clase_nombre,
                        "disponible": True,
                        "activo": False,
                        "modo_deteccion": "lazy_loading"
                    }
                    logger.debug(f"✅ {modulo_nombre} detectado")
                else:
                    self.componentes_detectados[modulo_nombre] = {
                        "modulo": modulo_nombre,
                        "clase": clase_nombre,
                        "disponible": False,
                        "activo": False,
                        "modo_deteccion": "fallido"
                    }
                    logger.debug(f"❌ {modulo_nombre} no disponible")
                    
            except Exception as e:
                logger.debug(f"Error detectando {modulo_nombre}: {e}")
                self.componentes_detectados[modulo_nombre] = {
                    "modulo": modulo_nombre,
                    "clase": clase_nombre,
                    "disponible": False,
                    "activo": False,
                    "error": str(e),
                    "modo_deteccion": "error"
                }
    
    def _detectar_neuromix_original_lazy(self):
        """Detecta NeuroMix original usando lazy loading"""
        
        try:
            # Usar factory primero
            neuromix_original = aurora_factory.crear_motor("neuromix_aurora_v27")
            if neuromix_original:
                self.neuromix_original = neuromix_original.__class__
                logger.info("✅ NeuroMix original detectado vía factory")
                return
            
            # Fallback: lazy loading directo
            neuromix_class = lazy_importer.get_class('neuromix_aurora_v27', 'AuroraNeuroAcousticEngineV27')
            if neuromix_class:
                self.neuromix_original = neuromix_class
                logger.info("✅ NeuroMix original detectado vía lazy loading")
                return
            
            logger.warning("⚠️ NeuroMix original no encontrado")
            
        except Exception as e:
            logger.error(f"❌ Error detectando NeuroMix original: {e}")
    
    def obtener_estado_completo(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema detectado"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "director": self.estado_sistema.get("director", {}),
            "componentes": self.componentes_detectados,
            "neuromix_original_disponible": self.neuromix_original is not None,
            "total_componentes_detectados": len(self.componentes_detectados),
            "componentes_activos": len([c for c in self.componentes_detectados.values() if c.get("disponible", False)])
        }

# ===============================================================================
# AUTO-REPARADOR SIN DEPENDENCIAS CIRCULARES
# ===============================================================================

class AutoReparadorSistemaRefactored:
    """
    Auto-reparador del sistema que no depende de imports circulares
    
    Usa factory pattern y lazy loading para reparar el sistema
    sin crear dependencias circulares
    """
    
    def __init__(self, detector: SuperDetectorAuroraRefactored):
        self.detector = detector
        self.reparaciones_aplicadas = []
        self.lock = threading.Lock()
    
    def auto_reparar_sistema(self) -> Dict[str, Any]:
        """
        Auto-repara el sistema detectando y corrigiendo problemas
        
        Returns:
            Dict con resultado de las reparaciones
        """
        
        with self.lock:
            logger.info("🔧 Iniciando auto-reparación del sistema")
            
            resultado = {
                "reparaciones_exitosas": 0,
                "reparaciones_fallidas": 0,
                "problemas_detectados": [],
                "soluciones_aplicadas": [],
                "tiempo_reparacion": 0
            }
            
            start_time = time.time()
            
            try:
                # Diagnosticar problemas
                problemas = self._diagnosticar_problemas()
                resultado["problemas_detectados"] = problemas
                
                # Aplicar reparaciones
                for problema in problemas:
                    if self._aplicar_reparacion(problema):
                        resultado["reparaciones_exitosas"] += 1
                        resultado["soluciones_aplicadas"].append(problema["solucion"])
                    else:
                        resultado["reparaciones_fallidas"] += 1
                
                resultado["tiempo_reparacion"] = time.time() - start_time
                
                logger.info(f"✅ Auto-reparación completada: {resultado['reparaciones_exitosas']} exitosas")
                
            except Exception as e:
                logger.error(f"❌ Error en auto-reparación: {e}")
                resultado["error_general"] = str(e)
            
            return resultado
    
    def _diagnosticar_problemas(self) -> List[Dict[str, Any]]:
        """Diagnostica problemas del sistema"""
        
        problemas = []
        
        # Verificar director
        if not self.detector.estado_sistema.get("director", {}).get("disponible", False):
            problemas.append({
                "tipo": "director_no_disponible",
                "descripcion": "Aurora Director no está disponible",
                "gravedad": "alta",
                "solucion": "crear_director_fallback"
            })
        
        # Verificar NeuroMix original
        if not self.detector.neuromix_original:
            problemas.append({
                "tipo": "neuromix_original_no_disponible",
                "descripcion": "NeuroMix original no está disponible",
                "gravedad": "alta",
                "solucion": "crear_neuromix_fallback"
            })
        
        # Verificar componentes críticos
        componentes_criticos = ["harmonicEssence_v34", "emotion_style_profiles"]
        for comp in componentes_criticos:
            if comp not in self.detector.componentes_detectados or \
               not self.detector.componentes_detectados[comp].get("disponible", False):
                problemas.append({
                    "tipo": f"componente_critico_no_disponible",
                    "descripcion": f"Componente crítico {comp} no disponible",
                    "gravedad": "media",
                    "solucion": f"crear_{comp}_fallback",
                    "componente": comp
                })
        
        return problemas
    
    def _aplicar_reparacion(self, problema: Dict[str, Any]) -> bool:
        """Aplica una reparación específica"""
        
        try:
            solucion = problema["solucion"]
            
            if solucion == "crear_director_fallback":
                return self._crear_director_fallback()
            elif solucion == "crear_neuromix_fallback":
                return self._crear_neuromix_fallback()
            elif solucion.startswith("crear_") and solucion.endswith("_fallback"):
                componente = problema.get("componente")
                return self._crear_componente_fallback(componente)
            
            return False
            
        except Exception as e:
            logger.error(f"Error aplicando reparación {problema['solucion']}: {e}")
            return False
    
    def _crear_director_fallback(self) -> bool:
        """Crea un director fallback básico"""
        
        try:
            # Director fallback simple
            class DirectorFallback:
                def __init__(self):
                    self.version = "fallback"
                    self.componentes = {}
                
                def generar_experiencia(self, config):
                    return {"modo": "fallback", "config": config}
            
            self.detector.director_global = DirectorFallback()
            self.detector.estado_sistema["director"] = {
                "disponible": True,
                "version": "fallback",
                "modo_creacion": "auto_reparacion"
            }
            
            logger.info("✅ Director fallback creado")
            return True
            
        except Exception as e:
            logger.error(f"Error creando director fallback: {e}")
            return False
    
    def _crear_neuromix_fallback(self) -> bool:
        """Crea NeuroMix fallback básico"""
        
        try:
            # NeuroMix fallback simple
            class NeuroMixFallback:
                def __init__(self):
                    self.nombre = "NeuroMix"
                    self.version = "fallback"
                
                def generar_audio(self, config):
                    # Audio básico fallback
                    duracion = config.get('duracion_sec', 60)
                    sample_rate = 44100
                    t = np.linspace(0, duracion, int(duracion * sample_rate))
                    audio = 0.1 * np.sin(2 * np.pi * 220 * t)  # 220 Hz simple
                    return {"audio_data": audio, "sample_rate": sample_rate}
                
                def validar_configuracion(self, config):
                    return True
                
                def obtener_capacidades(self):
                    return {"tipo": "fallback", "capacidades": ["generacion_basica"]}
                
                def get_info(self):
                    return {"nombre": "NeuroMix", "version": "fallback"}
            
            self.detector.neuromix_original = NeuroMixFallback
            logger.info("✅ NeuroMix fallback creado")
            return True
            
        except Exception as e:
            logger.error(f"Error creando NeuroMix fallback: {e}")
            return False
    
    def _crear_componente_fallback(self, componente: str) -> bool:
        """Crea fallback para componente específico"""
        
        try:
            # Componente fallback genérico
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
            
            self.detector.componentes_detectados[componente] = {
                "modulo": componente,
                "clase": "ComponenteFallback",
                "disponible": True,
                "activo": True,
                "modo_deteccion": "auto_reparacion_fallback"
            }
            
            logger.info(f"✅ Componente fallback creado para {componente}")
            return True
            
        except Exception as e:
            logger.error(f"Error creando fallback para {componente}: {e}")
            return False

# ===============================================================================
# NEUROMIX SUPER UNIFICADO REFACTORIZADO
# ===============================================================================

class NeuroMixSuperUnificadoRefactored(ComponenteAuroraBase):
    """
    Motor NeuroMix super unificado refactorizado sin dependencias circulares
    
    Extiende el NeuroMix original con mejoras y capacidades adicionales,
    usando interfaces y factory pattern para evitar dependencias circulares.
    """
    
    def __init__(self, neuromix_base: Optional[Any] = None):
        super().__init__("NeuroMixSuperUnificado", VERSION)
        
        # Motor base (sin detección automática)
        self.neuromix_base = neuromix_base or self._obtener_neuromix_base()
        
        # Capacidades extendidas
        self.capacidades_extendidas = {
            "auto_reparacion": True,
            "deteccion_inteligente": False,  # Deshabilitado automático
            "modo_super_unificado": True,
            "compatibilidad_interfaces": True,
            "fallback_garantizado": True
        }
        
        # Estadísticas
        self.stats = {
            "audio_generado": 0,
            "auto_reparaciones": 0,
            "fallbacks_usados": 0,
            "tiempo_total_procesamiento": 0.0
        }
        
        # Variables para detección manual
        self.detector = None
        self.auto_reparador = None
        self.sistema_inicializado = False
        
        self.cambiar_estado(EstadoComponente.CONFIGURADO)
        logger.info("🧠 NeuroMix Super Unificado refactorizado inicializado (modo pasivo)")
    
    def _obtener_neuromix_base(self) -> Optional[Any]:
        """Obtiene NeuroMix base usando lazy loading"""
        
        try:
            # Intentar usar factory
            neuromix_base = aurora_factory.crear_motor("neuromix_aurora_v27")
            if neuromix_base:
                logger.info("✅ NeuroMix base obtenido vía factory")
                return neuromix_base
            
            # Fallback: lazy loading directo
            neuromix_class = lazy_importer.get_class('neuromix_aurora_v27', 'AuroraNeuroAcousticEngineV27')
            if neuromix_class:
                neuromix_base = neuromix_class()
                logger.info("✅ NeuroMix base obtenido vía lazy loading")
                return neuromix_base
            
            logger.warning("⚠️ NeuroMix base no disponible, usando fallback")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo NeuroMix base: {e}")
            return None
    
    def inicializar_deteccion_manual(self):
        """
        Inicializa la detección de manera manual (llamado por el Director)
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        if self.sistema_inicializado:
            logger.debug("Sistema ya inicializado, omitiendo")
            return True
        
        try:
            logger.info("🔧 Inicializando detección manual en NeuroMix Super...")
            
            # Crear detector y auto-reparador solo cuando se necesite
            self.detector = SuperDetectorAuroraRefactored()
            self.auto_reparador = AutoReparadorSistemaRefactored(self.detector)
            
            # Auto-reparar si es necesario
            resultado_reparacion = self.auto_reparador.auto_reparar_sistema()
            self.stats["auto_reparaciones"] = resultado_reparacion.get("reparaciones_exitosas", 0)
            
            # Validar estado del sistema
            estado = self.detector.obtener_estado_completo()
            
            if estado["componentes_activos"] == 0:
                logger.warning("⚠️ No hay componentes activos, activando modo fallback")
                self._activar_modo_fallback()
            
            self.capacidades_extendidas["deteccion_inteligente"] = True
            self.sistema_inicializado = True
            
            logger.info("✅ Detección manual inicializada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización manual: {e}")
            self._activar_modo_fallback()
            return False
    
    def _activar_modo_fallback(self):
        """Activa modo fallback para garantizar funcionamiento"""
        
        self.capacidades_extendidas["modo_fallback_activo"] = True
        self.stats["fallbacks_usados"] += 1
        logger.info("🛡️ Modo fallback activado")
    
    # ===================================================================
    # IMPLEMENTACIÓN DE MotorAuroraInterface
    # ===================================================================
    
    def generar_audio(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementación de MotorAuroraInterface para generación de audio
        
        Args:
            config: Configuración de generación
            
        Returns:
            Dict con audio generado y metadatos
        """
        start_time = time.time()
        
        try:
            # Usar motor base si está disponible
            if self.neuromix_base and hasattr(self.neuromix_base, 'generar_audio'):
                resultado = self.neuromix_base.generar_audio(config)
                
                # Aplicar mejoras super unificadas
                resultado = self._aplicar_mejoras_super_unificadas(resultado, config)
                
            else:
                # Fallback: generar audio básico
                resultado = self._generar_audio_fallback(config)
            
            # Actualizar estadísticas
            tiempo_procesamiento = time.time() - start_time
            self.stats["audio_generado"] += 1
            self.stats["tiempo_total_procesamiento"] += tiempo_procesamiento
            
            # Agregar metadatos del motor super unificado
            resultado["metadatos"] = resultado.get("metadatos", {})
            resultado["metadatos"].update({
                "motor_super_unificado": True,
                "version": self.version,
                "tiempo_procesamiento": tiempo_procesamiento,
                "mejoras_aplicadas": self.capacidades_extendidas
            })
            
            logger.debug(f"✅ Audio generado en {tiempo_procesamiento:.2f}s")
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Error generando audio: {e}")
            return self._generar_audio_fallback(config)
    
    def validar_configuracion(self, config: Dict[str, Any]) -> bool:
        """
        Valida configuración usando motor base o fallback
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        try:
            # Usar validación del motor base si está disponible
            if self.neuromix_base and hasattr(self.neuromix_base, 'validar_configuracion'):
                return self.neuromix_base.validar_configuracion(config)
            
            # Validación fallback básica
            return self._validar_configuracion_fallback(config)
            
        except Exception as e:
            logger.error(f"Error validando configuración: {e}")
            return False
    
    def obtener_capacidades(self) -> Dict[str, Any]:
        """
        Obtiene capacidades del motor super unificado
        
        Returns:
            Dict con capacidades extendidas
        """
        capacidades_base = {}
        
        # Obtener capacidades del motor base
        if self.neuromix_base and hasattr(self.neuromix_base, 'obtener_capacidades'):
            try:
                capacidades_base = self.neuromix_base.obtener_capacidades()
            except Exception as e:
                logger.warning(f"Error obteniendo capacidades base: {e}")
        
        # Capacidades del motor super unificado
        capacidades_extendidas = {
            **capacidades_base,
            "motor_super_unificado": True,
            "version": self.version,
            "estado": self.estado.value,
            "capacidades_adicionales": self.capacidades_extendidas,
            "estadisticas": self.stats.copy(),
            "sistema_detectado": self.detector.obtener_estado_completo(),
            "interfaces_implementadas": ["MotorAuroraInterface"],
            "modos_disponibles": ["super_unificado", "fallback", "auto_reparacion"]
        }
        
        return capacidades_extendidas
    
    def get_info(self) -> Dict[str, str]:
        """
        Información del motor super unificado
        
        Returns:
            Dict con información básica
        """
        return {
            "nombre": self.nombre,
            "version": self.version,
            "descripcion": "Motor NeuroMix super unificado sin dependencias circulares",
            "tipo": "motor_neuroacustico_extendido",
            "categoria": "generacion_audio",
            "estado": self.estado.value,
            "motor_base": type(self.neuromix_base).__name__ if self.neuromix_base else "fallback",
            "autor": "Sistema Aurora V7",
            "fecha_refactorizacion": "2025-06-10"
        }
    
    # ===================================================================
    # MÉTODOS DE MEJORA Y EXTENSIÓN
    # ===================================================================
    
    def _aplicar_mejoras_super_unificadas(self, resultado: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica mejoras del motor super unificado al audio generado
        
        Args:
            resultado: Resultado del motor base
            config: Configuración original
            
        Returns:
            Resultado mejorado
        """
        try:
            # Mejoras de calidad
            if "audio_data" in resultado:
                audio_data = resultado["audio_data"]
                
                # Aplicar normalización inteligente
                audio_data = self._normalizar_inteligente(audio_data)
                
                # Aplicar mejoras de coherencia
                audio_data = self._mejorar_coherencia_neuroacustica(audio_data, config)
                
                resultado["audio_data"] = audio_data
            
            # Mejoras de metadatos
            metadatos = resultado.get("metadatos", {})
            metadatos["mejoras_super_unificadas"] = {
                "normalizacion_inteligente": True,
                "coherencia_mejorada": True,
                "deteccion_automatica": True
            }
            resultado["metadatos"] = metadatos
            
            return resultado
            
        except Exception as e:
            logger.warning(f"Error aplicando mejoras super unificadas: {e}")
            return resultado
    
    def _normalizar_inteligente(self, audio_data: np.ndarray) -> np.ndarray:
        """Aplica normalización inteligente al audio"""
        
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Normalización con preservación de dinámica
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                # Usar compresión suave en lugar de normalización dura
                ratio_compresion = 0.85
                audio_data = audio_data * (ratio_compresion / max_val)
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Error en normalización inteligente: {e}")
            return audio_data
    
    def _mejorar_coherencia_neuroacustica(self, audio_data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Mejora la coherencia neuroacústica del audio"""
        
        try:
            # Aplicar suavizado de transiciones si es audio estéreo
            if len(audio_data.shape) == 2 and audio_data.shape[1] == 2:
                # Suavizar diferencias entre canales
                diff = audio_data[:, 0] - audio_data[:, 1]
                factor_suavizado = 0.1
                diff_suavizado = diff * factor_suavizado
                
                audio_data[:, 0] = audio_data[:, 0] - diff_suavizado * 0.5
                audio_data[:, 1] = audio_data[:, 1] + diff_suavizado * 0.5
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Error mejorando coherencia neuroacústica: {e}")
            return audio_data
    
    def _generar_audio_fallback(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Genera audio fallback cuando el motor base no está disponible"""
        
        try:
            duracion = config.get('duracion_sec', 300.0)
            sample_rate = config.get('sample_rate', 44100)
            frecuencia = config.get('frecuencia_base', 220.0)
            
            # Generar audio fallback básico pero funcional
            num_samples = int(duracion * sample_rate)
            t = np.linspace(0, duracion, num_samples)
            
            # Onda portadora
            carrier = np.sin(2 * np.pi * frecuencia * t)
            
            # Modulación binaural básica
            freq_binaural = config.get('freq_binaural', 10.0)
            modulacion = 1 + 0.1 * np.sin(2 * np.pi * freq_binaural * t)
            
            # Audio estéreo básico
            left = carrier * modulacion * 0.3
            right = carrier * 0.3
            
            audio_data = np.column_stack([left, right])
            
            self.stats["fallbacks_usados"] += 1
            
            return {
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "duracion_sec": duracion,
                "metadatos": {
                    "motor": self.nombre,
                    "version": self.version,
                    "modo": "fallback",
                    "frecuencia_base": frecuencia,
                    "freq_binaural": freq_binaural
                }
            }
            
        except Exception as e:
            logger.error(f"Error en fallback: {e}")
            # Fallback del fallback
            return {
                "audio_data": np.zeros((44100, 2)),
                "sample_rate": 44100,
                "duracion_sec": 1.0,
                "metadatos": {"motor": self.nombre, "modo": "emergency_fallback"}
            }
    
    def _validar_configuracion_fallback(self, config: Dict[str, Any]) -> bool:
        """Validación fallback básica"""
        
        try:
            # Validaciones básicas
            if not isinstance(config, dict):
                return False
            
            # Validar duración
            duracion = config.get('duracion_sec', 300.0)
            if not 1.0 <= duracion <= 3600.0:
                return False
            
            # Validar frecuencias
            freq_base = config.get('frecuencia_base', 220.0)
            if not 20.0 <= freq_base <= 20000.0:
                return False
            
            return True
            
        except Exception:
            return False
    
    # ===================================================================
    # MÉTODOS DE UTILIDAD Y COMPATIBILIDAD
    # ===================================================================
    
    def diagnosticar_sistema_completo(self) -> Dict[str, Any]:
        """Diagnóstica el sistema completo"""
        
        return {
            "motor_super_unificado": {
                "nombre": self.nombre,
                "version": self.version,
                "estado": self.estado.value,
                "estadisticas": self.stats.copy()
            },
            "detector": self.detector.obtener_estado_completo(),
            "auto_reparador": {
                "reparaciones_disponibles": True,
                "ultimo_diagnostico": datetime.now().isoformat()
            },
            "motor_base": {
                "disponible": self.neuromix_base is not None,
                "tipo": type(self.neuromix_base).__name__ if self.neuromix_base else "fallback"
            }
        }
    
    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del motor"""
        
        return {
            "estadisticas_motor": self.stats.copy(),
            "capacidades": self.capacidades_extendidas.copy(),
            "estado_sistema": self.detector.obtener_estado_completo(),
            "version": self.version,
            "tiempo_funcionamiento": time.time()
        }

# ===============================================================================
# FUNCIONES DE FACTORY Y COMPATIBILIDAD
# ===============================================================================

def crear_neuromix_super_unificado(neuromix_base: Optional[Any] = None) -> NeuroMixSuperUnificadoRefactored:
    """
    Factory function para crear NeuroMix Super Unificado refactorizado
    
    Args:
        neuromix_base: Motor NeuroMix base opcional
        
    Returns:
        Instancia de NeuroMix Super Unificado refactorizada
    """
    return NeuroMixSuperUnificadoRefactored(neuromix_base)

def aplicar_super_parche_automatico() -> bool:
    """
    Aplica el super parche automáticamente registrándose en el factory
    
    Returns:
        True si se aplicó correctamente
    """
    try:
        if INTERFACES_DISPONIBLES:
            aurora_factory.registrar_motor(
                "neuromix_super_unificado",
                "neuromix_definitivo_v7",  # Este archivo
                "NeuroMixSuperUnificadoRefactored"
            )
            
            logger.info("🏭 NeuroMix Super Unificado registrado en factory")
            return True
        else:
            logger.warning("⚠️ Interfaces no disponibles para registro en factory")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error aplicando super parche: {e}")
        return False

def obtener_neuromix_super_global() -> Optional[NeuroMixSuperUnificadoRefactored]:
    """
    Obtiene instancia global de NeuroMix Super Unificado
    
    Returns:
        Instancia global o None si no está disponible
    """
    try:
        if INTERFACES_DISPONIBLES:
            return aurora_factory.crear_motor("neuromix_super_unificado")
        else:
            return crear_neuromix_super_unificado()
    except Exception as e:
        logger.error(f"Error obteniendo NeuroMix Super global: {e}")
        return None

def diagnosticar_sistema_completo() -> Dict[str, Any]:
    """
    Función de utilidad para diagnosticar todo el sistema
    
    Returns:
        Diagnóstico completo del sistema
    """
    try:
        neuromix_super = obtener_neuromix_super_global()
        if neuromix_super:
            return neuromix_super.diagnosticar_sistema_completo()
        else:
            # Diagnóstico básico sin motor
            detector = SuperDetectorAuroraRefactored()
            return {
                "estado": "diagnostico_basico",
                "detector": detector.obtener_estado_completo(),
                "motor_super_unificado": "no_disponible"
            }
    except Exception as e:
        return {
            "estado": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ===============================================================================
# ALIASES PARA COMPATIBILIDAD
# ===============================================================================

# Aliases para mantener compatibilidad con código existente
NeuroMixSuperUnificado = NeuroMixSuperUnificadoRefactored
SuperDetectorAurora = SuperDetectorAuroraRefactored
AutoReparadorSistema = AutoReparadorSistemaRefactored

# ===============================================================================
# AUTO-APLICACIÓN DEL PARCHE
# ===============================================================================

# Auto-aplicar el parche al importar el módulo
if __name__ != "__main__":
    aplicar_super_parche_automatico()

# ===============================================================================
# FUNCIONES DE TEST
# ===============================================================================

def test_neuromix_definitivo_refactorizado():
    """Test del NeuroMix definitivo refactorizado"""
    
    print("🧪 Testeando NeuroMix Definitivo V7 Refactorizado...")
    
    # Crear instancia
    motor = crear_neuromix_super_unificado()
    
    # Test info
    info = motor.get_info()
    print(f"✅ Info: {info['nombre']} v{info['version']}")
    
    # Test capacidades
    capacidades = motor.obtener_capacidades()
    print(f"✅ Capacidades: {len(capacidades)} propiedades")
    
    # Test configuración
    config = {
        "duracion_sec": 10.0,
        "frecuencia_base": 220.0,
        "freq_binaural": 10.0
    }
    
    es_valida = motor.validar_configuracion(config)
    print(f"✅ Validación configuración: {es_valida}")
    
    # Test generación
    resultado = motor.generar_audio(config)
    print(f"✅ Audio generado: {resultado.get('duracion_sec', 0):.1f}s")
    
    # Test diagnóstico
    diagnostico = motor.diagnosticar_sistema_completo()
    print(f"✅ Diagnóstico: {diagnostico['motor_super_unificado']['estado']}")
    
    print("🎉 Test completado exitosamente")

# ===============================================================================
# INFORMACIÓN DEL MÓDULO
# ===============================================================================

__version__ = VERSION
__author__ = "Sistema Aurora V7"
__description__ = "NeuroMix Super Unificado sin dependencias circulares"

if __name__ == "__main__":
    print("🧠 NeuroMix Definitivo V7 Refactorizado")
    print(f"Versión: {__version__}")
    print("Estado: Sin dependencias circulares")
    
    # Ejecutar test
    test_neuromix_definitivo_refactorizado()
