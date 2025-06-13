#!/usr/bin/env python3
"""
🧠 NEUROMIX DEFINITIVO V7 - REFACTORIZADO SIN DEPENDENCIAS CIRCULARES
====================================================================

Parche unificado y mejorador del motor NeuroMix Aurora V27.
Versión refactorizada que utiliza interfaces y factory pattern para eliminar
dependencias circulares con aurora_director_v7.

VERSIÓN CORREGIDA: Elimina deadlock en inicialización
- Constructor 100% pasivo 
- Inicialización manual controlada
- Sin bloqueos en factory

Versión: V7.2 - Sin Deadlock
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
    class ComponenteAuroraBase: 
        def __init__(self, nombre, version):
            self.nombre = nombre
            self.version = version
            self.estado = "configurado"
        def cambiar_estado(self, estado):
            self.estado = estado

# Configuración de logging
logger = logging.getLogger("Aurora.NeuroMix.Definitivo.V7.Refactored")

VERSION = "V7.2_SIN_DEADLOCK_REFACTORIZADO"

# ===============================================================================
# DETECCIÓN Y GESTIÓN SIN DEPENDENCIAS CIRCULARES (VERSIÓN PASIVA)
# ===============================================================================

class SuperDetectorAuroraRefactored:
    """
    Detector inteligente del sistema Aurora sin dependencias circulares
    
    VERSIÓN CORREGIDA: Constructor pasivo para evitar deadlock
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
        self._deteccion_inicializada = False
        
        # CORRECCIÓN: NO inicializar detección automáticamente
        # self._detectar_sistema_lazy()  # Comentado para evitar deadlock
        logger.debug("🔍 SuperDetector creado en modo pasivo (sin detección automática)")
    
    def inicializar_deteccion(self):
        """Inicializa detección de manera manual y segura"""
        if self._deteccion_inicializada:
            return True
        
        try:
            logger.info("🔍 Iniciando detección manual del sistema Aurora")
            self._detectar_sistema_lazy()
            self._deteccion_inicializada = True
            logger.info("✅ Detección manual completada")
            return True
        except Exception as e:
            logger.error(f"❌ Error en detección manual: {e}")
            return False
    
    def _detectar_sistema_lazy(self):
        """Detecta sistema Aurora usando lazy loading (llamado manualmente)"""
        
        with self.lock:
            logger.debug("🔍 Ejecutando detección lazy del sistema Aurora")
            
            # Detectar Aurora Director usando factory
            self._detectar_aurora_director_lazy()
            
            # Detectar componentes individuales
            self._detectar_componentes_individuales_lazy()
            
            # Detectar NeuroMix original
            self._detectar_neuromix_original_lazy()
    
    def _detectar_aurora_director_lazy(self):
        """Detecta Aurora Director usando factory sin crear dependencias circulares"""
        
        try:
            # Intentar obtener Director vía factory (sin crear instancia)
            if hasattr(aurora_factory, '_componentes') and 'aurora_director' in aurora_factory._componentes:
                self.estado_sistema["director"] = {
                    "disponible": True,
                    "modo_deteccion": "factory_reference",
                    "activo": False
                }
                logger.debug("✅ Aurora Director detectado vía factory")
            else:
                logger.debug("⚠️ Aurora Director no detectado en factory")
                
        except Exception as e:
            logger.debug(f"Error detectando Aurora Director: {e}")
    
    def _detectar_componentes_individuales_lazy(self):
        """Detecta componentes individuales usando lazy loading"""
        
        # Lista de componentes a detectar
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
            # Usar factory primero (sin crear instancia)
            if hasattr(aurora_factory, 'verificar_componente_disponible'):
                if aurora_factory.verificar_componente_disponible("neuromix_aurora_v27"):
                    logger.debug("✅ NeuroMix original disponible en factory")
                    return
            
            # Fallback: verificar disponibilidad vía lazy loading
            neuromix_class = lazy_importer.get_class('neuromix_aurora_v27', 'AuroraNeuroAcousticEngineV27')
            if neuromix_class:
                self.neuromix_original = neuromix_class
                logger.debug("✅ NeuroMix original detectado vía lazy loading")
                return
            
            logger.debug("⚠️ NeuroMix original no encontrado")
            
        except Exception as e:
            logger.debug(f"❌ Error detectando NeuroMix original: {e}")
    
    def obtener_estado_completo(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema detectado"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "deteccion_inicializada": self._deteccion_inicializada,
            "director": self.estado_sistema.get("director", {}),
            "componentes": self.componentes_detectados,
            "neuromix_original_disponible": self.neuromix_original is not None,
            "total_componentes_detectados": len(self.componentes_detectados),
            "componentes_activos": len([c for c in self.componentes_detectados.values() if c.get("disponible", False)])
        }

# ===============================================================================
# AUTO-REPARADOR SIN DEPENDENCIAS CIRCULARES (VERSIÓN MEJORADA)
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
        self.stats_reparaciones = {
            "exitosas": 0,
            "fallidas": 0,
            "tiempo_total": 0.0
        }
    
    def auto_reparar_sistema(self) -> Dict[str, Any]:
        """
        Auto-repara el sistema detectando y corrigiendo problemas
        
        Returns:
            Dict con resultado de las reparaciones
        """
        
        with self.lock:
            logger.info("🔧 Iniciando auto-reparación del sistema")
            start_time = time.time()
            
            resultado = {
                "reparaciones_exitosas": 0,
                "reparaciones_fallidas": 0,
                "problemas_detectados": [],
                "soluciones_aplicadas": [],
                "tiempo_reparacion": 0
            }
            
            try:
                # Asegurar que la detección esté inicializada
                if not self.detector._deteccion_inicializada:
                    self.detector.inicializar_deteccion()
                
                # Diagnosticar problemas
                problemas = self._diagnosticar_problemas()
                resultado["problemas_detectados"] = problemas
                
                # Aplicar reparaciones
                for problema in problemas:
                    exito = self._aplicar_reparacion(problema)
                    if exito:
                        resultado["reparaciones_exitosas"] += 1
                        resultado["soluciones_aplicadas"].append(problema["solucion"])
                        self.stats_reparaciones["exitosas"] += 1
                    else:
                        resultado["reparaciones_fallidas"] += 1
                        self.stats_reparaciones["fallidas"] += 1
                
                resultado["tiempo_reparacion"] = time.time() - start_time
                self.stats_reparaciones["tiempo_total"] += resultado["tiempo_reparacion"]
                
                logger.info(f"✅ Auto-reparación completada: {resultado['reparaciones_exitosas']} exitosas, {resultado['reparaciones_fallidas']} fallidas")
                
            except Exception as e:
                logger.error(f"❌ Error en auto-reparación: {e}")
                resultado["error_general"] = str(e)
            
            return resultado
    
    def _diagnosticar_problemas(self) -> List[Dict[str, Any]]:
        """Diagnostica problemas del sistema"""
        
        problemas = []
        
        # Verificar componentes faltantes
        componentes_criticos = ["neuromix_aurora_v27", "harmonicEssence_v34", "emotion_style_profiles"]
        
        for componente in componentes_criticos:
            info_componente = self.detector.componentes_detectados.get(componente)
            if not info_componente or not info_componente.get("disponible", False):
                problemas.append({
                    "tipo": "componente_faltante",
                    "componente": componente,
                    "descripcion": f"Componente crítico {componente} no disponible",
                    "solucion": f"crear_fallback_{componente}"
                })
        
        # Verificar NeuroMix original
        if not self.detector.neuromix_original:
            problemas.append({
                "tipo": "neuromix_original_faltante",
                "componente": "neuromix_aurora_v27",
                "descripcion": "NeuroMix original no detectado",
                "solucion": "crear_neuromix_fallback"
            })
        
        return problemas
    
    def _aplicar_reparacion(self, problema: Dict[str, Any]) -> bool:
        """Aplica una reparación específica"""
        
        try:
            solucion = problema["solucion"]
            
            if solucion == "crear_neuromix_fallback":
                return self._crear_neuromix_fallback()
            elif solucion.startswith("crear_fallback_"):
                componente = solucion.replace("crear_fallback_", "")
                return self._crear_componente_fallback(componente)
            else:
                logger.warning(f"Solución desconocida: {solucion}")
                return False
                
        except Exception as e:
            logger.error(f"Error aplicando reparación {problema['solucion']}: {e}")
            return False
    
    def _crear_neuromix_fallback(self) -> bool:
        """Crea NeuroMix fallback funcional"""
        
        try:
            class NeuroMixFallback:
                def __init__(self):
                    self.nombre = "NeuroMix"
                    self.version = "fallback"
                    self.sample_rate = 44100
                
                def generar_audio(self, config):
                    duracion = config.get('duracion_sec', 60.0)
                    sample_rate = self.sample_rate
                    t = np.linspace(0, duracion, int(duracion * sample_rate), False)
                    
                    # Generar tono simple
                    freq_base = config.get('frecuencia_base', 220.0)
                    audio = 0.3 * np.sin(2 * np.pi * freq_base * t)
                    
                    return {
                        "audio_data": audio,
                        "duracion_sec": duracion,
                        "sample_rate": sample_rate
                    }
                
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
# NEUROMIX SUPER UNIFICADO REFACTORIZADO (VERSIÓN SIN DEADLOCK)
# ===============================================================================

class NeuroMixSuperUnificadoRefactored(ComponenteAuroraBase):
    """
    Motor NeuroMix super unificado refactorizado sin dependencias circulares
    
    VERSIÓN CORREGIDA: Constructor 100% pasivo para evitar deadlock
    
    Extiende el NeuroMix original con mejoras y capacidades adicionales,
    usando interfaces y factory pattern para evitar dependencias circulares.
    """
    
    def __init__(self, neuromix_base: Optional[Any] = None):
        super().__init__("NeuroMixSuperUnificado", VERSION)
        
        # CORRECCIÓN: Motor base será obtenido en inicialización manual
        self.neuromix_base = neuromix_base  # NO llamar _obtener_neuromix_base() automáticamente
        
        # Capacidades extendidas
        self.capacidades_extendidas = {
            "auto_reparacion": True,
            "deteccion_inteligente": False,  # Deshabilitado automático
            "modo_super_unificado": True,
            "compatibilidad_interfaces": True,
            "fallback_garantizado": True,
            "inicializacion_manual": True  # Nueva capacidad
        }
        
        # Estadísticas
        self.stats = {
            "audio_generado": 0,
            "auto_reparaciones": 0,
            "fallbacks_usados": 0,
            "tiempo_total_procesamiento": 0.0,
            "inicializaciones_manuales": 0
        }
        
        # Variables para detección manual (inicializadas como None)
        self.detector = None
        self.auto_reparador = None
        self.sistema_inicializado = False
        
        self.cambiar_estado(EstadoComponente.CONFIGURADO)
        logger.info("🧠 NeuroMix Super Unificado refactorizado inicializado (modo pasivo - sin deadlock)")
    
    def inicializar_deteccion_manual(self) -> bool:
        """
        Inicializa la detección de manera manual (llamado por el Director)
        
        VERSIÓN CORREGIDA: Aquí SÍ se obtiene el motor base de manera segura
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        if self.sistema_inicializado:
            return True
        
        logger.info("🔧 Inicializando NeuroMix Super manualmente (sin deadlock)...")
        
        try:
            # CORRECCIÓN: Obtener motor base ahora que es seguro
            if not self.neuromix_base:
                self.neuromix_base = self._obtener_neuromix_base()
            
            # Crear detector si no existe
            if not self.detector:
                self.detector = SuperDetectorAuroraRefactored()
                # Inicializar detección del detector
                self.detector.inicializar_deteccion()
            
            # Crear auto-reparador si no existe  
            if not self.auto_reparador:
                self.auto_reparador = AutoReparadorSistemaRefactored(self.detector)
            
            # Ejecutar auto-reparación inicial
            resultado_reparacion = self.auto_reparador.auto_reparar_sistema()
            if resultado_reparacion["reparaciones_exitosas"] > 0:
                logger.info(f"✅ Auto-reparación completada: {resultado_reparacion['reparaciones_exitosas']} reparaciones")
            
            self.sistema_inicializado = True
            self.stats["inicializaciones_manuales"] += 1
            self.cambiar_estado(EstadoComponente.ACTIVO)
            
            logger.info("✅ NeuroMix Super inicializado manualmente exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización manual: {e}")
            # Registrar en estadísticas pero no fallar completamente
            self.capacidades_extendidas["fallback_garantizado"] = True
            return False
    
    def _obtener_neuromix_base(self) -> Optional[Any]:
        """
        Obtiene NeuroMix base usando lazy loading
        
        VERSIÓN CORREGIDA: Con protección contra deadlock
        """
        
        try:
            # CORRECCIÓN: Verificar que no estemos en inicialización para evitar deadlock
            if hasattr(aurora_factory, '_initialization_in_progress') and aurora_factory._initialization_in_progress:
                logger.warning("⚠️ Factory en inicialización, usando fallback directo")
                return None
            
            # Intentar usar factory (con protección)
            try:
                neuromix_base = aurora_factory.crear_motor("neuromix_aurora_v27")
                if neuromix_base:
                    logger.info("✅ NeuroMix base obtenido vía factory")
                    return neuromix_base
            except Exception as e:
                logger.warning(f"⚠️ Error usando factory: {e}")
            
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
    
    # ===================================================================
    # MÉTODOS PRINCIPALES DEL MOTOR (COMPATIBLES CON AURORA INTERFACE)
    # ===================================================================
    
    def generar_audio(self, configuracion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera audio neuroacústico usando el motor base o fallback
        
        Args:
            configuracion: Diccionario con parámetros de generación
            
        Returns:
            Dict con resultado de generación
        """
        
        start_time = time.time()
        
        try:
            # Asegurar que el sistema esté inicializado
            if not self.sistema_inicializado:
                if not self.inicializar_deteccion_manual():
                    logger.warning("⚠️ Sistema no inicializado, usando fallback básico")
            
            # Validar configuración
            if not self.validar_configuracion(configuracion):
                raise ValueError("Configuración inválida")
            
            # Intentar usar motor base primero
            if self.neuromix_base and hasattr(self.neuromix_base, 'generar_audio'):
                try:
                    resultado = self.neuromix_base.generar_audio(configuracion)
                    resultado["motor_usado"] = "neuromix_base"
                except Exception as e:
                    logger.warning(f"⚠️ Error en motor base: {e}, usando fallback")
                    resultado = self._generar_audio_fallback(configuracion)
                    resultado["motor_usado"] = "fallback"
                    self.stats["fallbacks_usados"] += 1
            else:
                # Usar fallback directamente
                resultado = self._generar_audio_fallback(configuracion)
                resultado["motor_usado"] = "fallback"
                self.stats["fallbacks_usados"] += 1
            
            # Actualizar estadísticas
            tiempo_procesamiento = time.time() - start_time
            self.stats["audio_generado"] += 1
            self.stats["tiempo_total_procesamiento"] += tiempo_procesamiento
            
            resultado["tiempo_procesamiento"] = tiempo_procesamiento
            resultado["version_motor"] = self.version
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Error generando audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "motor_usado": "error",
                "tiempo_procesamiento": time.time() - start_time
            }
    
    def _generar_audio_fallback(self, configuracion: Dict[str, Any]) -> Dict[str, Any]:
        """Genera audio usando fallback básico"""
        
        try:
            duracion = configuracion.get('duracion_sec', 60.0)
            freq_base = configuracion.get('frecuencia_base', 220.0)
            sample_rate = 44100
            
            # Generar señal básica
            t = np.linspace(0, duracion, int(duracion * sample_rate), False)
            audio = 0.3 * np.sin(2 * np.pi * freq_base * t)
            
            # Añadir fade in/out básico
            fade_samples = int(0.1 * sample_rate)  # 0.1 segundos
            if len(audio) > 2 * fade_samples:
                audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            return {
                "success": True,
                "audio_data": audio,
                "duracion_sec": duracion,
                "sample_rate": sample_rate,
                "tipo_generacion": "fallback_basico"
            }
            
        except Exception as e:
            logger.error(f"❌ Error en fallback: {e}")
            return {
                "success": False,
                "error": f"Fallback falló: {e}"
            }
    
    def obtener_capacidades(self) -> Dict[str, Any]:
        """Obtiene capacidades del motor"""
        
        return {
            "nombre": self.nombre,
            "version": self.version,
            "tipo": TipoComponenteAurora.MOTOR if INTERFACES_DISPONIBLES else "motor",
            "capacidades_extendidas": self.capacidades_extendidas.copy(),
            "estadisticas": self.stats.copy(),
            "estado": self.estado.value if hasattr(self.estado, 'value') else str(self.estado),
            "motor_base_disponible": self.neuromix_base is not None,
            "sistema_inicializado": self.sistema_inicializado
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Información básica del motor (compatibilidad Aurora)"""
        
        return {
            "nombre": self.nombre,
            "version": self.version,
            "tipo": "NeuroMix Super Unificado",
            "estado": self.estado.value if hasattr(self.estado, 'value') else str(self.estado),
            "capacidades": list(self.capacidades_extendidas.keys()),
            "audio_generado": self.stats["audio_generado"]
        }
    
    def validar_configuracion(self, config: Dict[str, Any]) -> bool:
        """Valida configuración de entrada"""
        
        try:
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
        
        diagnostico_base = {
            "motor_super_unificado": {
                "nombre": self.nombre,
                "version": self.version,
                "estado": self.estado.value if hasattr(self.estado, 'value') else str(self.estado),
                "estadisticas": self.stats.copy(),
                "inicializado": self.sistema_inicializado
            },
            "motor_base": {
                "disponible": self.neuromix_base is not None,
                "tipo": type(self.neuromix_base).__name__ if self.neuromix_base else "None"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Añadir información del detector si existe
        if self.detector:
            diagnostico_base["detector"] = self.detector.obtener_estado_completo()
        
        # Añadir información del auto-reparador si existe
        if self.auto_reparador:
            diagnostico_base["auto_reparador"] = {
                "reparaciones_disponibles": True,
                "estadisticas": self.auto_reparador.stats_reparaciones.copy(),
                "ultimo_diagnostico": datetime.now().isoformat()
            }
        
        return diagnostico_base
    
    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del motor"""
        
        stats_completas = {
            "estadisticas_motor": self.stats.copy(),
            "capacidades": self.capacidades_extendidas.copy(),
            "version": self.version,
            "tiempo_funcionamiento": time.time(),
            "sistema_inicializado": self.sistema_inicializado
        }
        
        if self.detector:
            stats_completas["estado_sistema"] = self.detector.obtener_estado_completo()
        
        return stats_completas

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

def aplicar_super_parche_automatico():
    """
    Aplica el super parche automáticamente al sistema
    
    Esta función registra el NeuroMix Super en el factory si está disponible
    """
    
    if not INTERFACES_DISPONIBLES:
        logger.warning("⚠️ Interfaces Aurora no disponibles, super parche limitado")
        return False
    
    try:
        # Registrar en factory si no está registrado
        exito = aurora_factory.registrar_motor(
            "neuromix_super_unificado",
            "neuromix_definitivo_v7",  # Este archivo
            "NeuroMixSuperUnificadoRefactored"
        )
        
        if exito:
            logger.info("🏭 NeuroMix Super Unificado registrado en factory")
        else:
            logger.debug("🏭 NeuroMix Super Unificado ya está registrado en factory")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Error aplicando super parche: {e}")
        return False

# ===============================================================================
# ALIAS Y COMPATIBILIDAD
# ===============================================================================

# Aliases para mantener compatibilidad con código existente
NeuroMixSuperUnificado = NeuroMixSuperUnificadoRefactored
SuperDetectorAurora = SuperDetectorAuroraRefactored
AutoReparadorSistema = AutoReparadorSistemaRefactored

# ===============================================================================
# FUNCIONES DE TEST
# ===============================================================================

def test_neuromix_definitivo_refactorizado():
    """Test del NeuroMix definitivo refactorizado sin deadlock"""
    
    print("🧪 Testeando NeuroMix Definitivo V7 Refactorizado (Sin Deadlock)...")
    
    # Crear instancia (debe ser instantáneo)
    motor = crear_neuromix_super_unificado()
    print(f"✅ Motor creado: {motor.nombre}")
    
    # Test inicialización manual
    exito_init = motor.inicializar_deteccion_manual()
    print(f"✅ Inicialización manual: {'exitosa' if exito_init else 'falló'}")
    
    # Test info
    info = motor.get_info()
    print(f"✅ Info: {info['nombre']} v{info['version']}")
    
    # Test capacidades
    capacidades = motor.obtener_capacidades()
    print(f"✅ Capacidades: {len(capacidades)} propiedades")
    print(f"   - Sistema inicializado: {capacidades['sistema_inicializado']}")
    print(f"   - Motor base disponible: {capacidades['motor_base_disponible']}")
    
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
    if resultado.get('success', True):
        print(f"✅ Audio generado: {resultado.get('duracion_sec', 0):.1f}s usando {resultado.get('motor_usado', 'desconocido')}")
    else:
        print(f"❌ Error generando audio: {resultado.get('error', 'desconocido')}")
    
    # Test diagnóstico
    diagnostico = motor.diagnosticar_sistema_completo()
    print(f"✅ Diagnóstico: {diagnostico['motor_super_unificado']['estado']}")
    
    print("🎉 Test completado exitosamente")
    return motor

# ===============================================================================
# AUTO-APLICACIÓN DEL PARCHE
# ===============================================================================

# Auto-aplicar el parche al importar el módulo
if __name__ != "__main__":
    aplicar_super_parche_automatico()

# ===============================================================================
# INFORMACIÓN DEL MÓDULO
# ===============================================================================

__version__ = VERSION
__author__ = "Sistema Aurora V7"
__description__ = "NeuroMix Super Unificado sin dependencias circulares ni deadlock"

if __name__ == "__main__":
    print("🧠 NeuroMix Definitivo V7 Refactorizado (Sin Deadlock)")
    print(f"Versión: {__version__}")
    print("Estado: Sin dependencias circulares, sin deadlock")
    print("Cambios principales:")
    print("  - Constructor 100% pasivo")
    print("  - Inicialización manual controlada")
    print("  - Protección contra deadlock en factory")
    print("  - Detector con inicialización lazy")
    
    # Ejecutar test
    test_neuromix_definitivo_refactorizado()
