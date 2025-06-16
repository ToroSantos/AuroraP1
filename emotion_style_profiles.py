#!/usr/bin/env python3
"""
🎨 EMOTION STYLE PROFILES V7 - REFACTORIZADO SIN DEPENDENCIAS CIRCULARES
======================================================================

Gestor de perfiles emocionales y estilos para el Sistema Aurora V7.
Versión refactorizada que utiliza interfaces y lazy loading para eliminar
dependencias circulares con harmonicEssence_v34 y otros motores.

Versión: V7.1 - Refactorizado
Propósito: Gestión de perfiles emocionales sin dependencias circulares
Compatibilidad: Aurora Director V7, HyperMod V32
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import logging
import math
import json
import warnings
import time
from datetime import datetime

# Importar interfaces Aurora (sin ciclos)
try:
    from aurora_interfaces import (
        GestorPerfilesInterface, MotorAuroraInterface, ComponenteAuroraBase,
        TipoComponenteAurora, EstadoComponente, NivelCalidad,
        validar_motor_aurora, validar_gestor_perfiles
    )
    from aurora_factory import aurora_factory, lazy_importer
    INTERFACES_DISPONIBLES = True
except ImportError as e:
    logging.warning(f"Aurora interfaces no disponibles: {e}")
    INTERFACES_DISPONIBLES = False
    
    # Clases dummy para compatibilidad
    class GestorPerfilesInterface: pass
    class MotorAuroraInterface: pass
    class ComponenteAuroraBase: pass

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aurora.EmotionStyle.V7.Refactored")

VERSION = "V7.1_REFACTORED_NO_CIRCULAR_DEPS"

# ===============================================================================
# ENUMS Y CONFIGURACIONES BASE (Sin cambios para compatibilidad)
# ===============================================================================

class CategoriaEmocional(Enum):
    COGNITIVO = "cognitivo"
    EMOCIONAL = "emocional"
    SOCIAL = "social"
    CREATIVO = "creativo"
    ESPIRITUAL = "espiritual"
    TERAPEUTICO = "terapeutico"
    PERFORMANCE = "performance"
    EXPERIMENTAL = "experimental"
    RELAJACION = "relajacion"
    CONCENTRACION = "concentracion"
    CREATIVIDAD = "creatividad"
    ENERGIA = "energia"
    SANACION = "sanacion"
    EQUILIBRIO = "equilibrio"

class CategoriaEstilo(Enum):
    MEDITATIVO = "meditativo"
    ENERGIZANTE = "energizante"
    CREATIVO = "creativo"
    TERAPEUTICO = "terapeutico"
    AMBIENTAL = "ambiental"
    EXPERIMENTAL = "experimental"
    TRADICIONAL = "tradicional"
    FUTURISTA = "futurista"
    ORGANICO = "organico"
    ESPIRITUAL = "espiritual"

class NivelIntensidad(Enum):
    SUAVE = "suave"
    SUTIL = "sutil"
    MODERADO = "moderado"
    INTENSO = "intenso"
    PROFUNDO = "profundo"
    TRASCENDENTE = "trascendente"

class TipoPad(Enum):
    SUAVE = "suave"
    CRISTALINO = "cristalino"
    ORGANICO = "organico"
    ESPACIAL = "espacial"
    NEURONAL = "neuronal"
    AMBIENTAL = "ambiental"

class EstiloRuido(Enum):
    ROSA = "rosa"
    BLANCO = "blanco"
    MARRON = "marron"
    NATURAL = "natural"
    ESPACIAL = "espacial"

# ===============================================================================
# CLASES DE DATOS PARA PERFILES
# ===============================================================================

@dataclass
class PresetEmocional:
    """Configuración emocional base"""
    nombre: str
    categoria: CategoriaEmocional
    intensidad: NivelIntensidad
    tipo_pad: TipoPad
    estilo_ruido: EstiloRuido
    efectos_esperados: List[str]
    descripcion: str = ""
    neurotransmisores: Dict[str, float] = field(default_factory=dict)
    frecuencias_objetivo: Dict[str, float] = field(default_factory=dict)
    duracion_recomendada: float = 300.0
    configuracion_motor: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para compatibilidad"""
        return {
            "nombre": self.nombre,
            "categoria": self.categoria.value,
            "intensidad": self.intensidad.value,
            "tipo_pad": self.tipo_pad.value,
            "estilo_ruido": self.estilo_ruido.value,
            "efectos_esperados": self.efectos_esperados,
            "descripcion": self.descripcion,
            "neurotransmisores": self.neurotransmisores,
            "frecuencias_objetivo": self.frecuencias_objetivo,
            "duracion_recomendada": self.duracion_recomendada,
            "configuracion_motor": self.configuracion_motor
        }

@dataclass
class PerfilEstilo:
    """Perfil de estilo estético"""
    nombre: str
    categoria: CategoriaEstilo
    configuracion_textura: Dict[str, Any]
    parametros_generacion: Dict[str, Any] = field(default_factory=dict)
    efectos_recomendados: List[str] = field(default_factory=list)
    compatibilidad_motores: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para compatibilidad"""
        return {
            "nombre": self.nombre,
            "categoria": self.categoria.value,
            "configuracion_textura": self.configuracion_textura,
            "parametros_generacion": self.parametros_generacion,
            "efectos_recomendados": self.efectos_recomendados,
            "compatibilidad_motores": self.compatibilidad_motores
        }

# ===============================================================================
# PRESETS EMOCIONALES BASE (Estructura compatible)
# ===============================================================================

PRESETS_EMOCIONALES_AURORA = {
    "relajacion_profunda": PresetEmocional(
        nombre="relajacion_profunda",
        categoria=CategoriaEmocional.RELAJACION,
        intensidad=NivelIntensidad.SUAVE,
        tipo_pad=TipoPad.SUAVE,
        estilo_ruido=EstiloRuido.ROSA,
        efectos_esperados=["calma", "serenidad", "reduccion_estres"],
        descripcion="Relajación profunda con ondas alfa y theta",
        neurotransmisores={"serotonina": 0.8, "gaba": 0.7},
        frecuencias_objetivo={"alfa": 10.0, "theta": 6.0},
        configuracion_motor={
            "tipo_textura": "ambiental",
            "modo_generacion": "terapeutico",
            "complejidad": "simple",
            "intensidad": 0.3,
            "reverb_factor": 0.5,
            "spatial_width": 0.8
        }
    ),
    
    "enfoque_mental": PresetEmocional(
        nombre="enfoque_mental",
        categoria=CategoriaEmocional.COGNITIVO,
        intensidad=NivelIntensidad.MODERADO,
        tipo_pad=TipoPad.CRISTALINO,
        estilo_ruido=EstiloRuido.BLANCO,
        efectos_esperados=["concentracion", "claridad_mental", "productividad"],
        descripcion="Mejora del enfoque con ondas beta y gamma",
        neurotransmisores={"dopamina": 0.7, "noradrenalina": 0.6},
        frecuencias_objetivo={"beta": 15.0, "gamma": 40.0},
        configuracion_motor={
            "tipo_textura": "neuronal",
            "modo_generacion": "neuroacustico",
            "complejidad": "moderado",
            "intensidad": 0.6,
            "reverb_factor": 0.2,
            "spatial_width": 0.4
        }
    ),
    
    "creatividad_flow": PresetEmocional(
        nombre="creatividad_flow",
        categoria=CategoriaEmocional.CREATIVO,
        intensidad=NivelIntensidad.INTENSO,
        tipo_pad=TipoPad.ORGANICO,
        estilo_ruido=EstiloRuido.NATURAL,
        efectos_esperados=["creatividad", "inspiracion", "flow_state"],
        descripcion="Estado de flow creativo con ondas theta y alfa",
        neurotransmisores={"dopamina": 0.8, "serotonina": 0.6, "acetilcolina": 0.7},
        frecuencias_objetivo={"theta": 7.0, "alfa": 9.0},
        configuracion_motor={
            "tipo_textura": "organica",
            "modo_generacion": "experimental",
            "complejidad": "complejo",
            "intensidad": 0.7,
            "reverb_factor": 0.6,
            "spatial_width": 0.9
        }
    ),
    
    "energia_vitalizante": PresetEmocional(
        nombre="energia_vitalizante",
        categoria=CategoriaEmocional.ENERGIA,
        intensidad=NivelIntensidad.PROFUNDO,
        tipo_pad=TipoPad.ESPACIAL,
        estilo_ruido=EstiloRuido.ESPACIAL,
        efectos_esperados=["energia", "vitalidad", "motivacion"],
        descripcion="Energización con ondas beta altas y gamma",
        neurotransmisores={"dopamina": 0.9, "noradrenalina": 0.8},
        frecuencias_objetivo={"beta": 20.0, "gamma": 30.0},
        configuracion_motor={
            "tipo_textura": "cristalina",
            "modo_generacion": "estetico",
            "complejidad": "avanzado",
            "intensidad": 0.8,
            "reverb_factor": 0.3,
            "spatial_width": 0.5
        }
    ),
    
    "sanacion_emocional": PresetEmocional(
        nombre="sanacion_emocional",
        categoria=CategoriaEmocional.TERAPEUTICO,
        intensidad=NivelIntensidad.SUTIL,
        tipo_pad=TipoPad.AMBIENTAL,
        estilo_ruido=EstiloRuido.ROSA,
        efectos_esperados=["sanacion", "equilibrio_emocional", "paz_interior"],
        descripcion="Sanación emocional con frecuencias específicas",
        neurotransmisores={"serotonina": 0.7, "oxitocina": 0.6, "endorfinas": 0.5},
        frecuencias_objetivo={"delta": 2.0, "theta": 4.0, "alfa": 8.0},
        configuracion_motor={
            "tipo_textura": "ambiental",
            "modo_generacion": "terapeutico",
            "complejidad": "simple",
            "intensidad": 0.4,
            "reverb_factor": 0.7,
            "spatial_width": 1.0
        }
    ),
    
    "meditacion_trascendente": PresetEmocional(
        nombre="meditacion_trascendente",
        categoria=CategoriaEmocional.ESPIRITUAL,
        intensidad=NivelIntensidad.TRASCENDENTE,
        tipo_pad=TipoPad.NEURONAL,
        estilo_ruido=EstiloRuido.MARRON,
        efectos_esperados=["trascendencia", "conexion_espiritual", "expansion_consciencia"],
        descripcion="Estados trascendentes con ondas delta y theta profundas",
        neurotransmisores={"serotonina": 0.9, "dmt_endogeno": 0.3, "endorfinas": 0.8},
        frecuencias_objetivo={"delta": 1.5, "theta": 3.5},
        configuracion_motor={
            "tipo_textura": "espacial",
            "modo_generacion": "experimental",
            "complejidad": "extremo",
            "intensidad": 0.5,
            "reverb_factor": 0.9,
            "spatial_width": 1.2
        }
    )
}

# ===============================================================================
# PERFILES DE ESTILO (Estructura compatible)
# ===============================================================================

PERFILES_ESTILO_AURORA = {
    "ambiental_suave": PerfilEstilo(
        nombre="ambiental_suave",
        categoria=CategoriaEstilo.AMBIENTAL,
        configuracion_textura={
            "tipo_textura": "ambiental",
            "intensidad": 0.3,
            "reverb_factor": 0.6,
            "efectos_extras": {"chorus": {"depth": 0.3, "rate_hz": 0.2}}
        },
        parametros_generacion={"complejidad": "simple", "modo": "terapeutico"},
        efectos_recomendados=["reverb", "chorus"],
        compatibilidad_motores=["harmonicEssence", "neuromix"]
    ),
    
    "neuronal_cristalino": PerfilEstilo(
        nombre="neuronal_cristalino",
        categoria=CategoriaEstilo.FUTURISTA,
        configuracion_textura={
            "tipo_textura": "cristalina",
            "intensidad": 0.7,
            "reverb_factor": 0.2,
            "efectos_extras": {"delay": {"delay_ms": 200, "feedback": 0.3}}
        },
        parametros_generacion={"complejidad": "complejo", "modo": "neuroacustico"},
        efectos_recomendados=["delay", "spatial_processing"],
        compatibilidad_motores=["harmonicEssence", "hypermod"]
    ),
    
    "organico_natural": PerfilEstilo(
        nombre="organico_natural",
        categoria=CategoriaEstilo.ORGANICO,
        configuracion_textura={
            "tipo_textura": "organica",
            "intensidad": 0.5,
            "reverb_factor": 0.4,
            "efectos_extras": {"filtro_emocional": "calido"}
        },
        parametros_generacion={"complejidad": "moderado", "modo": "estetico"},
        efectos_recomendados=["filtro_emocional", "spatial_processing"],
        compatibilidad_motores=["harmonicEssence", "neuromix", "hypermod"]
    ),
    
    "espacial_envolvente": PerfilEstilo(
        nombre="espacial_envolvente",
        categoria=CategoriaEstilo.EXPERIMENTAL,
        configuracion_textura={
            "tipo_textura": "espacial",
            "intensidad": 0.6,
            "reverb_factor": 0.8,
            "spatial_width": 1.0,
            "efectos_extras": {"chorus": {"depth": 0.5, "rate_hz": 0.1}}
        },
        parametros_generacion={"complejidad": "avanzado", "modo": "experimental"},
        efectos_recomendados=["reverb", "spatial_processing", "chorus"],
        compatibilidad_motores=["harmonicEssence"]
    )
}

# ===============================================================================
# GESTOR PRINCIPAL REFACTORIZADO
# ===============================================================================

class GestorPresetsEmocionalesRefactored(ComponenteAuroraBase):
    """
    Gestor de presets emocionales refactorizado sin dependencias circulares
    
    Implementa GestorPerfilesInterface y utiliza lazy loading para
    obtener capacidades de motores sin importarlos directamente.
    """
    
    def __init__(self):
        super().__init__("GestorPresetsEmocionales", "V7.1-Refactored")
        
        # Cargar presets base
        self.presets = PRESETS_EMOCIONALES_AURORA.copy()
        self.perfiles_estilo = PERFILES_ESTILO_AURORA.copy()
        
        # Cache para optimización
        self._cache_capacidades_motores = {}
        self._cache_perfiles_adaptados = {}
        
        # Estadísticas
        self.stats = {
            "perfiles_servidos": 0,
            "adaptaciones_realizadas": 0,
            "cache_hits": 0,
            "errores": 0
        }
        
        self.cambiar_estado(EstadoComponente.CONFIGURADO)
        logger.info("🎨 Gestor de presets emocionales refactorizado inicializado")
    
    # ===================================================================
    # IMPLEMENTACIÓN DE GestorPerfilesInterface
    # ===================================================================
    
    def obtener_perfil(self, nombre: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un perfil por nombre (implementa GestorPerfilesInterface)
        
        Args:
            nombre: Nombre del perfil a obtener
            
        Returns:
            Dict con configuración del perfil o None si no existe
        """
        try:
            # Buscar en presets emocionales
            if nombre in self.presets:
                preset = self.presets[nombre]
                perfil = preset.to_dict()
                
                # Adaptar perfil según capacidades disponibles
                perfil_adaptado = self._adaptar_perfil_a_capacidades(perfil)
                
                self.stats["perfiles_servidos"] += 1
                self.log(f"Perfil '{nombre}' servido")
                
                return perfil_adaptado
            
            # Buscar en perfiles de estilo
            if nombre in self.perfiles_estilo:
                perfil_estilo = self.perfiles_estilo[nombre]
                perfil = perfil_estilo.to_dict()
                
                self.stats["perfiles_servidos"] += 1
                self.log(f"Perfil de estilo '{nombre}' servido")
                
                return perfil
            
            # Buscar por alias o variaciones
            perfil_encontrado = self._buscar_perfil_por_alias(nombre)
            if perfil_encontrado:
                self.stats["perfiles_servidos"] += 1
                return perfil_encontrado
            
            self.log(f"Perfil '{nombre}' no encontrado", "WARNING")
            return None
            
        except Exception as e:
            self.stats["errores"] += 1
            self.log(f"Error obteniendo perfil '{nombre}': {e}", "ERROR")
            return None
    
    def listar_perfiles(self) -> List[str]:
        """
        Lista todos los perfiles disponibles
        
        Returns:
            Lista de nombres de perfiles
        """
        perfiles = list(self.presets.keys()) + list(self.perfiles_estilo.keys())
        return sorted(perfiles)
    
    def validar_perfil(self, nombre: str) -> bool:
        """
        Valida si un perfil es correcto
        
        Args:
            perfil: Configuración del perfil
            
        Returns:
            True si el perfil es válido
        """
        try:
            # Validaciones básicas
            if not isinstance(perfil, dict):
                return False
            
            # Debe tener nombre
            if "nombre" not in perfil:
                return False
            
            # Validar configuración de motor si existe
            if "configuracion_motor" in perfil:
                config_motor = perfil["configuracion_motor"]
                if not isinstance(config_motor, dict):
                    return False
                
                # Validar parámetros básicos
                if "intensidad" in config_motor:
                    if not 0.0 <= config_motor["intensidad"] <= 1.0:
                        return False
            
            return True
            
        except Exception as e:
            self.log(f"Error validando perfil: {e}", "ERROR")
            return False
    
    def obtener_categoria(self) -> str:
        """Categoría del gestor de perfiles"""
        return "emotion_style"
    
    # ===================================================================
    # MÉTODOS DE ADAPTACIÓN SIN DEPENDENCIAS CIRCULARES
    # ===================================================================
    
    def _adaptar_perfil_a_capacidades(self, perfil: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapta el perfil según las capacidades disponibles de los motores
        usando lazy loading sin crear dependencias circulares
        
        Args:
            perfil: Perfil original
            
        Returns:
            Perfil adaptado
        """
        # Usar cache si está disponible
        perfil_hash = hash(str(perfil))
        if perfil_hash in self._cache_perfiles_adaptados:
            self.stats["cache_hits"] += 1
            return self._cache_perfiles_adaptados[perfil_hash]
        
        try:
            perfil_adaptado = perfil.copy()
            
            # Obtener capacidades de motores disponibles
            capacidades_motores = self._obtener_capacidades_motores_lazy()
            
            # Adaptar configuración de motor si hay capacidades disponibles
            if "configuracion_motor" in perfil_adaptado and capacidades_motores:
                config_motor = perfil_adaptado["configuracion_motor"]
                config_adaptada = self._adaptar_configuracion_motor(config_motor, capacidades_motores)
                perfil_adaptado["configuracion_motor"] = config_adaptada
            
            # Adaptar efectos según motores disponibles
            if "efectos_recomendados" in perfil_adaptado and capacidades_motores:
                efectos_adaptados = self._adaptar_efectos_disponibles(
                    perfil_adaptado["efectos_recomendados"], 
                    capacidades_motores
                )
                perfil_adaptado["efectos_adaptados"] = efectos_adaptados
            
            # Cachear resultado
            self._cache_perfiles_adaptados[perfil_hash] = perfil_adaptado
            self.stats["adaptaciones_realizadas"] += 1
            
            return perfil_adaptado
            
        except Exception as e:
            self.log(f"Error adaptando perfil: {e}", "WARNING")
            return perfil  # Retornar original si falla la adaptación
    
    def _obtener_capacidades_motores_lazy(self) -> Dict[str, Any]:
        """
        Obtiene capacidades de motores usando lazy loading sin dependencias circulares
        
        Returns:
            Dict con capacidades de motores disponibles
        """
        # Usar cache si está disponible
        if self._cache_capacidades_motores:
            return self._cache_capacidades_motores
        
        capacidades = {}
        
        try:
            # Lista de motores a consultar
            motores_objetivo = [
                ("harmonicEssence_v34", "HarmonicEssenceV34Refactored"),
                ("neuromix_aurora_v27", "AuroraNeuroAcousticEngineV27"),
                ("hypermod_v32", "HyperModEngineV32AuroraConnected")
            ]
            
            for modulo_nombre, clase_nombre in motores_objetivo:
                try:
                    # Intentar crear motor usando factory
                    motor = aurora_factory.crear_motor(modulo_nombre)
                    if motor and hasattr(motor, 'obtener_capacidades'):
                        capacidades[modulo_nombre] = motor.obtener_capacidades()
                        self.log(f"Capacidades de {modulo_nombre} obtenidas vía factory")
                        continue
                    
                    # Fallback: lazy loading directo
                    motor_class = lazy_importer.get_class(modulo_nombre, clase_nombre)
                    if motor_class:
                        # Crear instancia temporal solo para obtener capacidades
                        motor_temp = motor_class()
                        if hasattr(motor_temp, 'obtener_capacidades'):
                            capacidades[modulo_nombre] = motor_temp.obtener_capacidades()
                            self.log(f"Capacidades de {modulo_nombre} obtenidas vía lazy loading")
                        del motor_temp  # Limpiar inmediatamente
                    
                except Exception as e:
                    self.log(f"No se pudieron obtener capacidades de {modulo_nombre}: {e}", "DEBUG")
                    continue
            
            # Cachear resultados
            self._cache_capacidades_motores = capacidades
            
        except Exception as e:
            self.log(f"Error obteniendo capacidades de motores: {e}", "WARNING")
        
        return capacidades
    
    def _adaptar_configuracion_motor(self, config: Dict[str, Any], capacidades: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapta configuración de motor según capacidades disponibles
        
        Args:
            config: Configuración original
            capacidades: Capacidades de motores disponibles
            
        Returns:
            Configuración adaptada
        """
        config_adaptada = config.copy()
        
        # Adaptar según HarmonicEssence si está disponible
        if "harmonicEssence_v34" in capacidades:
            he_caps = capacidades["harmonicEssence_v34"]
            
            # Validar tipo de textura
            if "tipo_textura" in config_adaptada:
                tipos_disponibles = he_caps.get("tipos_textura", [])
                if config_adaptada["tipo_textura"] not in tipos_disponibles and tipos_disponibles:
                    config_adaptada["tipo_textura"] = tipos_disponibles[0]
                    self.log("Tipo de textura adaptado según capacidades disponibles")
            
            # Validar efectos
            if "efectos_extras" in config_adaptada:
                efectos_disponibles = he_caps.get("efectos_disponibles", [])
                efectos_adaptados = {}
                for efecto, params in config_adaptada["efectos_extras"].items():
                    if efecto in efectos_disponibles:
                        efectos_adaptados[efecto] = params
                
                config_adaptada["efectos_extras"] = efectos_adaptados
        
        return config_adaptada
    
    def _adaptar_efectos_disponibles(self, efectos: List[str], capacidades: Dict[str, Any]) -> List[str]:
        """
        Adapta lista de efectos según lo que esté disponible en los motores
        
        Args:
            efectos: Lista de efectos deseados
            capacidades: Capacidades de motores
            
        Returns:
            Lista de efectos disponibles
        """
        efectos_adaptados = []
        
        for motor, caps in capacidades.items():
            efectos_motor = caps.get("efectos_disponibles", [])
            for efecto in efectos:
                if efecto in efectos_motor and efecto not in efectos_adaptados:
                    efectos_adaptados.append(efecto)
        
        return efectos_adaptados
    
    def _buscar_perfil_por_alias(self, nombre: str) -> Optional[Dict[str, Any]]:
        """
        Busca perfil por alias o nombres similares
        
        Args:
            nombre: Nombre o alias a buscar
            
        Returns:
            Perfil encontrado o None
        """
        nombre_lower = nombre.lower()
        
        # Mapeo de alias comunes
        alias_mapping = {
            "relax": "relajacion_profunda",
            "relajacion": "relajacion_profunda",
            "focus": "enfoque_mental",
            "enfoque": "enfoque_mental",
            "concentracion": "enfoque_mental",
            "creative": "creatividad_flow",
            "creatividad": "creatividad_flow",
            "energia": "energia_vitalizante",
            "energy": "energia_vitalizante",
            "healing": "sanacion_emocional",
            "sanacion": "sanacion_emocional",
            "meditation": "meditacion_trascendente",
            "meditacion": "meditacion_trascendente"
        }
        
        if nombre_lower in alias_mapping:
            return self.obtener_perfil(alias_mapping[nombre_lower])
        
        # Búsqueda por coincidencia parcial
        for preset_name in self.presets.keys():
            if nombre_lower in preset_name.lower():
                return self.obtener_perfil(preset_name)
        
        return None
    
    # ===================================================================
    # MÉTODOS DE UTILIDAD Y COMPATIBILIDAD
    # ===================================================================
    
    def obtener_preset(self, nombre: str) -> Optional[PresetEmocional]:
        """Método de compatibilidad para obtener preset como objeto"""
        if nombre in self.presets:
            return self.presets[nombre]
        return None
    
    def buscar_por_categoria(self, categoria: Union[str, CategoriaEmocional]) -> List[str]:
        """
        Busca presets por categoría emocional
        
        Args:
            categoria: Categoría a buscar
            
        Returns:
            Lista de nombres de presets
        """
        if isinstance(categoria, str):
            categoria_str = categoria.lower()
        else:
            categoria_str = categoria.value
        
        resultados = []
        for nombre, preset in self.presets.items():
            if preset.categoria.value == categoria_str:
                resultados.append(nombre)
        
        return resultados
    
    def buscar_por_efecto(self, efecto: str) -> List[str]:
        """
        Busca presets que produzcan un efecto específico
        
        Args:
            efecto: Efecto deseado
            
        Returns:
            Lista de nombres de presets
        """
        efecto_lower = efecto.lower()
        resultados = []
        
        for nombre, preset in self.presets.items():
            efectos_preset = [e.lower() for e in preset.efectos_esperados]
            if any(efecto_lower in e for e in efectos_preset):
                resultados.append(nombre)
        
        return resultados
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor"""
        return {
            "total_presets": len(self.presets),
            "total_perfiles_estilo": len(self.perfiles_estilo),
            "categorias_disponibles": len(set(p.categoria for p in self.presets.values())),
            "motores_con_capacidades": len(self._cache_capacidades_motores),
            "estadisticas_uso": self.stats.copy(),
            "version": self.version,
            "estado": self.estado.value
        }
    
    def limpiar_cache(self):
        """Limpia todos los caches"""
        self._cache_capacidades_motores.clear()
        self._cache_perfiles_adaptados.clear()
        self.log("Cache limpiado")

# ===============================================================================
# GESTORES ADICIONALES PARA COMPATIBILIDAD
# ===============================================================================

class GestorPerfilesEstiloRefactored(ComponenteAuroraBase):
    """Gestor específico para perfiles de estilo"""
    
    def __init__(self):
        super().__init__("GestorPerfilesEstilo", "V7.1-Refactored")
        self.perfiles = PERFILES_ESTILO_AURORA.copy()
        self.cambiar_estado(EstadoComponente.CONFIGURADO)
    
    def obtener_perfil(self, nombre: str) -> Optional[Dict[str, Any]]:
        """Implementa GestorPerfilesInterface para perfiles de estilo"""
        if nombre in self.perfiles:
            return self.perfiles[nombre].to_dict()
        return None
    
    def listar_perfiles(self) -> List[str]:
        """Lista perfiles de estilo disponibles"""
        return list(self.perfiles.keys())
    
    def validar_perfil(self, nombre: str) -> bool:
        """Valida si un perfil existe y es válido"""
        return nombre in self.presets or nombre in self.perfiles_estilo

# ===============================================================================
# FUNCIONES DE FACTORY Y COMPATIBILIDAD
# ===============================================================================

def crear_gestor_presets() -> GestorPresetsEmocionalesRefactored:
    """Factory function para crear gestor de presets"""
    return GestorPresetsEmocionalesRefactored()

def crear_gestor_estilos() -> GestorPerfilesEstiloRefactored:
    """Factory function para crear gestor de estilos"""
    return GestorPerfilesEstiloRefactored()

def crear_gestor_estilos_esteticos() -> GestorPerfilesEstiloRefactored:
    """Alias de compatibilidad"""
    return crear_gestor_estilos()

def crear_gestor_fases():
    """Factory para gestor de fases (fallback básico)"""
    class GestorFasesFallback:
        def obtener_perfil(self, nombre: str):
            return {"nombre": nombre, "tipo": "fase_basica"}
        def listar_perfiles(self):
            return ["intro", "desarrollo", "climax", "resolucion"]
        def validar_perfil(self, perfil):
            return True
    
    return GestorFasesFallback()

def crear_gestor_optimizado() -> GestorPresetsEmocionalesRefactored:
    """Factory para gestor optimizado"""
    return crear_gestor_presets()

def obtener_perfil_estilo(nombre: str) -> Optional[Dict[str, Any]]:
    """Función de utilidad para obtener perfil de estilo"""
    if nombre in PERFILES_ESTILO_AURORA:
        return PERFILES_ESTILO_AURORA[nombre].to_dict()
    return None

def listar_estilos_disponibles() -> List[str]:
    """Lista todos los estilos disponibles"""
    return list(PERFILES_ESTILO_AURORA.keys())

def obtener_preset_emocional(nombre: str) -> Optional[PresetEmocional]:
    """Función de utilidad para obtener preset emocional"""
    if nombre in PRESETS_EMOCIONALES_AURORA:
        return PRESETS_EMOCIONALES_AURORA[nombre]
    return None

def listar_presets_disponibles() -> List[str]:
    """Lista todos los presets emocionales disponibles"""
    return list(PRESETS_EMOCIONALES_AURORA.keys())

# ===============================================================================
# INSTANCIA GLOBAL PARA COMPATIBILIDAD
# ===============================================================================

# Instancia global para compatibilidad con código existente
_gestor_global = None

def obtener_gestor_global() -> GestorPresetsEmocionalesRefactored:
    """Obtiene instancia global del gestor"""
    global _gestor_global
    if _gestor_global is None:
        _gestor_global = crear_gestor_presets()
    return _gestor_global

# ===============================================================================
# VARIABLES GLOBALES PARA COMPATIBILIDAD LEGACY
# ===============================================================================

# Variables globales que otros módulos pueden estar esperando
presets_emocionales = PRESETS_EMOCIONALES_AURORA
GestorPresetsEmocionales = GestorPresetsEmocionalesRefactored

# ===============================================================================
# AUTO-REGISTRO EN FACTORY
# ===============================================================================

if INTERFACES_DISPONIBLES:
    try:
        aurora_factory.registrar_gestor(
            "emotion_style_profiles",
            "emotion_style_profiles",  # Este archivo
            "GestorPresetsEmocionalesRefactored"
        )
        logger.info("🏭 Emotion Style Profiles refactorizado registrado en factory")
    except Exception as e:
        logger.warning(f"No se pudo registrar en factory: {e}")

# ===============================================================================
# FUNCIONES DE TEST Y VALIDACIÓN
# ===============================================================================

def test_emotion_style_profiles_refactorizado():
    """Test del gestor refactorizado"""
    
    print("🧪 Testeando Emotion Style Profiles Refactorizado...")
    
    # Crear gestor
    gestor = crear_gestor_presets()
    
    # Test listado
    perfiles = gestor.listar_perfiles()
    print(f"✅ Perfiles disponibles: {len(perfiles)}")
    
    # Test obtención de perfil
    perfil = gestor.obtener_perfil("relajacion_profunda")
    print(f"✅ Perfil obtenido: {perfil['nombre'] if perfil else 'None'}")
    
    # Test validación
    es_valido = gestor.validar_perfil(perfil) if perfil else False
    print(f"✅ Validación: {es_valido}")
    
    # Test búsqueda por categoría
    relajacion_presets = gestor.buscar_por_categoria("relajacion")
    print(f"✅ Presets de relajación: {len(relajacion_presets)}")
    
    # Test estadísticas
    stats = gestor.obtener_estadisticas()
    print(f"✅ Estadísticas: {stats['total_presets']} presets, {stats['total_perfiles_estilo']} estilos")
    
    print("🎉 Test completado exitosamente")

# ===============================================================================
# INFORMACIÓN DEL MÓDULO
# ===============================================================================

__version__ = VERSION
__author__ = "Sistema Aurora V7"
__description__ = "Gestión de perfiles emocionales sin dependencias circulares"

if __name__ == "__main__":
    print("🎨 Emotion Style Profiles V7 Refactorizado")
    print(f"Versión: {__version__}")
    print("Estado: Sin dependencias circulares")
    
    # Ejecutar test
    test_emotion_style_profiles_refactorizado()
