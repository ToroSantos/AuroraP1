#!/usr/bin/env python3
"""
🎼 HARMONIC ESSENCE V34 - REFACTORIZADO SIN DEPENDENCIAS CIRCULARES
=================================================================

Motor de texturas emocionales y estéticas para el Sistema Aurora V7.
Versión refactorizada que utiliza interfaces y lazy loading para eliminar
dependencias circulares con emotion_style_profiles.

Versión: V34.1 - Refactorizado
Propósito: Generación de texturas sin dependencias circulares
Compatibilidad: Aurora Director V7
"""

import numpy as np
import logging
import math
import random
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import lru_cache
import warnings

# Importar interfaces Aurora (sin ciclos)
try:
    from aurora_interfaces import (
        MotorAuroraInterface, GestorPerfilesInterface, ComponenteAuroraBase,
        TipoComponenteAurora, EstadoComponente, NivelCalidad
    )
    from aurora_factory import aurora_factory, lazy_importer
    INTERFACES_DISPONIBLES = True
except ImportError as e:
    logging.warning(f"Aurora interfaces no disponibles: {e}")
    INTERFACES_DISPONIBLES = False
    
    # Clases dummy para compatibilidad
    class MotorAuroraInterface: pass
    class GestorPerfilesInterface: pass
    class ComponenteAuroraBase: pass

# Configuración de logging
logger = logging.getLogger("Aurora.HarmonicEssence.V34.Refactored")

# ===============================================================================
# CONFIGURACIONES Y ENUMS
# ===============================================================================

class TipoTextura(Enum):
    """Tipos de texturas disponibles"""
    AMBIENTAL = "ambiental"
    NEURONAL = "neuronal"
    ORGANICA = "organica"
    CRISTALINA = "cristalina"
    CAOSTICA = "caostica"
    ESPACIAL = "espacial"
    TEMPORAL = "temporal"
    FRACTAL = "fractal"

class ModoGeneracion(Enum):
    """Modos de generación de texturas"""
    ESTETICO = "estetico"
    NEUROACUSTICO = "neuroacustico"
    TERAPEUTICO = "terapeutico"
    EXPERIMENTAL = "experimental"
    HIBRIDO = "hibrido"

class NivelComplejidad(Enum):
    """Niveles de complejidad para las texturas"""
    SIMPLE = "simple"
    MODERADO = "moderado"
    COMPLEJO = "complejo"
    AVANZADO = "avanzado"
    EXTREMO = "extremo"

@dataclass
class ConfiguracionTextura:
    """Configuración para generación de texturas"""
    tipo: TipoTextura = TipoTextura.AMBIENTAL
    modo: ModoGeneracion = ModoGeneracion.ESTETICO
    complejidad: NivelComplejidad = NivelComplejidad.MODERADO
    intensidad: float = 0.5
    duracion_sec: float = 300.0
    frecuencia_base: float = 110.0
    modulacion_am: float = 0.2
    modulacion_fm: float = 0.1
    reverb_factor: float = 0.3
    spatial_width: float = 0.7
    efectos_extras: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResultadoTextura:
    """Resultado de generación de texturas"""
    audio_data: np.ndarray
    metadatos: Dict[str, Any]
    configuracion_usada: ConfiguracionTextura
    tiempo_generacion: float
    calidad_estimada: float
    efectos_aplicados: List[str]

# ===============================================================================
# MOTOR HARMONIC ESSENCE REFACTORIZADO
# ===============================================================================

class HarmonicEssenceV34Refactored(ComponenteAuroraBase):
    """
    Motor de texturas emocionales refactorizado sin dependencias circulares
    
    Utiliza interfaces para comunicarse con gestores de perfiles sin
    importar directamente emotion_style_profiles.
    """
    
    def __init__(self, gestor_perfiles: Optional[GestorPerfilesInterface] = None):
        super().__init__("HarmonicEssence", "V34.1-Refactored")
        
        # Configuración inicial
        self.sample_rate = 44100
        self.gestor_perfiles = gestor_perfiles
        
        # Capacidades del motor
        self.capacidades = {
            "tipos_textura": [t.value for t in TipoTextura],
            "modos_generacion": [m.value for m in ModoGeneracion],
            "efectos_disponibles": [
                "reverb", "delay", "chorus", "phaser", "flanger",
                "spatial_processing", "neural_envelope", "harmonic_distortion"
            ],
            "formatos_salida": ["stereo", "mono", "binaural"],
            "frecuencia_min": 20.0,
            "frecuencia_max": 20000.0,
            "duracion_max": 3600.0
        }
        
        # Cache para optimización
        self._cache_texturas = {}
        self._cache_perfiles = {}
        
        # Estadísticas
        self.stats = {
            "texturas_generadas": 0,
            "tiempo_total_generacion": 0.0,
            "cache_hits": 0,
            "errores": 0
        }
        
        self.cambiar_estado(EstadoComponente.CONFIGURADO)
        logger.info("🎼 HarmonicEssence V34 refactorizado inicializado")
    
    # ===================================================================
    # IMPLEMENTACIÓN DE MotorAuroraInterface
    # ===================================================================
    
    def generar_audio(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementación de la interface MotorAuroraInterface
        
        Args:
            config: Configuración de generación
            
        Returns:
            Dict con audio_data, metadatos, etc.
        """
        start_time = time.time()
        
        try:
            # Convertir config a ConfiguracionTextura
            config_textura = self._convertir_configuracion(config)
            
            # Obtener perfil emocional si está disponible
            perfil_emocional = self._obtener_perfil_emocional(config)
            
            # Generar textura base
            audio_data = self._generar_textura_base(config_textura)
            
            # Aplicar perfil emocional si existe
            if perfil_emocional:
                audio_data = self._aplicar_perfil_emocional(audio_data, perfil_emocional)
            
            # Aplicar efectos
            audio_data = self._aplicar_efectos(audio_data, config_textura)
            
            # Normalizar
            audio_data = self._normalizar_audio(audio_data)
            
            # Preparar resultado
            tiempo_generacion = time.time() - start_time
            resultado = {
                "audio_data": audio_data,
                "sample_rate": self.sample_rate,
                "duracion_sec": len(audio_data) / self.sample_rate,
                "metadatos": {
                    "motor": self.nombre,
                    "version": self.version,
                    "tipo_textura": config_textura.tipo.value,
                    "modo_generacion": config_textura.modo.value,
                    "tiempo_generacion": tiempo_generacion,
                    "perfil_emocional_usado": perfil_emocional is not None
                }
            }
            
            # Actualizar estadísticas
            self.stats["texturas_generadas"] += 1
            self.stats["tiempo_total_generacion"] += tiempo_generacion
            
            logger.debug(f"✅ Textura generada en {tiempo_generacion:.2f}s")
            return resultado
            
        except Exception as e:
            self.stats["errores"] += 1
            logger.error(f"❌ Error generando audio: {e}")
            return self._generar_fallback(config)
    
    def validar_configuracion(self, config: Dict[str, Any]) -> bool:
        """
        Valida si la configuración es correcta para este motor
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        try:
            # Validaciones básicas
            if not isinstance(config, dict):
                return False
            
            # Validar duración
            duracion = config.get('duracion_sec', 300.0)
            if not 1.0 <= duracion <= self.capacidades["duracion_max"]:
                return False
            
            # Validar frecuencia base
            freq_base = config.get('frecuencia_base', 110.0)
            if not self.capacidades["frecuencia_min"] <= freq_base <= self.capacidades["frecuencia_max"]:
                return False
            
            # Validar tipo de textura
            tipo_textura = config.get('tipo_textura', 'ambiental')
            if tipo_textura not in self.capacidades["tipos_textura"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando configuración: {e}")
            return False
    
    def obtener_capacidades(self) -> Dict[str, Any]:
        """
        Retorna las capacidades del motor
        
        Returns:
            Dict con capacidades, formatos soportados, etc.
        """
        return {
            **self.capacidades,
            "estado": self.estado.value,
            "estadisticas": self.stats.copy(),
            "version": self.version,
            "compatible_con": ["Aurora Director V7", "Emotion Style Profiles"],
            "interfaces_implementadas": ["MotorAuroraInterface"]
        }
    
    def get_info(self) -> Dict[str, str]:
        """
        Información básica del motor
        
        Returns:
            Dict con nombre, versión, descripción, etc.
        """
        return {
            "nombre": self.nombre,
            "version": self.version,
            "descripcion": "Motor de texturas emocionales y estéticas sin dependencias circulares",
            "tipo": "motor_textura",
            "categoria": "generacion_audio",
            "estado": self.estado.value,
            "autor": "Sistema Aurora V7",
            "fecha_refactorizacion": "2025-06-10"
        }
    
    # ===================================================================
    # MÉTODOS DE INTEGRACIÓN CON PERFILES EMOCIONALES
    # ===================================================================
    
    def set_gestor_perfiles(self, gestor: GestorPerfilesInterface):
        """
        Inyecta el gestor de perfiles después de la inicialización
        
        Args:
            gestor: Gestor de perfiles que implementa la interface
        """
        self.gestor_perfiles = gestor
        self.log("Gestor de perfiles configurado")
    
    def _obtener_perfil_emocional(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Obtiene perfil emocional usando lazy loading o gestor inyectado
        
        Args:
            config: Configuración con posible referencia a perfil
            
        Returns:
            Dict con perfil emocional o None si no disponible
        """
        # Obtener nombre del perfil/estilo
        estilo = config.get('estilo', config.get('preset_emocional'))
        if not estilo:
            return None
        
        # Usar gestor inyectado si está disponible
        if self.gestor_perfiles:
            try:
                perfil = self.gestor_perfiles.obtener_perfil(estilo)
                if perfil:
                    self.log(f"Perfil '{estilo}' obtenido del gestor inyectado")
                    return perfil
            except Exception as e:
                self.log(f"Error obteniendo perfil del gestor: {e}", "WARNING")
        
        # Lazy loading como fallback
        return self._obtener_perfil_lazy_loading(estilo)
    
    def _obtener_perfil_lazy_loading(self, estilo: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene perfil usando lazy loading sin crear dependencia circular
        
        Args:
            estilo: Nombre del estilo/perfil a obtener
            
        Returns:
            Dict con perfil o None si no disponible
        """
        # Usar cache si está disponible
        cache_key = f"perfil_{estilo}"
        if cache_key in self._cache_perfiles:
            self.stats["cache_hits"] += 1
            return self._cache_perfiles[cache_key]
        
        try:
            # Usar el factory para obtener un gestor de perfiles
            gestor = aurora_factory.crear_gestor("emotion_style_profiles")
            if gestor:
                perfil = gestor.obtener_perfil(estilo)
                if perfil:
                    # Cachear para uso futuro
                    self._cache_perfiles[cache_key] = perfil
                    self.log(f"Perfil '{estilo}' obtenido vía factory")
                    return perfil
            
            # Fallback: usar lazy importer directamente
            emotion_module = lazy_importer.import_module('emotion_style_profiles')
            if emotion_module:
                # Intentar diferentes formas de obtener el perfil
                if hasattr(emotion_module, 'obtener_perfil_estilo'):
                    perfil = emotion_module.obtener_perfil_estilo(estilo)
                elif hasattr(emotion_module, 'PERFILES_ESTILO_AURORA'):
                    perfil = emotion_module.PERFILES_ESTILO_AURORA.get(estilo.lower())
                else:
                    # Crear gestor temporal
                    gestor_class = lazy_importer.get_class('emotion_style_profiles', 'GestorPresetsEmocionales')
                    if gestor_class:
                        gestor_temp = gestor_class()
                        perfil = gestor_temp.obtener_perfil(estilo) if hasattr(gestor_temp, 'obtener_perfil') else None
                    else:
                        perfil = None
                
                if perfil:
                    self._cache_perfiles[cache_key] = perfil
                    self.log(f"Perfil '{estilo}' obtenido vía lazy loading")
                    return perfil
            
        except Exception as e:
            self.log(f"Error en lazy loading del perfil '{estilo}': {e}", "WARNING")
        
        return None
    
    # ===================================================================
    # MÉTODOS DE GENERACIÓN DE TEXTURAS
    # ===================================================================
    
    def _convertir_configuracion(self, config: Dict[str, Any]) -> ConfiguracionTextura:
        """Convierte dict de configuración a ConfiguracionTextura"""
        
        # Mapear tipo de textura
        tipo_str = config.get('tipo_textura', 'ambiental')
        try:
            tipo = TipoTextura(tipo_str)
        except ValueError:
            tipo = TipoTextura.AMBIENTAL
        
        # Mapear modo de generación
        modo_str = config.get('modo_generacion', 'estetico')
        try:
            modo = ModoGeneracion(modo_str)
        except ValueError:
            modo = ModoGeneracion.ESTETICO
        
        # Mapear nivel de complejidad
        complejidad_str = config.get('complejidad', 'moderado')
        try:
            complejidad = NivelComplejidad(complejidad_str)
        except ValueError:
            complejidad = NivelComplejidad.MODERADO
        
        return ConfiguracionTextura(
            tipo=tipo,
            modo=modo,
            complejidad=complejidad,
            intensidad=config.get('intensidad', 0.5),
            duracion_sec=config.get('duracion_sec', 300.0),
            frecuencia_base=config.get('frecuencia_base', 110.0),
            modulacion_am=config.get('modulacion_am', 0.2),
            modulacion_fm=config.get('modulacion_fm', 0.1),
            reverb_factor=config.get('reverb_factor', 0.3),
            spatial_width=config.get('spatial_width', 0.7),
            efectos_extras=config.get('efectos_extras', {})
        )
    
    def _generar_textura_base(self, config: ConfiguracionTextura) -> np.ndarray:
        """
        Genera la textura base según la configuración
        
        Args:
            config: Configuración de la textura
            
        Returns:
            Array de audio estéreo
        """
        num_samples = int(config.duracion_sec * self.sample_rate)
        t = np.linspace(0, config.duracion_sec, num_samples)
        
        # Generar según tipo de textura
        if config.tipo == TipoTextura.AMBIENTAL:
            audio = self._generar_textura_ambiental(t, config)
        elif config.tipo == TipoTextura.NEURONAL:
            audio = self._generar_textura_neuronal(t, config)
        elif config.tipo == TipoTextura.ORGANICA:
            audio = self._generar_textura_organica(t, config)
        elif config.tipo == TipoTextura.CRISTALINA:
            audio = self._generar_textura_cristalina(t, config)
        elif config.tipo == TipoTextura.ESPACIAL:
            audio = self._generar_textura_espacial(t, config)
        else:
            # Fallback a ambiental
            audio = self._generar_textura_ambiental(t, config)
        
        # Asegurar formato estéreo
        if len(audio.shape) == 1:
            audio = np.column_stack([audio, audio])
        
        return audio
    
    def _generar_textura_ambiental(self, t: np.ndarray, config: ConfiguracionTextura) -> np.ndarray:
        """Genera textura ambiental suave"""
        
        # Oscilador principal
        freq_base = config.frecuencia_base
        carrier = np.sin(2 * np.pi * freq_base * t)
        
        # Modulación AM lenta
        mod_am = 1 + config.modulacion_am * np.sin(2 * np.pi * 0.1 * t)
        
        # Ruido rosa filtrado
        noise = np.random.normal(0, 0.1, len(t))
        # Filtro pasa-bajos simple
        noise_filtered = np.convolve(noise, np.ones(100)/100, mode='same')
        
        # Combinar
        audio = (carrier * mod_am + noise_filtered) * config.intensidad
        
        # Envelope suave
        envelope = np.exp(-0.5 * (t - config.duracion_sec/2)**2 / (config.duracion_sec/4)**2)
        audio *= envelope
        
        return audio
    
    def _generar_textura_neuronal(self, t: np.ndarray, config: ConfiguracionTextura) -> np.ndarray:
        """Genera textura con características neuronales"""
        
        freq_base = config.frecuencia_base
        
        # Múltiples osciladores con frecuencias relacionadas
        osc1 = np.sin(2 * np.pi * freq_base * t)
        osc2 = np.sin(2 * np.pi * freq_base * 1.618 * t)  # Proporción áurea
        osc3 = np.sin(2 * np.pi * freq_base * 2.0 * t)
        
        # Modulación FM neuronal
        mod_freq = 8.0  # Frecuencia alfa
        fm_mod = config.modulacion_fm * np.sin(2 * np.pi * mod_freq * t)
        
        # Aplicar FM
        carrier_fm = np.sin(2 * np.pi * freq_base * t + fm_mod)
        
        # Combinar osciladores con pesos neuronales
        audio = (0.5 * carrier_fm + 0.3 * osc2 + 0.2 * osc3) * config.intensidad
        
        # Envelope neuronal (pulsos)
        pulse_freq = 10.0  # 10 Hz
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * pulse_freq * t)
        audio *= envelope
        
        return audio
    
    def _generar_textura_organica(self, t: np.ndarray, config: ConfiguracionTextura) -> np.ndarray:
        """Genera textura orgánica con evolución natural"""
        
        freq_base = config.frecuencia_base
        
        # Evolución de frecuencia orgánica
        freq_evolution = freq_base * (1 + 0.1 * np.sin(2 * np.pi * 0.01 * t))
        
        # Oscilador principal con evolución
        phase = np.cumsum(2 * np.pi * freq_evolution / self.sample_rate)
        carrier = np.sin(phase)
        
        # Armónicos orgánicos
        harmonics = 0
        for h in range(2, 6):
            amplitude = 1.0 / (h ** 1.5)  # Decaimiento natural
            harmonics += amplitude * np.sin(h * phase)
        
        # Combinar
        audio = (carrier + 0.3 * harmonics) * config.intensidad
        
        # Envelope orgánico (respiración)
        breath_freq = 0.2  # 0.2 Hz = 12 respiraciones por minuto
        envelope = 0.7 + 0.3 * np.sin(2 * np.pi * breath_freq * t)
        audio *= envelope
        
        return audio
    
    def _generar_textura_cristalina(self, t: np.ndarray, config: ConfiguracionTextura) -> np.ndarray:
        """Genera textura cristalina brillante"""
        
        freq_base = config.frecuencia_base
        
        # Múltiples tonos cristalinos
        cristal = 0
        freqs_cristal = [freq_base, freq_base * 2, freq_base * 3, freq_base * 5]
        
        for i, freq in enumerate(freqs_cristal):
            amplitude = 1.0 / (i + 1)
            phase_offset = np.random.random() * 2 * np.pi
            tone = amplitude * np.sin(2 * np.pi * freq * t + phase_offset)
            cristal += tone
        
        # Modulación ring
        ring_freq = freq_base * 4
        ring_mod = np.sin(2 * np.pi * ring_freq * t)
        cristal *= (0.7 + 0.3 * ring_mod)
        
        # Reverb simulado
        delay_samples = int(0.1 * self.sample_rate)  # 100ms delay
        if len(cristal) > delay_samples:
            delayed = np.concatenate([np.zeros(delay_samples), cristal[:-delay_samples]])
            cristal += 0.3 * delayed
        
        audio = cristal * config.intensidad
        
        return audio
    
    def _generar_textura_espacial(self, t: np.ndarray, config: ConfiguracionTextura) -> np.ndarray:
        """Genera textura espacial con movimiento 3D"""
        
        freq_base = config.frecuencia_base
        
        # Generador base
        carrier = np.sin(2 * np.pi * freq_base * t)
        
        # Paneo espacial
        pan_speed = 0.1  # Hz
        pan_lfo = np.sin(2 * np.pi * pan_speed * t)
        
        # Canal izquierdo y derecho con diferente fase
        left = carrier * (0.5 + 0.5 * pan_lfo)
        right = carrier * (0.5 - 0.5 * pan_lfo)
        
        # Reverb espacial (early reflections)
        delay1 = int(0.02 * self.sample_rate)  # 20ms
        delay2 = int(0.05 * self.sample_rate)  # 50ms
        
        if len(left) > delay2:
            left += 0.2 * np.concatenate([np.zeros(delay1), left[:-delay1]])
            right += 0.2 * np.concatenate([np.zeros(delay2), right[:-delay2]])
        
        # Combinar canales
        audio = np.column_stack([left, right]) * config.intensidad
        
        return audio
    
    def _aplicar_perfil_emocional(self, audio: np.ndarray, perfil: Dict[str, Any]) -> np.ndarray:
        """
        Aplica transformaciones basadas en el perfil emocional
        
        Args:
            audio: Audio base
            perfil: Perfil emocional obtenido
            
        Returns:
            Audio modificado según el perfil
        """
        try:
            # Extraer parámetros del perfil
            intensidad_mod = perfil.get('intensidad', 1.0)
            reverb_mod = perfil.get('reverb_factor', 1.0)
            freq_mod = perfil.get('freq_modifier', 1.0)
            
            # Modificar intensidad
            audio *= intensidad_mod
            
            # Aplicar filtro emocional si está definido
            if 'filtro_emocional' in perfil:
                audio = self._aplicar_filtro_emocional(audio, perfil['filtro_emocional'])
            
            # Reverb emocional
            if reverb_mod > 1.0:
                audio = self._aplicar_reverb_simple(audio, reverb_mod)
            
            self.log(f"Perfil emocional aplicado: intensidad={intensidad_mod}, reverb={reverb_mod}")
            
        except Exception as e:
            self.log(f"Error aplicando perfil emocional: {e}", "WARNING")
        
        return audio
    
    def _aplicar_efectos(self, audio: np.ndarray, config: ConfiguracionTextura) -> np.ndarray:
        """Aplica efectos según la configuración"""
        
        # Reverb
        if config.reverb_factor > 0:
            audio = self._aplicar_reverb_simple(audio, config.reverb_factor)
        
        # Spatial processing
        if config.spatial_width > 0:
            audio = self._aplicar_spatial_processing(audio, config.spatial_width)
        
        # Efectos extras
        for efecto, parametros in config.efectos_extras.items():
            if efecto == "chorus":
                audio = self._aplicar_chorus(audio, parametros)
            elif efecto == "delay":
                audio = self._aplicar_delay(audio, parametros)
        
        return audio
    
    def _aplicar_reverb_simple(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Aplica reverb simple con múltiples delays"""
        
        if len(audio.shape) == 1:
            # Mono
            delays = [0.03, 0.07, 0.12, 0.18]  # En segundos
            levels = [0.3, 0.2, 0.15, 0.1]
            
            reverb_audio = audio.copy()
            for delay_time, level in zip(delays, levels):
                delay_samples = int(delay_time * self.sample_rate)
                if len(audio) > delay_samples:
                    delayed = np.concatenate([np.zeros(delay_samples), audio[:-delay_samples]])
                    reverb_audio += level * factor * delayed
        else:
            # Estéreo
            reverb_audio = audio.copy()
            for channel in range(audio.shape[1]):
                channel_audio = audio[:, channel]
                delays = [0.03, 0.07, 0.12, 0.18]
                levels = [0.3, 0.2, 0.15, 0.1]
                
                for delay_time, level in zip(delays, levels):
                    delay_samples = int(delay_time * self.sample_rate)
                    if len(channel_audio) > delay_samples:
                        delayed = np.concatenate([np.zeros(delay_samples), channel_audio[:-delay_samples]])
                        reverb_audio[:, channel] += level * factor * delayed
        
        return reverb_audio
    
    def _aplicar_spatial_processing(self, audio: np.ndarray, width: float) -> np.ndarray:
        """Aplica procesamiento espacial para ampliar la imagen estéreo"""
        
        if len(audio.shape) == 1:
            # Convertir a estéreo si es mono
            audio = np.column_stack([audio, audio])
        
        if audio.shape[1] >= 2:
            # Procesamiento M/S (Mid/Side)
            mid = (audio[:, 0] + audio[:, 1]) / 2
            side = (audio[:, 0] - audio[:, 1]) / 2
            
            # Ampliar el componente side según width
            side *= width
            
            # Reconvertir a L/R
            left = mid + side
            right = mid - side
            
            audio = np.column_stack([left, right])
        
        return audio
    
    def _aplicar_filtro_emocional(self, audio: np.ndarray, tipo_filtro: str) -> np.ndarray:
        """Aplica filtros basados en la emoción"""
        
        if tipo_filtro == "brillante":
            # Realce de agudos simple
            if len(audio.shape) == 1:
                diff = np.diff(audio, prepend=audio[0])
                audio += 0.1 * diff
            else:
                for ch in range(audio.shape[1]):
                    diff = np.diff(audio[:, ch], prepend=audio[0, ch])
                    audio[:, ch] += 0.1 * diff
        
        elif tipo_filtro == "calido":
            # Atenuación de agudos
            if len(audio.shape) == 1:
                # Filtro pasa-bajos simple
                audio = np.convolve(audio, np.ones(5)/5, mode='same')
            else:
                for ch in range(audio.shape[1]):
                    audio[:, ch] = np.convolve(audio[:, ch], np.ones(5)/5, mode='same')
        
        return audio
    
    def _aplicar_chorus(self, audio: np.ndarray, parametros: Dict[str, Any]) -> np.ndarray:
        """Aplica efecto chorus"""
        
        delay_time = parametros.get('delay_ms', 20) / 1000.0
        depth = parametros.get('depth', 0.5)
        rate = parametros.get('rate_hz', 0.5)
        
        if len(audio.shape) == 1:
            audio = np.column_stack([audio, audio])
        
        # LFO para modulación del delay
        t = np.arange(len(audio)) / self.sample_rate
        lfo = depth * np.sin(2 * np.pi * rate * t)
        
        # Aplicar chorus a cada canal
        for ch in range(audio.shape[1]):
            delay_samples = int((delay_time + lfo[::len(lfo)//len(audio)]) * self.sample_rate)
            # Simplificación: usar delay fijo promedio
            avg_delay = int(delay_time * self.sample_rate)
            if len(audio) > avg_delay:
                delayed = np.concatenate([np.zeros(avg_delay), audio[:-avg_delay, ch]])
                audio[:, ch] += 0.5 * delayed
        
        return audio
    
    def _aplicar_delay(self, audio: np.ndarray, parametros: Dict[str, Any]) -> np.ndarray:
        """Aplica efecto delay"""
        
        delay_time = parametros.get('delay_ms', 300) / 1000.0
        feedback = parametros.get('feedback', 0.3)
        mix_level = parametros.get('mix', 0.3)
        
        delay_samples = int(delay_time * self.sample_rate)
        
        if len(audio.shape) == 1:
            if len(audio) > delay_samples:
                delayed = np.concatenate([np.zeros(delay_samples), audio[:-delay_samples]])
                audio += mix_level * delayed
        else:
            for ch in range(audio.shape[1]):
                if len(audio) > delay_samples:
                    delayed = np.concatenate([np.zeros(delay_samples), audio[:-delay_samples, ch]])
                    audio[:, ch] += mix_level * delayed
        
        return audio
    
    def _normalizar_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normaliza el audio para evitar clipping"""
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Normalizar con headroom
            target_level = 0.85
            audio = audio * (target_level / max_val)
        
        return audio
    
    def _generar_fallback(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Genera audio fallback en caso de error"""
        
        duracion = config.get('duracion_sec', 60.0)
        freq = config.get('frecuencia_base', 220.0)
        
        num_samples = int(duracion * self.sample_rate)
        t = np.linspace(0, duracion, num_samples)
        
        # Tono simple con envelope
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-t / (duracion * 0.3))  # Decay exponencial
        audio *= envelope
        
        # Convertir a estéreo
        audio_stereo = np.column_stack([audio, audio])
        
        return {
            "audio_data": audio_stereo,
            "sample_rate": self.sample_rate,
            "duracion_sec": duracion,
            "metadatos": {
                "motor": self.nombre,
                "version": self.version,
                "modo": "fallback",
                "mensaje": "Audio generado en modo fallback"
            }
        }

# ===============================================================================
# FUNCIONES DE COMPATIBILIDAD Y FACTORY
# ===============================================================================

def crear_harmonic_essence_refactorizado(gestor_perfiles: Optional[GestorPerfilesInterface] = None) -> HarmonicEssenceV34Refactored:
    """
    Factory function para crear instancia refactorizada
    
    Args:
        gestor_perfiles: Gestor de perfiles opcional para inyección
        
    Returns:
        Instancia de HarmonicEssence refactorizada
    """
    return HarmonicEssenceV34Refactored(gestor_perfiles)

def test_harmonic_essence_refactorizado():
    """Test básico del motor refactorizado"""
    
    print("🧪 Testeando HarmonicEssence V34 Refactorizado...")
    
    # Crear instancia
    motor = crear_harmonic_essence_refactorizado()
    
    # Test capacidades
    capacidades = motor.obtener_capacidades()
    print(f"✅ Capacidades: {len(capacidades)} propiedades")
    
    # Test info
    info = motor.get_info()
    print(f"✅ Info: {info['nombre']} v{info['version']}")
    
    # Test configuración
    config = {
        "tipo_textura": "ambiental",
        "duracion_sec": 10.0,
        "intensidad": 0.5
    }
    
    es_valida = motor.validar_configuracion(config)
    print(f"✅ Validación configuración: {es_valida}")
    
    # Test generación
    resultado = motor.generar_audio(config)
    print(f"✅ Audio generado: {resultado['duracion_sec']:.1f}s")
    
    # Mostrar estadísticas
    stats = motor.obtener_capacidades()["estadisticas"]
    print(f"✅ Estadísticas: {stats}")
    
    print("🎉 Test completado exitosamente")

# Auto-registro en el factory si está disponible
if INTERFACES_DISPONIBLES:
    try:
        aurora_factory.registrar_motor(
            "harmonicEssence_v34_refactored",
            "harmonicEssence_v34",  # Este archivo
            "HarmonicEssenceV34Refactored"
        )
        logger.info("🏭 HarmonicEssence V34 refactorizado registrado en factory")
    except Exception as e:
        logger.warning(f"No se pudo registrar en factory: {e}")

# ===============================================================================
# COMPATIBILIDAD CON VERSIÓN ANTERIOR
# ===============================================================================

# Alias para compatibilidad
HarmonicEssenceV34AuroraConnected = HarmonicEssenceV34Refactored

# ===============================================================================
# INFORMACIÓN DEL MÓDULO
# ===============================================================================

__version__ = "V34.1-Refactored"
__author__ = "Sistema Aurora V7"
__description__ = "Motor de texturas emocionales sin dependencias circulares"

if __name__ == "__main__":
    print("🎼 HarmonicEssence V34 Refactorizado")
    print(f"Versión: {__version__}")
    print("Estado: Sin dependencias circulares")
    
    # Ejecutar test si se ejecuta directamente
    test_harmonic_essence_refactorizado()
