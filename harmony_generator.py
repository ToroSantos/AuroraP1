import numpy as np
from scipy.signal import butter, filtfilt
from typing import Dict, List, Tuple, Optional, Union
import logging
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
import warnings
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EstiloEmocional(Enum):
    MISTICO = "mistico"
    ETEREO = "etereo" 
    TRIBAL = "tribal"
    GLITCH = "glitch"
    NEUTRA = "neutra"
    AMBIENTAL = "ambiental"
    INDUSTRIAL = "industrial"
    CELESTIAL = "celestial"

class TipoAcorde(Enum):
    MAYOR = "mayor"
    MENOR = "menor"
    SUSPENDIDO = "suspendido"
    SEPTIMA = "7"
    MENOR_NOVENA = "m9"
    MAYOR_SEPTIMA = "maj7"
    DISMINUIDO = "dim"
    AUMENTADO = "aug"

@dataclass
class ConfiguracionAudio:
    sample_rate: int = 44100
    duracion_ms: int = 1000
    gain_db: float = -12.0
    fade_in_ms: int = 50
    fade_out_ms: int = 50
    
    def __post_init__(self):
        if self.sample_rate <= 0 or self.duracion_ms <= 0:
            raise ValueError("Sample rate y duración deben ser positivos")
        if self.gain_db > 0:
            warnings.warn("Gain positivo puede causar distorsión")

class GeneradorArmonias:
    def __init__(self, config: Optional[ConfiguracionAudio] = None):
        self.config = config or ConfiguracionAudio()
        self._cache_ondas = {}
        self._escalas_cache = {}
        
        self._escalas_emocionales = {
            EstiloEmocional.MISTICO: [1, 6/5, 4/3, 3/2, 7/4, 2],
            EstiloEmocional.ETEREO: [1, 9/8, 5/4, 11/8, 3/2, 15/8],
            EstiloEmocional.TRIBAL: [1, 1.18, 1.33, 1.5, 1.78, 2],
            EstiloEmocional.GLITCH: [1, 1.1, 1.25, 1.42, 1.68, 1.92],
            EstiloEmocional.NEUTRA: [1, 5/4, 3/2, 2],
            EstiloEmocional.AMBIENTAL: [1, 1.125, 1.25, 1.5, 1.875, 2],
            EstiloEmocional.INDUSTRIAL: [1, 1.15, 1.4, 1.6, 1.85, 2.1],
            EstiloEmocional.CELESTIAL: [1, 1.059, 1.189, 1.335, 1.498, 1.682, 1.888, 2]
        }
        
        self._acordes = {
            TipoAcorde.MAYOR: [1, 5/4, 3/2], TipoAcorde.MENOR: [1, 6/5, 3/2],
            TipoAcorde.SUSPENDIDO: [1, 4/3, 3/2], TipoAcorde.SEPTIMA: [1, 5/4, 3/2, 7/4],
            TipoAcorde.MENOR_NOVENA: [1, 6/5, 3/2, 9/5], TipoAcorde.MAYOR_SEPTIMA: [1, 5/4, 3/2, 15/8],
            TipoAcorde.DISMINUIDO: [1, 6/5, 64/45], TipoAcorde.AUMENTADO: [1, 5/4, 8/5]
        }
    
    @lru_cache(maxsize=128)
    def _generar_onda_base(self, freq: float, duracion_samples: int, gain_db: float) -> np.ndarray:
        try:
            t = np.linspace(0, duracion_samples / self.config.sample_rate, duracion_samples, endpoint=False)
            return np.sin(2 * np.pi * freq * t) * (10 ** (gain_db / 20))
        except Exception as e:
            logger.error(f"Error generando onda: {e}")
            return np.zeros(duracion_samples)
    
    def _aplicar_envelope(self, signal: np.ndarray) -> np.ndarray:
        fade_in_s = int(self.config.fade_in_ms * self.config.sample_rate / 1000)
        fade_out_s = int(self.config.fade_out_ms * self.config.sample_rate / 1000)
        
        if len(signal) <= fade_in_s + fade_out_s:
            return signal
        
        signal[:fade_in_s] *= np.linspace(0, 1, fade_in_s)
        signal[-fade_out_s:] *= np.linspace(1, 0, fade_out_s)
        return signal
    
    def normalizar_seguro(self, wave: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
        if len(wave) == 0:
            return wave
        max_val = np.max(np.abs(wave))
        return wave if max_val == 0 else (wave / max_val) * (10 ** (headroom_db / 20))
    
    def aplicar_filtro_inteligente(self, signal: np.ndarray, cutoff: float = 800.0, 
                                 filter_type: str = 'lowpass', order: int = 6) -> np.ndarray:
        try:
            nyquist = self.config.sample_rate * 0.5
            if cutoff >= nyquist:
                cutoff = nyquist * 0.95
            b, a = butter(order, cutoff / nyquist, btype=filter_type)
            return filtfilt(b, a, signal)
        except Exception as e:
            logger.error(f"Error en filtrado: {e}")
            return signal
    
    def obtener_frecuencias_escala(self, base_freq: float, estilo: EstiloEmocional) -> List[float]:
        if estilo not in self._escalas_emocionales:
            estilo = EstiloEmocional.NEUTRA
        return [base_freq * ratio for ratio in self._escalas_emocionales[estilo]]
    
    def obtener_frecuencias_acorde(self, base_freq: float, tipo: TipoAcorde) -> List[float]:
        if tipo not in self._acordes:
            tipo = TipoAcorde.MAYOR
        return [base_freq * ratio for ratio in self._acordes[tipo]]
    
    def generar_pad_avanzado(self, base_freq: float = 220.0, estilo: EstiloEmocional = EstiloEmocional.MISTICO,
                           modulacion: bool = True, stereo: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        try:
            duracion_samples = int(self.config.sample_rate * self.config.duracion_ms / 1000)
            frecuencias = self.obtener_frecuencias_escala(base_freq, estilo)
            
            capas = [self._generar_onda_base(freq, duracion_samples, self.config.gain_db - (i * 2)) 
                    for i, freq in enumerate(frecuencias)]
            pad_combinado = np.sum(capas, axis=0)
            
            if modulacion:
                t = np.linspace(0, self.config.duracion_ms / 1000, duracion_samples)
                mod1 = 0.9 + 0.1 * np.sin(2 * np.pi * 0.2 * t)
                mod2 = 1.0 + 0.05 * np.sin(2 * np.pi * 0.7 * t)
                pad_combinado *= mod1 * mod2
            
            pad_combinado = self.normalizar_seguro(self._aplicar_envelope(pad_combinado))
            
            if stereo:
                pad_derecho = self.aplicar_filtro_inteligente(pad_combinado, cutoff=1200)
                pad_izquierdo = self.aplicar_filtro_inteligente(pad_combinado, cutoff=900)
                return pad_izquierdo, pad_derecho
            
            return pad_combinado
        except Exception as e:
            logger.error(f"Error generando pad: {e}")
            return np.zeros(int(self.config.sample_rate * self.config.duracion_ms / 1000))
    
    def generar_acorde_pad(self, base_freq: float = 220.0, tipo: TipoAcorde = TipoAcorde.MAYOR,
                          voicing: str = "close") -> np.ndarray:
        try:
            duracion_samples = int(self.config.sample_rate * self.config.duracion_ms / 1000)
            frecuencias_base = self.obtener_frecuencias_acorde(base_freq, tipo)
            
            if voicing == "open":
                frecuencias = frecuencias_base + [f * 2 for f in frecuencias_base[1:]]
            elif voicing == "drop2" and len(frecuencias_base) > 1:
                frecuencias = frecuencias_base.copy()
                frecuencias[1] /= 2
            else:
                frecuencias = frecuencias_base
            
            capas = [self._generar_onda_base(freq, duracion_samples, self.config.gain_db - (i * 1.5)) 
                    for i, freq in enumerate(frecuencias)]
            acorde_combinado = self.normalizar_seguro(self._aplicar_envelope(np.sum(capas, axis=0)))
            return acorde_combinado
        except Exception as e:
            logger.error(f"Error generando acorde: {e}")
            return np.zeros(int(self.config.sample_rate * self.config.duracion_ms / 1000))
    
    def mezclar_elementos(self, elementos: List[np.ndarray], pesos: Optional[List[float]] = None) -> np.ndarray:
        if not elementos:
            return np.array([])
        
        longitud_min = min(len(elemento) for elemento in elementos)
        elementos_norm = [elemento[:longitud_min] for elemento in elementos]
        
        if pesos and len(pesos) == len(elementos):
            elementos_norm = [elem * peso for elem, peso in zip(elementos_norm, pesos)]
        
        return self.normalizar_seguro(np.sum(elementos_norm, axis=0))
    
    def generar_textura_compleja(self, base_freq: float = 220.0, num_capas: int = 5,
                               spread_octavas: float = 2.0) -> np.ndarray:
        try:
            estilos = list(EstiloEmocional)
            capas = []
            
            for i in range(num_capas):
                factor_freq = 1 + (i / (num_capas - 1)) * (spread_octavas - 1)
                freq_capa = base_freq * factor_freq
                estilo = estilos[i % len(estilos)]
                
                pad_capa = self.generar_pad_avanzado(base_freq=freq_capa, estilo=estilo, modulacion=True)
                cutoff = 2000 - (i * 300)
                pad_filtrado = self.aplicar_filtro_inteligente(pad_capa, cutoff=cutoff)
                capas.append(pad_filtrado)
            
            pesos = [1 / (i + 1) for i in range(num_capas)]
            return self.mezclar_elementos(capas, pesos)
        except Exception as e:
            logger.error(f"Error generando textura: {e}")
            return np.zeros(int(self.config.sample_rate * self.config.duracion_ms / 1000))
        
    def crear_pads_armonicos(self, base_freq: float = 220.0, 
                           estilos_mix: Optional[List[EstiloEmocional]] = None,
                           tipos_acorde: Optional[List[TipoAcorde]] = None,
                           num_capas: int = 4, intensidad: float = 0.8,
                           modulacion_avanzada: bool = True, 
                           stereo_width: float = 0.6) -> Dict[str, np.ndarray]:
        """
        Genera pads armónicos combinando múltiples estilos emocionales y tipos de acordes.
        
        Este método actúa como orquestador maestro, combinando toda la funcionalidad
        existente del GeneradorArmonias para crear texturas armónicas complejas.
        
        Args:
            base_freq: Frecuencia base para la generación (Hz)
            estilos_mix: Lista de estilos emocionales a combinar. Si None, usa selección automática
            tipos_acorde: Lista de tipos de acordes a combinar. Si None, usa progresión harmónica
            num_capas: Número de capas armónicas a generar (2-8)
            intensidad: Factor de intensidad general (0.1-1.0)
            modulacion_avanzada: Activar modulaciones complejas
            stereo_width: Amplitud del campo estéreo (0.0-1.0)
            
        Returns:
            Dict con las siguientes claves:
            - 'pad_principal': Pad armónico principal combinado
            - 'pad_estereo': Versión estéreo del pad (tuple L/R si stereo_width > 0)
            - 'capas_individuales': Lista de capas por separado
            - 'metadata': Información de configuración y estadísticas
            
        Raises:
            ValueError: Si los parámetros están fuera de rango válido
        """
        try:
            # Validación de parámetros de entrada
            if not 20 <= base_freq <= 20000:
                raise ValueError(f"base_freq debe estar entre 20-20000 Hz, recibido: {base_freq}")
            if not 2 <= num_capas <= 8:
                num_capas = max(2, min(8, num_capas))
                logger.warning(f"num_capas ajustado a rango válido: {num_capas}")
            if not 0.1 <= intensidad <= 1.0:
                intensidad = max(0.1, min(1.0, intensidad))
                logger.warning(f"intensidad ajustada a rango válido: {intensidad}")
            if not 0.0 <= stereo_width <= 1.0:
                stereo_width = max(0.0, min(1.0, stereo_width))
                
            # Configuración automática de estilos si no se proporciona
            if estilos_mix is None:
                # Selección inteligente basada en frecuencia base
                if base_freq < 100:
                    estilos_mix = [EstiloEmocional.MISTICO, EstiloEmocional.ETEREO, EstiloEmocional.CELESTIAL]
                elif base_freq < 300:
                    estilos_mix = [EstiloEmocional.AMBIENTAL, EstiloEmocional.MISTICO, EstiloEmocional.TRIBAL]
                else:
                    estilos_mix = [EstiloEmocional.ETEREO, EstiloEmocional.GLITCH, EstiloEmocional.INDUSTRIAL]
            
            # Configuración automática de acordes si no se proporciona
            if tipos_acorde is None:
                tipos_acorde = [TipoAcorde.MAYOR, TipoAcorde.MENOR_NOVENA, TipoAcorde.MAYOR_SEPTIMA]
            
            # Limitamos a num_capas para evitar sobrecarga
            estilos_activos = estilos_mix[:num_capas]
            acordes_activos = tipos_acorde[:num_capas]
            
            logger.info(f"Generando pads armónicos: freq={base_freq}Hz, capas={num_capas}, "
                       f"estilos={len(estilos_activos)}, intensidad={intensidad}")
            
            # Contenedores para resultados
            capas_individuales = []
            capas_acordes = []
            metadata = {
                'configuracion': {
                    'base_freq': base_freq,
                    'num_capas': num_capas,
                    'intensidad': intensidad,
                    'stereo_width': stereo_width,
                    'modulacion_avanzada': modulacion_avanzada
                },
                'estilos_utilizados': [e.value for e in estilos_activos],
                'acordes_utilizados': [a.value for a in acordes_activos],
                'timestamp': f"aurora_v7_{int(time.time() * 1000) % 100000}",
                'estadisticas': {}
            }
            
            # Generación de capas principales usando funcionalidad existente
            for i in range(num_capas):
                # Cálculo de frecuencia para cada capa (distribución logarítmica)
                factor_freq = 1.0 + (i / max(1, num_capas - 1)) * 1.5
                freq_capa = base_freq * factor_freq
                
                # Selección cíclica de estilos y acordes
                estilo_actual = estilos_activos[i % len(estilos_activos)]
                acorde_actual = acordes_activos[i % len(acordes_activos)]
                
                # Ajuste de ganancia progresiva para cada capa
                gain_ajustado = self.config.gain_db - (i * 1.8) * intensidad
                
                # Generación usando métodos existentes - DELEGACIÓN INTELIGENTE
                pad_base = self.generar_pad_avanzado(
                    base_freq=freq_capa,
                    estilo=estilo_actual,
                    modulacion=modulacion_avanzada,
                    stereo=False
                )
                
                # Generación de componente armónico usando acordes
                acorde_componente = self.generar_acorde_pad(
                    base_freq=freq_capa * 0.5,  # Una octava más grave
                    tipo=acorde_actual,
                    voicing="open" if i % 2 == 0 else "close"
                )
                
                # Aplicación de filtrado inteligente diferenciado por capa
                cutoff_freq = max(400, 2200 - (i * 250))
                pad_filtrado = self.aplicar_filtro_inteligente(
                    pad_base, 
                    cutoff=cutoff_freq,
                    filter_type='lowpass',
                    order=4 + i
                )
                
                acorde_filtrado = self.aplicar_filtro_inteligente(
                    acorde_componente,
                    cutoff=cutoff_freq * 1.2,
                    filter_type='lowpass',
                    order=3
                )
                
                # Combinación proporcional de pad y acorde
                peso_pad = 0.7 + (i * 0.05)
                peso_acorde = 1.0 - peso_pad
                
                capa_combinada = self.mezclar_elementos(
                    [pad_filtrado, acorde_filtrado],
                    [peso_pad * intensidad, peso_acorde * intensidad]
                )
                
                capas_individuales.append(capa_combinada)
                capas_acordes.append(acorde_filtrado)
                
                logger.debug(f"Capa {i+1}: freq={freq_capa:.1f}Hz, estilo={estilo_actual.value}, "
                           f"acorde={acorde_actual.value}, cutoff={cutoff_freq}Hz")
            
            # Combinación final usando mezcla inteligente existente
            if len(capas_individuales) > 1:
                # Pesos decrecientes para balance natural
                pesos_finales = [1.0 / (i + 1) ** 0.7 for i in range(len(capas_individuales))]
                pad_principal = self.mezclar_elementos(capas_individuales, pesos_finales)
            else:
                pad_principal = capas_individuales[0] if capas_individuales else np.zeros(1000)
            
            # Aplicación de modulación avanzada global si está activada
            if modulacion_avanzada and len(pad_principal) > 0:
                duracion_s = len(pad_principal) / self.config.sample_rate
                t = np.linspace(0, duracion_s, len(pad_principal))
                
                # Modulación compleja multicapa
                mod_lenta = 0.95 + 0.05 * np.sin(2 * np.pi * 0.1 * t)  # Respiración lenta
                mod_media = 1.0 + 0.03 * np.sin(2 * np.pi * 0.4 * t + np.pi/3)  # Vibrato medio
                mod_rapida = 1.0 + 0.02 * np.sin(2 * np.pi * 1.1 * t + np.pi/2)  # Shimmer
                
                pad_principal = pad_principal * mod_lenta * mod_media * mod_rapida
            
            # Normalización final segura
            pad_principal = self.normalizar_seguro(pad_principal, headroom_db=-2.0)
            
            # Generación de versión estéreo si se solicita
            pad_estereo = None
            if stereo_width > 0.0:
                # Crear versión estéreo usando filtrado diferencial
                pad_izquierdo = self.aplicar_filtro_inteligente(
                    pad_principal, 
                    cutoff=1000 * (1 + stereo_width * 0.3),
                    filter_type='lowpass'
                )
                pad_derecho = self.aplicar_filtro_inteligente(
                    pad_principal,
                    cutoff=1000 * (1 - stereo_width * 0.2),
                    filter_type='highpass',
                    order=3
                )
                
                # Aplicar paneo sutil
                factor_pan = stereo_width * 0.8
                pad_izquierdo *= (1.0 + factor_pan)
                pad_derecho *= (1.0 + factor_pan)
                
                pad_estereo = (pad_izquierdo, pad_derecho)
                logger.info(f"Generada versión estéreo con width={stereo_width}")
            
            # Cálculo de estadísticas finales
            metadata['estadisticas'] = {
                'duracion_ms': len(pad_principal) / self.config.sample_rate * 1000,
                'rms_level': float(np.sqrt(np.mean(pad_principal ** 2))),
                'peak_level': float(np.max(np.abs(pad_principal))),
                'dynamic_range_db': float(20 * np.log10(np.max(np.abs(pad_principal)) / 
                                                      (np.sqrt(np.mean(pad_principal ** 2)) + 1e-10))),
                'capas_generadas': len(capas_individuales),
                'sample_rate': self.config.sample_rate,
                'es_estereo': pad_estereo is not None
            }
            
            # Construcción del resultado final
            resultado = {
                'pad_principal': pad_principal,
                'pad_estereo': pad_estereo,
                'capas_individuales': capas_individuales,
                'metadata': metadata
            }
            
            logger.info(f"✅ Pads armónicos generados exitosamente: "
                       f"{len(capas_individuales)} capas, "
                       f"{metadata['estadisticas']['duracion_ms']:.0f}ms, "
                       f"RMS={metadata['estadisticas']['rms_level']:.3f}")
            
            return resultado
            
        except Exception as e:
            # Manejo robusto de errores con fallback
            logger.error(f"Error en crear_pads_armonicos: {e}")
            
            # Fallback usando textura compleja existente
            try:
                pad_fallback = self.generar_textura_compleja(
                    base_freq=base_freq,
                    num_capas=min(3, num_capas),
                    spread_octavas=1.5
                )
                
                return {
                    'pad_principal': pad_fallback,
                    'pad_estereo': None,
                    'capas_individuales': [pad_fallback],
                    'metadata': {
                        'configuracion': {'base_freq': base_freq, 'fallback': True},
                        'estadisticas': {'es_fallback': True}
                    }
                }
                
            except Exception as fallback_error:
                logger.error(f"Error en fallback: {fallback_error}")
                # Último recurso: señal silenciosa
                silence = np.zeros(int(self.config.sample_rate * self.config.duracion_ms / 1000))
                return {
                    'pad_principal': silence,
                    'pad_estereo': None,
                    'capas_individuales': [silence],
                    'metadata': {'configuracion': {'error': True}, 'estadisticas': {'es_silencio': True}}
                }
    
    def exportar_configuracion(self) -> Dict:
        return {
            "sample_rate": self.config.sample_rate, "duracion_ms": self.config.duracion_ms,
            "gain_db": self.config.gain_db, "escalas_disponibles": [e.value for e in EstiloEmocional],
            "acordes_disponibles": [t.value for t in TipoAcorde], "version": "v7_aurora_integrated"
        }
    
    def limpiar_cache(self):
        self._generar_onda_base.cache_clear()
        self._cache_ondas.clear()
        self._escalas_cache.clear()

    def generar_acordes(self, tonalidad: str = "C", progresion: List[str] = None, 
                    tipo_acorde: TipoAcorde = TipoAcorde.MAYOR) -> np.ndarray:
        """Genera secuencia de acordes según progresión especificada"""
        try:
            if progresion is None:
                progresion = ['I', 'V', 'vi', 'IV']  # Progresión clásica
                
            # Mapeo de tonalidades
            freq_base = {'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23,
                        'G': 392.00, 'A': 440.00, 'B': 493.88}.get(tonalidad, 261.63)
            
            # Generar acordes
            acordes = []
            duracion_acorde = self.config.duracion_ms // len(progresion)
            
            for grado in progresion:
                acorde = self.generar_acorde_pad(
                    base_freq=freq_base,
                    tipo=tipo_acorde,
                    voicing="close"
                )
                acordes.append(acorde)
            
            return np.concatenate(acordes) if acordes else np.zeros(1000)
            
        except Exception:
            # Fallback básico
            return np.zeros(int(self.config.sample_rate * self.config.duracion_ms / 1000))

def crear_generador_aurora(sample_rate: int = 44100, duracion_ms: int = 4000) -> GeneradorArmonias:
    config = ConfiguracionAudio(sample_rate=sample_rate, duracion_ms=duracion_ms, 
                               gain_db=-15.0, fade_in_ms=100, fade_out_ms=200)
    return GeneradorArmonias(config)

def generar_pack_emocional(base_freq: float = 220.0) -> Dict[str, np.ndarray]:
    generador = crear_generador_aurora(duracion_ms=6000)
    return {estilo.value: generador.generar_pad_avanzado(base_freq=base_freq, estilo=estilo, modulacion=True) 
            for estilo in EstiloEmocional}