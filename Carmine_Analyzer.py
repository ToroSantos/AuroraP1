"""Carmine Analyzer V2.1 – Evaluador Profesional Aurora-Compatible - SINTAXIS COMPLETAMENTE CORREGIDA"""
import numpy as np
import logging
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq
import time

warnings.filterwarnings('ignore')
SAMPLE_RATE = 44100
logger = logging.getLogger("CarmineAurora")
logger.setLevel(logging.INFO)

class QualityLevel(Enum):
    CRITICAL = "🔴 CRÍTICO"
    WARNING = "🟡 OBSERVACIÓN"
    OPTIMAL = "🟢 ÓPTIMO"
    THERAPEUTIC = "💙 TERAPÉUTICO"

class NeuroAnalysisType(Enum):
    BINAURAL_BEATS = "binaural"
    ISOCHRONIC_PULSES = "isochronic"
    MODULATION_AM = "am_modulation"
    MODULATION_FM = "fm_modulation"
    SPATIAL_3D = "spatial_3d"
    SPATIAL_8D = "spatial_8d"
    BRAINWAVE_ENTRAINMENT = "brainwave"

class Brainwaveband(Enum):
    DELTA = (0.5, 4.0)
    THETA = (4.0, 8.0)
    ALPHA = (8.0, 13.0)
    BETA = (13.0, 30.0)
    GAMMA = (30.0, 100.0)

class TherapeuticIntent(Enum):
    RELAXATION = "relajación"
    FOCUS = "concentración"
    MEDITATION = "meditación"
    SLEEP = "sueño"
    ENERGY = "energía"
    EMOTIONAL = "equilibrio_emocional"

@dataclass
class NeuroMetrics:
    binaural_frequency_diff: float = 0.0
    binaural_strength: float = 0.0
    isochronic_detected: bool = False
    isochronic_frequency: float = 0.0
    modulation_depth_am: float = 0.0
    modulation_depth_fm: float = 0.0
    spatial_movement_detected: bool = False
    spatial_complexity: float = 0.0
    brainwave_dominance: Dict[str, float] = field(default_factory=dict)
    entrainment_effectiveness: float = 0.0

@dataclass 
class AuroraAnalysisResult:
    score: int
    quality: QualityLevel
    suggestions: List[str]
    issues: List[str]
    technical_metrics: Dict[str, float]
    neuro_metrics: NeuroMetrics
    therapeutic_score: int
    therapeutic_intent: Optional[TherapeuticIntent]
    frequency_analysis: Dict[str, Any]
    phase_coherence: Dict[str, float]
    emotional_flow: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    aurora_compatible: bool = True
    gpt_summary: str = ""

class CarmineAuroraAnalyzer:
    def __init__(self, thresholds: Dict = None, gpt_enabled: bool = False):
        self.sample_rate = SAMPLE_RATE
        self.gpt_enabled = gpt_enabled
        self.thresholds = thresholds or {
            'peak': 0.95,
            'lufs': -23.0,
            'crest_factor': 8.0,
            'phase_corr_min': 0.1,
            'binaural_min_strength': 0.3,
            'binaural_max_diff': 40.0,
            'isochronic_min_depth': 0.2,
            'modulation_max_depth': 0.8,
            'spatial_min_movement': 0.1,
            'entrainment_threshold': 0.6
        }
        logger.info("Carmine Aurora Analyzer V2.1 inicializado")

    def analyze_audio(self, audio: np.ndarray, expected_intent: Optional[TherapeuticIntent] = None) -> AuroraAnalysisResult:
        logger.info("Iniciando análisis Aurora completo...")
        
        # Verificar dimensiones del audio
        if audio.shape[0] == 2 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        left_channel = audio[:, 0]
        right_channel = audio[:, 1]
        score = 100
        therapeutic_score = 100
        issues = []
        suggestions = []
        technical_metrics = {}
        
        # Análisis de picos
        peak_L = np.max(np.abs(left_channel))
        peak_R = np.max(np.abs(right_channel))
        true_peak = max(peak_L, peak_R)
        technical_metrics['true_peak'] = float(true_peak)
        
        if true_peak > self.thresholds['peak']:
            score -= 10
            issues.append("True Peak excedido - riesgo de clipping")
            suggestions.append("Aplicar limitador suave o reducir ganancia general")
        
        # Análisis RMS y LUFS
        rms_L = np.sqrt(np.mean(left_channel**2))
        rms_R = np.sqrt(np.mean(right_channel**2))
        avg_rms = (rms_L + rms_R) / 2
        lufs = -0.691 + 10 * np.log10(avg_rms + 1e-12)
        
        technical_metrics.update({
            'lufs': float(lufs),
            'rms_left': float(rms_L),
            'rms_right': float(rms_R)
        })
        
        if lufs > self.thresholds['lufs']:
            score -= 8
            issues.append("LUFS alto para contenido terapéutico")
            suggestions.append("Reducir volumen para óptima experiencia relajante")
        
        # Análisis Crest Factor
        crest_factor = true_peak / (avg_rms + 1e-6)
        technical_metrics['crest_factor'] = float(crest_factor)
        
        if crest_factor < self.thresholds['crest_factor']:
            score -= 12
            therapeutic_score -= 15
            issues.append("Crest Factor bajo: sobrecompresión detectada")
            suggestions.append("Reducir compresión para preservar naturalidad terapéutica")
        
        # Análisis de correlación de fase
        phase_corr = np.corrcoef(left_channel, right_channel)[0, 1]
        technical_metrics['phase_correlation'] = float(phase_corr)
        
        if phase_corr < self.thresholds['phase_corr_min']:
            score -= 8
            issues.append("Problema de correlación de fase")
            suggestions.append("Verificar paneo o aplicar corrección Mid/Side")
        
        # Análisis neuroacústico
        neuro_metrics = self._analyze_neuroacoustic_features(left_channel, right_channel)
        frequency_analysis = self._analyze_frequency_spectrum(left_channel, right_channel)
        phase_coherence = self._analyze_phase_coherence(left_channel, right_channel)
        emotional_flow = self._analyze_emotional_flow(audio)
        
        # Validaciones neuroacústicas
        if neuro_metrics.binaural_strength > 0:
            if neuro_metrics.binaural_strength < self.thresholds['binaural_min_strength']:
                therapeutic_score -= 10
                suggestions.append("Incrementar fuerza de diferencia binaural")
            elif neuro_metrics.binaural_frequency_diff > self.thresholds['binaural_max_diff']:
                therapeutic_score -= 15
                issues.append("Diferencia binaural excesiva - puede causar incomodidad")
        
        if neuro_metrics.entrainment_effectiveness < self.thresholds['entrainment_threshold']:
            therapeutic_score -= 12
            suggestions.append("Mejorar coherencia para mayor efectividad neuroacústica")
        
        # Detección de intención terapéutica
        detected_intent = self._detect_therapeutic_intent(neuro_metrics, frequency_analysis)
        
        if expected_intent and detected_intent != expected_intent:
            therapeutic_score -= 8
            issues.append(f"Intención detectada ({detected_intent.value}) difiere de esperada ({expected_intent.value})")
        
        # Cálculo de puntuación final
        final_score = int((score * 0.6) + (therapeutic_score * 0.4))
        
        # Determinación de calidad
        if final_score >= 95 and therapeutic_score >= 90:
            quality = QualityLevel.THERAPEUTIC
        elif final_score >= 90:
            quality = QualityLevel.OPTIMAL
        elif final_score >= 70:
            quality = QualityLevel.WARNING
        else:
            quality = QualityLevel.CRITICAL
        
        # Generar resumen GPT si está habilitado
        gpt_summary = ""
        if self.gpt_enabled:
            gpt_summary = self._generate_aurora_summary(final_score, therapeutic_score, quality, neuro_metrics, detected_intent, issues, suggestions)
        
        return AuroraAnalysisResult(
            score=final_score,
            quality=quality,
            suggestions=suggestions,
            issues=issues,
            technical_metrics=technical_metrics,
            neuro_metrics=neuro_metrics,
            therapeutic_score=therapeutic_score,
            therapeutic_intent=detected_intent,
            frequency_analysis=frequency_analysis,
            phase_coherence=phase_coherence,
            emotional_flow=emotional_flow,
            gpt_summary=gpt_summary
        )

    def _analyze_neuroacoustic_features(self, left: np.ndarray, right: np.ndarray) -> NeuroMetrics:
        metrics = NeuroMetrics()
        
        # Análisis FFT
        left_fft = np.abs(fft(left))
        right_fft = np.abs(fft(right))
        freqs = fftfreq(len(left), 1/self.sample_rate)
        
        # Detección de picos dominantes
        left_peaks, _ = signal.find_peaks(left_fft[:len(left_fft)//2], height=np.max(left_fft)*0.1)
        right_peaks, _ = signal.find_peaks(right_fft[:len(right_fft)//2], height=np.max(right_fft)*0.1)
        
        # Análisis binaural
        if len(left_peaks) > 0 and len(right_peaks) > 0:
            left_dominant = freqs[left_peaks[np.argmax(left_fft[left_peaks])]]
            right_dominant = freqs[right_peaks[np.argmax(right_fft[right_peaks])]]
            
            if left_dominant > 0 and right_dominant > 0:
                freq_diff = abs(left_dominant - right_dominant)
                if 1 <= freq_diff <= 40:
                    metrics.binaural_frequency_diff = freq_diff
                    coherence = np.corrcoef(left, right)[0, 1]
                    metrics.binaural_strength = (1 - coherence) * min(freq_diff/40, 1.0)
        
        # Detección de pulsos isocrónicos
        envelope = np.abs(signal.hilbert(left + right))
        envelope_fft = np.abs(fft(envelope))
        envelope_freqs = fftfreq(len(envelope), 1/self.sample_rate)
        iso_range = (envelope_freqs >= 1) & (envelope_freqs <= 40)
        
        if np.any(iso_range):
            iso_peaks, _ = signal.find_peaks(envelope_fft[iso_range], height=np.max(envelope_fft)*0.05)
            if len(iso_peaks) > 0:
                metrics.isochronic_detected = True
                peak_idx = iso_peaks[np.argmax(envelope_fft[iso_range][iso_peaks])]
                metrics.isochronic_frequency = envelope_freqs[iso_range][peak_idx]
        
        # Análisis de modulación AM
        analytical_signal = signal.hilbert(left)
        amplitude_envelope = np.abs(analytical_signal)
        am_depth = (np.max(amplitude_envelope) - np.min(amplitude_envelope)) / (np.max(amplitude_envelope) + np.min(amplitude_envelope) + 1e-6)
        metrics.modulation_depth_am = am_depth
        
        # Análisis de modulación FM
        instantaneous_phase = np.unwrap(np.angle(analytical_signal))
        instantaneous_freq = np.diff(instantaneous_phase) * self.sample_rate / (2 * np.pi)
        fm_depth = np.std(instantaneous_freq) / (np.mean(np.abs(instantaneous_freq)) + 1e-6)
        metrics.modulation_depth_fm = min(fm_depth, 1.0)
        
        # Análisis de movimiento espacial
        window_size = int(self.sample_rate * 0.5)
        left_windows = [left[i:i+window_size] for i in range(0, len(left)-window_size, window_size//2)]
        right_windows = [right[i:i+window_size] for i in range(0, len(right)-window_size, window_size//2)]
        amplitude_ratios = []
        
        for l_win, r_win in zip(left_windows, right_windows):
            if len(l_win) == window_size and len(r_win) == window_size:
                l_rms = np.sqrt(np.mean(l_win**2))
                r_rms = np.sqrt(np.mean(r_win**2))
                amplitude_ratios.append(l_rms / (r_rms + 1e-6))
        
        if len(amplitude_ratios) > 2:
            spatial_movement = np.std(amplitude_ratios)
            metrics.spatial_movement_detected = spatial_movement > self.thresholds['spatial_min_movement']
            metrics.spatial_complexity = min(spatial_movement, 2.0)
        
        # Análisis de ondas cerebrales
        for band in Brainwaveband:
            band_name = band.name.lower()
            low_freq, high_freq = band.value
            sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=self.sample_rate, output='sos')
            filtered = signal.sosfilt(sos, left)
            band_energy = np.mean(filtered**2)
            total_energy = np.mean(left**2)
            metrics.brainwave_dominance[band_name] = band_energy / (total_energy + 1e-12)
        
        # Efectividad de entrainment
        dominant_band = max(metrics.brainwave_dominance.items(), key=lambda x: x[1])
        metrics.entrainment_effectiveness = dominant_band[1]
        
        return metrics

    def _analyze_frequency_spectrum(self, left: np.ndarray, right: np.ndarray) -> Dict[str, Any]:
        left_fft = np.abs(fft(left))
        right_fft = np.abs(fft(right))
        freqs = fftfreq(len(left), 1/self.sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        left_spectrum = left_fft[:len(left_fft)//2]
        right_spectrum = right_fft[:len(right_fft)//2]
        
        # Análisis por bandas de frecuencia
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000),
            'ultra_high': (8000, 20000)
        }
        
        band_energy = {}
        for band_name, (low, high) in bands.items():
            mask = (positive_freqs >= low) & (positive_freqs <= high)
            if np.any(mask):
                left_energy = np.mean(left_spectrum[mask]**2)
                right_energy = np.mean(right_spectrum[mask]**2)
                band_energy[band_name] = {
                    'left': float(left_energy),
                    'right': float(right_energy),
                    'total': float(left_energy + right_energy)
                }
        
        # Frecuencias dominantes
        left_peak_idx = np.argmax(left_spectrum)
        right_peak_idx = np.argmax(right_spectrum)
        
        return {
            'band_energy': band_energy,
            'dominant_freq_left': float(positive_freqs[left_peak_idx]),
            'dominant_freq_right': float(positive_freqs[right_peak_idx]),
            'spectral_centroid_left': float(np.average(positive_freqs, weights=left_spectrum)),
            'spectral_centroid_right': float(np.average(positive_freqs, weights=right_spectrum))
        }

    def _analyze_phase_coherence(self, left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
        global_coherence = float(np.corrcoef(left, right)[0, 1])
        window_size = int(self.sample_rate * 2)
        coherences = []
        
        for i in range(0, len(left) - window_size, window_size // 2):
            l_window = left[i:i+window_size]
            r_window = right[i:i+window_size]
            
            # CORRECCIÓN APLICADA: Separar las condiciones en líneas diferentes
            if len(l_window) == window_size:
                window_coh = np.corrcoef(l_window, r_window)[0, 1]
                if not np.isnan(window_coh):
                    coherences.append(window_coh)
        
        return {
            'global_coherence': global_coherence,
            'average_coherence': float(np.mean(coherences)) if coherences else 0.0,
            'coherence_stability': float(1.0 - np.std(coherences)) if coherences else 0.0,
            'min_coherence': float(np.min(coherences)) if coherences else 0.0,
            'max_coherence': float(np.max(coherences)) if coherences else 0.0
        }

    def _analyze_emotional_flow(self, audio: np.ndarray) -> Dict[str, float]:
        segment_duration = 30
        segment_samples = int(self.sample_rate * segment_duration)
        segments = []
        
        for i in range(0, len(audio) - segment_samples, segment_samples):
            segments.append(audio[i:i+segment_samples])
        
        if not segments:
            return {'emotional_stability': 0.0, 'energy_flow': 0.0, 'dynamic_range': 0.0}
        
        segment_energies = []
        segment_dynamics = []
        
        for segment in segments:
            energy = np.mean(segment**2)
            segment_energies.append(energy)
            dynamic_range = np.max(np.abs(segment)) - np.mean(np.abs(segment))
            segment_dynamics.append(dynamic_range)
        
        emotional_stability = 1.0 - min(np.std(segment_energies) / (np.mean(segment_energies) + 1e-6), 1.0)
        energy_changes = np.abs(np.diff(segment_energies))
        energy_flow = 1.0 - min(np.mean(energy_changes) / (np.mean(segment_energies) + 1e-6), 1.0)
        avg_dynamic_range = np.mean(segment_dynamics)
        
        return {
            'emotional_stability': float(emotional_stability),
            'energy_flow': float(energy_flow),
            'dynamic_range': float(avg_dynamic_range),
            'segment_count': len(segments)
        }

    def _detect_therapeutic_intent(self, neuro_metrics: NeuroMetrics, freq_analysis: Dict) -> TherapeuticIntent:
        dominant_wave = max(neuro_metrics.brainwave_dominance.items(), key=lambda x: x[1])
        dominant_band, strength = dominant_wave
        
        if dominant_band == 'delta' and strength > 0.3:
            return TherapeuticIntent.SLEEP
        elif dominant_band == 'theta' and strength > 0.25:
            return TherapeuticIntent.MEDITATION if neuro_metrics.binaural_strength > 0.3 else TherapeuticIntent.RELAXATION
        elif dominant_band == 'alpha' and strength > 0.2:
            return TherapeuticIntent.RELAXATION
        elif dominant_band == 'beta' and strength > 0.3:
            return TherapeuticIntent.FOCUS
        elif dominant_band == 'gamma' and strength > 0.2:
            return TherapeuticIntent.ENERGY
        
        if neuro_metrics.spatial_movement_detected and neuro_metrics.spatial_complexity > 0.5:
            return TherapeuticIntent.EMOTIONAL
        
        low_freq_energy = freq_analysis['band_energy'].get('bass', {}).get('total', 0)
        high_freq_energy = freq_analysis['band_energy'].get('high', {}).get('total', 0)
        
        return TherapeuticIntent.RELAXATION if low_freq_energy > high_freq_energy * 2 else TherapeuticIntent.FOCUS

    def _generate_aurora_summary(self, score: int, therapeutic_score: int, quality: QualityLevel, neuro_metrics: NeuroMetrics, intent: TherapeuticIntent, issues: List[str], suggestions: List[str]) -> str:
        summary = f"🎵 **Análisis Aurora Completo**\n\n**Puntuación General:** {score}/100\n**Puntuación Terapéutica:** {therapeutic_score}/100\n**Calidad:** {quality.value}\n**Intención Detectada:** {intent.value.title()}\n\n"
        
        summary += "🧠 **Características Neuroacústicas:**\n"
        if neuro_metrics.binaural_strength > 0:
            summary += f"• Binaurales: {neuro_metrics.binaural_frequency_diff:.1f}Hz de diferencia, fuerza {neuro_metrics.binaural_strength:.2f}\n"
        if neuro_metrics.isochronic_detected:
            summary += f"• Pulsos isocrónicos: {neuro_metrics.isochronic_frequency:.1f}Hz detectados\n"
        if neuro_metrics.spatial_movement_detected:
            summary += f"• Efectos espaciales: complejidad {neuro_metrics.spatial_complexity:.2f}\n"
        summary += f"• Efectividad neuroacústica: {neuro_metrics.entrainment_effectiveness:.1%}\n\n"
        
        dominant_waves = sorted(neuro_metrics.brainwave_dominance.items(), key=lambda x: x[1], reverse=True)[:3]
        summary += "🌊 **Ondas Cerebrales Dominantes:**\n"
        for wave, strength in dominant_waves:
            summary += f"• {wave.title()}: {strength:.1%}\n"
        
        if issues:
            summary += f"\n⚠️ **Observaciones:** {', '.join(issues)}\n"
        if suggestions:
            summary += f"\n💡 **Recomendaciones:** {'; '.join(suggestions)}\n"
        
        if quality == QualityLevel.THERAPEUTIC:
            summary += "\n✨ **Conclusión:** Audio de calidad terapéutica óptima para neuroestimulación."
        elif quality == QualityLevel.OPTIMAL:
            summary += "\n✅ **Conclusión:** Audio de alta calidad, apto para uso terapéutico."
        elif quality == QualityLevel.WARNING:
            summary += "\n🔄 **Conclusión:** Audio funcional, pero con oportunidades de mejora."
        else:
            summary += "\n🔧 **Conclusión:** Audio requiere optimización antes del uso terapéutico."
        
        return summary

    def aplicar_correcciones_neuroacusticas(self, audio: np.ndarray, analysis_result: AuroraAnalysisResult = None, objetivo: str = None, modo_agresivo: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        logger.info("🔧 Iniciando correcciones neuroacústicas automáticas...")
        
        if analysis_result is None:
            analysis_result = self.analyze_audio(audio)
        
        if audio.shape[0] == 2 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        audio_corregido = audio.copy()
        correcciones_aplicadas = []
        metadata_correcciones = {
            "correcciones_aplicadas": correcciones_aplicadas,
            "score_original": analysis_result.score,
            "score_objetivo": 85,
            "objetivo_terapeutico": objetivo,
            "modo_agresivo": modo_agresivo
        }
        
        # Aplicar correcciones según necesidades detectadas
        if analysis_result.neuro_metrics.binaural_strength < self.thresholds['binaural_min_strength']:
            audio_corregido, corr_meta = self._corregir_fuerza_binaural(audio_corregido, analysis_result.neuro_metrics, modo_agresivo)
            correcciones_aplicadas.append({"tipo": "fuerza_binaural", "detalles": corr_meta})
            logger.info("✅ Corrección de fuerza binaural aplicada")
        
        if analysis_result.neuro_metrics.binaural_frequency_diff > self.thresholds['binaural_max_diff']:
            audio_corregido, corr_meta = self._corregir_diferencia_binaural_excesiva(audio_corregido, analysis_result.neuro_metrics)
            correcciones_aplicadas.append({"tipo": "diferencia_binaural", "detalles": corr_meta})
            logger.info("✅ Corrección de diferencia binaural excesiva aplicada")
        
        if objetivo:
            audio_corregido, corr_meta = self._optimizar_ondas_cerebrales(audio_corregido, objetivo, analysis_result.neuro_metrics, modo_agresivo)
            correcciones_aplicadas.append({"tipo": "ondas_cerebrales", "detalles": corr_meta})
            logger.info(f"✅ Optimización de ondas cerebrales para '{objetivo}' aplicada")
        
        if analysis_result.neuro_metrics.modulation_depth_am < self.thresholds['isochronic_min_depth']:
            audio_corregido, corr_meta = self._incrementar_modulacion_am(audio_corregido, analysis_result.neuro_metrics)
            correcciones_aplicadas.append({"tipo": "modulacion_am", "detalles": corr_meta})
            logger.info("✅ Incremento de modulación AM aplicado")
        
        if analysis_result.neuro_metrics.entrainment_effectiveness < self.thresholds['entrainment_threshold']:
            audio_corregido, corr_meta = self._mejorar_coherencia_neuroacustica(audio_corregido, analysis_result.neuro_metrics, modo_agresivo)
            correcciones_aplicadas.append({"tipo": "coherencia_neuroacustica", "detalles": corr_meta})
            logger.info("✅ Mejora de coherencia neuroacústica aplicada")
        
        if not analysis_result.neuro_metrics.spatial_movement_detected and objetivo in ["creatividad", "3d", "espacial"]:
            audio_corregido, corr_meta = self._incrementar_movimiento_espacial(audio_corregido, objetivo)
            correcciones_aplicadas.append({"tipo": "movimiento_espacial", "detalles": corr_meta})
            logger.info("✅ Incremento de movimiento espacial aplicado")
        
        # Normalización segura
        audio_corregido = self._normalizar_seguro(audio_corregido)
        metadata_correcciones["total_correcciones"] = len(correcciones_aplicadas)
        metadata_correcciones["audio_shape"] = audio_corregido.shape
        
        logger.info(f"🎯 Correcciones neuroacústicas completadas: {len(correcciones_aplicadas)} aplicadas")
        
        return audio_corregido, metadata_correcciones

    def _corregir_fuerza_binaural(self, audio: np.ndarray, neuro_metrics: NeuroMetrics, modo_agresivo: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        left_channel = audio[:, 0].copy()
        right_channel = audio[:, 1].copy()
        
        # Análisis de frecuencias dominantes
        left_fft = np.abs(fft(left_channel))
        right_fft = np.abs(fft(right_channel))
        freqs = fftfreq(len(left_channel), 1/self.sample_rate)
        
        left_peak_idx = np.argmax(left_fft[:len(left_fft)//2])
        right_peak_idx = np.argmax(right_fft[:len(right_fft)//2])
        
        left_freq = abs(freqs[left_peak_idx])
        right_freq = abs(freqs[right_peak_idx])
        
        # Parámetros de corrección
        target_diff = 15.0 if modo_agresivo else 10.0
        
        # Generar ondas de corrección
        t = np.linspace(0, len(left_channel) / self.sample_rate, len(left_channel))
        base_freq = max(left_freq, 100.0)
        
        left_mod = 0.1 * np.sin(2 * np.pi * base_freq * t)
        right_freq_adjusted = base_freq + target_diff
        right_mod = 0.1 * np.sin(2 * np.pi * right_freq_adjusted * t)
        
        strength_factor = 0.5 if modo_agresivo else 0.3
        left_channel = left_channel + (left_mod * strength_factor)
        right_channel = right_channel + (right_mod * strength_factor)
        
        audio_corregido = np.column_stack([left_channel, right_channel])
        
        metadata = {
            "target_binaural_diff": target_diff,
            "base_frequency": base_freq,
            "strength_factor": strength_factor,
            "original_binaural_strength": neuro_metrics.binaural_strength,
            "modo_agresivo": modo_agresivo
        }
        
        return audio_corregido, metadata

    def _corregir_diferencia_binaural_excesiva(self, audio: np.ndarray, neuro_metrics: NeuroMetrics) -> Tuple[np.ndarray, Dict[str, Any]]:
        left_channel = audio[:, 0].copy()
        right_channel = audio[:, 1].copy()
        
        # Mezcla con señal mono para reducir diferencia
        mid_channel = (left_channel + right_channel) / 2
        excess_factor = min(1.0, (neuro_metrics.binaural_frequency_diff - 40.0) / 20.0)
        mix_factor = 0.3 * excess_factor
        
        left_corrected = left_channel * (1 - mix_factor) + mid_channel * mix_factor
        right_corrected = right_channel * (1 - mix_factor) + mid_channel * mix_factor
        
        audio_corregido = np.column_stack([left_corrected, right_corrected])
        
        metadata = {
            "original_binaural_diff": neuro_metrics.binaural_frequency_diff,
            "excess_factor": excess_factor,
            "mix_factor": mix_factor,
            "target_max_diff": 40.0
        }
        
        return audio_corregido, metadata

    def _optimizar_ondas_cerebrales(self, audio: np.ndarray, objetivo: str, neuro_metrics: NeuroMetrics, modo_agresivo: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        objetivos_ondas = {
            "relajacion": {"alpha": 0.4, "theta": 0.3, "delta": 0.2, "beta": 0.1},
            "concentracion": {"beta": 0.4, "alpha": 0.3, "gamma": 0.2, "theta": 0.1},
            "meditacion": {"theta": 0.5, "alpha": 0.3, "delta": 0.2},
            "sueño": {"delta": 0.6, "theta": 0.3, "alpha": 0.1},
            "energia": {"beta": 0.4, "gamma": 0.3, "alpha": 0.2, "theta": 0.1},
            "creatividad": {"alpha": 0.4, "theta": 0.3, "gamma": 0.2, "beta": 0.1}
        }
        
        objetivo_clean = objetivo.lower().replace("_", " ")
        target_bands = objetivos_ondas.get(objetivo_clean, objetivos_ondas["relajacion"])
        
        audio_corregido = audio.copy()
        adjustments_made = {}
        
        for band_name, target_ratio in target_bands.items():
            current_ratio = neuro_metrics.brainwave_dominance.get(band_name, 0)
            if abs(current_ratio - target_ratio) > 0.1:
                band_enum = getattr(Brainwaveband, band_name.upper())
                low_freq, high_freq = band_enum.value
                adjustment = (target_ratio - current_ratio) * (0.5 if modo_agresivo else 0.3)
                
                if adjustment > 0:
                    audio_corregido = self._realzar_banda_frecuencia(audio_corregido, low_freq, high_freq, 1 + adjustment)
                else:
                    audio_corregido = self._atenuar_banda_frecuencia(audio_corregido, low_freq, high_freq, 1 + adjustment)
                
                adjustments_made[band_name] = {
                    "target_ratio": target_ratio,
                    "current_ratio": current_ratio,
                    "adjustment": adjustment,
                    "freq_range": (low_freq, high_freq)
                }
        
        metadata = {
            "objetivo": objetivo,
            "target_bands": target_bands,
            "adjustments_made": adjustments_made,
            "modo_agresivo": modo_agresivo
        }
        
        return audio_corregido, metadata

    def _incrementar_modulacion_am(self, audio: np.ndarray, neuro_metrics: NeuroMetrics) -> Tuple[np.ndarray, Dict[str, Any]]:
        mod_freq = neuro_metrics.isochronic_frequency if neuro_metrics.isochronic_detected and neuro_metrics.isochronic_frequency > 0 else 10.0
        target_depth = max(0.3, self.thresholds['isochronic_min_depth'])
        current_depth = neuro_metrics.modulation_depth_am
        depth_increase = min(0.2, target_depth - current_depth)
        
        t = np.linspace(0, audio.shape[0] / self.sample_rate, audio.shape[0])
        modulation_signal = 1 + depth_increase * np.sin(2 * np.pi * mod_freq * t)
        
        audio_corregido = audio.copy()
        for channel in range(audio_corregido.shape[1]):
            audio_corregido[:, channel] *= modulation_signal
        
        metadata = {
            "modulation_frequency": mod_freq,
            "target_depth": target_depth,
            "current_depth": current_depth,
            "depth_increase": depth_increase,
            "original_isochronic_detected": neuro_metrics.isochronic_detected
        }
        
        return audio_corregido, metadata

    def _mejorar_coherencia_neuroacustica(self, audio: np.ndarray, neuro_metrics: NeuroMetrics, modo_agresivo: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        from scipy.signal import savgol_filter
        
        audio_corregido = audio.copy()
        window_length = 31 if modo_agresivo else 51
        polyorder = 3
        
        # Aplicar filtro suavizante
        for channel in range(audio_corregido.shape[1]):
            if len(audio_corregido[:, channel]) > window_length:
                audio_corregido[:, channel] = savgol_filter(audio_corregido[:, channel], window_length, polyorder)
        
        # Mejorar correlación entre canales
        correlation_factor = 0.2 if modo_agresivo else 0.1
        mid_signal = np.mean(audio_corregido, axis=1, keepdims=True)
        audio_corregido = audio_corregido * (1 - correlation_factor) + mid_signal * correlation_factor
        
        metadata = {
            "original_entrainment_effectiveness": neuro_metrics.entrainment_effectiveness,
            "target_entrainment": self.thresholds['entrainment_threshold'],
            "window_length": window_length,
            "correlation_factor": correlation_factor,
            "modo_agresivo": modo_agresivo
        }
        
        return audio_corregido, metadata

    def _incrementar_movimiento_espacial(self, audio: np.ndarray, objetivo: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        movement_freq = 0.2 if "3d" in objetivo.lower() or "espacial" in objetivo.lower() else 0.1
        movement_depth = 0.3 if "3d" in objetivo.lower() or "espacial" in objetivo.lower() else 0.2
        
        t = np.linspace(0, audio.shape[0] / self.sample_rate, audio.shape[0])
        pan_left = 0.5 + movement_depth * np.sin(2 * np.pi * movement_freq * t)
        pan_right = 0.5 - movement_depth * np.sin(2 * np.pi * movement_freq * t)
        
        pan_left = np.clip(pan_left, 0.1, 0.9)
        pan_right = np.clip(pan_right, 0.1, 0.9)
        
        audio_corregido = audio.copy()
        audio_corregido[:, 0] *= pan_left
        audio_corregido[:, 1] *= pan_right
        
        # Normalización de seguridad
        max_val = np.max(np.abs(audio_corregido))
        if max_val > 0.9:
            audio_corregido *= 0.9 / max_val
        
        metadata = {
            "movement_frequency": movement_freq,
            "movement_depth": movement_depth,
            "objetivo": objetivo,
            "pan_range": (np.min(pan_left), np.max(pan_left))
        }
        
        return audio_corregido, metadata

    def _realzar_banda_frecuencia(self, audio: np.ndarray, low_freq: float, high_freq: float, gain: float) -> np.ndarray:
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(4, [low_freq, high_freq], btype='band', fs=self.sample_rate, output='sos')
            audio_enhanced = audio.copy()
            
            for channel in range(audio.shape[1]):
                filtered = sosfilt(sos, audio[:, channel])
                audio_enhanced[:, channel] = audio[:, channel] + filtered * (gain - 1)
            
            return audio_enhanced
        except Exception as e:
            logger.warning(f"Error en realce de banda {low_freq}-{high_freq}Hz: {e}")
            return audio

    def _atenuar_banda_frecuencia(self, audio: np.ndarray, low_freq: float, high_freq: float, gain: float) -> np.ndarray:
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(4, [low_freq, high_freq], btype='band', fs=self.sample_rate, output='sos')
            audio_attenuated = audio.copy()
            
            for channel in range(audio.shape[1]):
                filtered = sosfilt(sos, audio[:, channel])
                audio_attenuated[:, channel] = audio[:, channel] - filtered * (1 - gain)
            
            return audio_attenuated
        except Exception as e:
            logger.warning(f"Error en atenuación de banda {low_freq}-{high_freq}Hz: {e}")
            return audio

    def _normalizar_seguro(self, audio: np.ndarray) -> np.ndarray:
        """Normalización segura del audio"""
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.9 / max_val)
        return np.clip(audio, -1.0, 1.0)

    def export_analysis_json(self, result: AuroraAnalysisResult, filename: str):
        """Exporta los resultados del análisis a JSON"""
        export_data = {
            'metadata': {
                'analyzer_version': '2.1',
                'timestamp': result.timestamp,
                'aurora_compatible': result.aurora_compatible
            },
            'scores': {
                'technical_score': result.score,
                'therapeutic_score': result.therapeutic_score,
                'quality_level': result.quality.value
            },
            'technical_metrics': result.technical_metrics,
            'neuroacoustic_metrics': {
                'binaural_frequency_diff': result.neuro_metrics.binaural_frequency_diff,
                'binaural_strength': result.neuro_metrics.binaural_strength,
                'isochronic_detected': result.neuro_metrics.isochronic_detected,
                'isochronic_frequency': result.neuro_metrics.isochronic_frequency,
                'modulation_depth_am': result.neuro_metrics.modulation_depth_am,
                'modulation_depth_fm': result.neuro_metrics.modulation_depth_fm,
                'spatial_movement_detected': result.neuro_metrics.spatial_movement_detected,
                'spatial_complexity': result.neuro_metrics.spatial_complexity,
                'brainwave_dominance': result.neuro_metrics.brainwave_dominance,
                'entrainment_effectiveness': result.neuro_metrics.entrainment_effectiveness
            },
            'frequency_analysis': result.frequency_analysis,
            'phase_coherence': result.phase_coherence,
            'emotional_flow': result.emotional_flow,
            'therapeutic_intent': result.therapeutic_intent.value if result.therapeutic_intent else None,
            'issues': result.issues,
            'suggestions': result.suggestions,
            'gpt_summary': result.gpt_summary
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


# ===== VERSIÓN MEJORADA CON FUNCIONALIDADES ADICIONALES =====

class CarmineAnalyzerEnhanced(CarmineAuroraAnalyzer):
    """Versión mejorada del analizador Carmine con funcionalidades adicionales"""
    
    def __init__(self, thresholds: Dict = None, gpt_enabled: bool = False):
        super().__init__(thresholds, gpt_enabled)
        self.version = "2.1_ENHANCED"
        self.cache_analisis = {}
        self.estadisticas_uso = {
            "analisis_realizados": 0,
            "correcciones_aplicadas": 0,
            "tiempo_promedio_analisis": 0.0,
            "calidad_promedio": 0.0
        }
        logger.info("🔬 Carmine Aurora Analyzer V2.1 Enhanced inicializado")
    
    def analyze_audio_cached(self, audio: np.ndarray, expected_intent: Optional[TherapeuticIntent] = None) -> AuroraAnalysisResult:
        """Análisis con cache para mejorar rendimiento en análisis repetitivos"""
        # Crear hash del audio para cache
        audio_hash = hash(audio.tobytes())
        cache_key = f"{audio_hash}_{expected_intent}"
        
        if cache_key in self.cache_analisis:
            logger.info("📋 Usando resultado de análisis desde cache")
            return self.cache_analisis[cache_key]
        
        # Análisis normal
        start_time = time.time()
        resultado = self.analyze_audio(audio, expected_intent)
        tiempo_analisis = time.time() - start_time
        
        # Actualizar estadísticas
        self._actualizar_estadisticas(resultado, tiempo_analisis)
        
        # Guardar en cache (limitar tamaño del cache)
        if len(self.cache_analisis) > 10:
            # Remover el análisis más antiguo
            self.cache_analisis.pop(next(iter(self.cache_analisis)))
        
        self.cache_analisis[cache_key] = resultado
        return resultado
    
    def _actualizar_estadisticas(self, resultado: AuroraAnalysisResult, tiempo_analisis: float):
        """Actualiza estadísticas de uso del analizador"""
        stats = self.estadisticas_uso
        total_analisis = stats["analisis_realizados"]
        
        # Actualizar promedios
        stats["tiempo_promedio_analisis"] = (
            (stats["tiempo_promedio_analisis"] * total_analisis + tiempo_analisis) / 
            (total_analisis + 1)
        )
        
        stats["calidad_promedio"] = (
            (stats["calidad_promedio"] * total_analisis + resultado.score) / 
            (total_analisis + 1)
        )
        
        stats["analisis_realizados"] += 1
    
    def analisis_rapido(self, audio: np.ndarray) -> Dict[str, Any]:
        """Análisis rápido que retorna solo métricas esenciales"""
        if audio.size == 0:
            return {"error": "Audio vacío", "calidad_estimada": 0}
        
        try:
            # Métricas básicas rápidas
            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.max(np.abs(audio)))
            
            if audio.ndim == 2 and audio.shape[0] == 2:
                coherencia = float(np.corrcoef(audio[0], audio[1])[0, 1])
            else:
                coherencia = 0.8
            
            # Score simplificado
            calidad_score = min(100, max(50, 80 + (1 - min(peak, 1.0)) * 20))
            
            return {
                "calidad_score": calidad_score,
                "peak": peak,
                "rms": rms,
                "coherencia": coherencia,
                "nivel_calidad": "ALTO" if calidad_score >= 85 else "MEDIO" if calidad_score >= 70 else "BAJO",
                "tiempo_analisis": "rapido",
                "version_analyzer": self.version
            }
        except Exception as e:
            return {"error": str(e), "calidad_estimada": 0}
    
    def aplicar_auto_correcciones(self, audio: np.ndarray, objetivo: str = "relajacion") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Aplica correcciones automáticas inteligentes basadas en análisis"""
        # Análisis inicial
        resultado_analisis = self.analyze_audio(audio)
        
        # Determinar nivel de agresividad basado en calidad
        modo_agresivo = resultado_analisis.score < 70
        
        # Aplicar correcciones
        audio_corregido, metadata = self.aplicar_correcciones_neuroacusticas(
            audio, resultado_analisis, objetivo, modo_agresivo
        )
        
        # Análisis post-corrección para validar mejoras
        resultado_post = self.analyze_audio(audio_corregido)
        
        # Actualizar estadísticas
        self.estadisticas_uso["correcciones_aplicadas"] += 1
        
        # Metadata enriquecida
        metadata_completa = {
            **metadata,
            "mejora_calidad": resultado_post.score - resultado_analisis.score,
            "calidad_original": resultado_analisis.score,
            "calidad_final": resultado_post.score,
            "correcciones_exitosas": resultado_post.score > resultado_analisis.score,
            "modo_auto": True,
            "analyzer_version": self.version
        }
        
        logger.info(f"🎯 Auto-correcciones completadas: {resultado_analisis.score} → {resultado_post.score}")
        
        return audio_corregido, metadata_completa
    
    def obtener_reporte_estadisticas(self) -> Dict[str, Any]:
        """Obtiene reporte completo de estadísticas del analizador"""
        return {
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "estadisticas_uso": self.estadisticas_uso,
            "cache_size": len(self.cache_analisis),
            "thresholds_configurados": self.thresholds,
            "gpt_habilitado": self.gpt_enabled,
            "capacidades": [
                "analisis_completo",
                "analisis_cached", 
                "analisis_rapido",
                "auto_correcciones",
                "estadisticas_uso",
                "exportacion_json"
            ]
        }
    
    def limpiar_cache(self):
        """Limpia el cache de análisis"""
        self.cache_analisis.clear()
        logger.info("🧹 Cache de análisis Carmine limpiado")


# ===== FUNCIÓN DE MIGRACIÓN PARA USAR LA VERSIÓN MEJORADA =====

def crear_carmine_analyzer_mejorado(thresholds: Dict = None, gpt_enabled: bool = False) -> CarmineAnalyzerEnhanced:
    """Crea una instancia del analizador Carmine mejorado"""
    return CarmineAnalyzerEnhanced(thresholds, gpt_enabled)


# ===== PARA TESTING Y VALIDACIÓN =====

def test_carmine_syntax():
    """Función de test para validar que la sintaxis esté corregida"""
    try:
        # Crear datos de prueba
        test_audio = np.random.normal(0, 0.1, (2, 44100))  # 1 segundo de audio estéreo
        
        # Crear analizador original
        analyzer = CarmineAuroraAnalyzer()
        print("✅ CarmineAuroraAnalyzer creado exitosamente")
        
        # Test análisis completo
        resultado_completo = analyzer.analyze_audio(test_audio)
        print(f"✅ Análisis completo exitoso - Score: {resultado_completo.score}")
        
        # Crear analizador mejorado
        analyzer_enhanced = CarmineAnalyzerEnhanced()
        print("✅ CarmineAnalyzerEnhanced creado exitosamente")
        
        # Test análisis rápido
        resultado_rapido = analyzer_enhanced.analisis_rapido(test_audio)
        print(f"✅ Análisis rápido exitoso - Score: {resultado_rapido.get('calidad_score', 'N/A')}")
        
        # Test análisis con cache
        resultado_cached = analyzer_enhanced.analyze_audio_cached(test_audio)
        print(f"✅ Análisis con cache exitoso - Score: {resultado_cached.score}")
        
        # Test auto-correcciones
        audio_corregido, metadata = analyzer_enhanced.aplicar_auto_correcciones(test_audio, "relajacion")
        print(f"✅ Auto-correcciones exitosas - Mejora: {metadata.get('mejora_calidad', 'N/A')}")
        
        # Test estadísticas
        stats = analyzer_enhanced.obtener_reporte_estadisticas()
        print(f"✅ Reporte de estadísticas exitoso - Análisis realizados: {stats['estadisticas_uso']['analisis_realizados']}")
        
        print(f"\n🎉 ¡Todas las pruebas de sintaxis pasaron exitosamente!")
        print(f"🔬 Versión original: CarmineAuroraAnalyzer")
        print(f"🚀 Versión mejorada: CarmineAnalyzerEnhanced v{analyzer_enhanced.version}")
        return True
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False


# Ejecutar test si se ejecuta directamente
if __name__ == "__main__":
    print("🔬 Carmine Analyzer V2.1 - Test de Sintaxis Completamente Corregida")
    print("=" * 70)
    
    success = test_carmine_syntax()
    
    if success:
        print("\n✨ Carmine Analyzer corregido y mejorado exitosamente")
        print("🚀 Listo para usar en Aurora V7")
        print("🎯 Sintaxis completamente validada")
        print("\n📋 Resumen de correcciones aplicadas:")
        print("   • Método _normalizar_seguro: asteriscos removidos del nombre")
        print("   • Método _analyze_phase_coherence: condicionales separadas")
        print("   • Importaciones: organizadas correctamente")
        print("   • Estructuras de clase: sintaxis validada")
        print("   • Funciones auxiliares: sintaxis verificada")
    else:
        print("\n❌ Aún hay problemas por resolver")
        print("🔧 Revisa los errores reportados arriba")
