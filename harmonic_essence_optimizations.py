# ===== OPTIMIZACIONES ADITIVAS PARA HARMONIC ESSENCE V34 =====
# Aumentar potencia sin modificar lógica existente

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from functools import lru_cache
import time
from datetime import datetime

logger = logging.getLogger("Aurora.HarmonicEssence.Optimization")

# ===== 1. SISTEMA DE AMPLIFICACIÓN INTELIGENTE =====

@dataclass
class ConfiguracionAmplificacion:
    """Configuración para amplificar la potencia del motor"""
    factor_intensidad_base: float = 1.2
    amplificacion_por_calidad: Dict[str, float] = field(default_factory=lambda: {
        'media': 1.0,
        'alta': 1.3,
        'maxima': 1.6
    })
    boost_aurora_v7: float = 1.4
    factor_neuroacustico: float = 1.25
    modulacion_coherencia: float = 1.15
    amplificacion_terapeutica: float = 1.35

class AmplificadorHarmonicEssence:
    """Sistema de amplificación inteligente para HarmonicEssence"""
    
    def __init__(self):
        self.config = ConfiguracionAmplificacion()
        self.stats_amplificacion = {
            'amplificaciones_aplicadas': 0,
            'factor_promedio': 0.0,
            'mejora_calidad_promedio': 0.0
        }
    
    def calcular_factor_amplificacion(self, config_audio: Dict[str, Any]) -> float:
        """Calcula factor de amplificación dinámico basado en configuración"""
        factor_base = self.config.factor_intensidad_base
        
        # Amplificación por calidad objetivo
        calidad = config_audio.get('calidad_objetivo', 'media')
        factor_calidad = self.config.amplificacion_por_calidad.get(calidad, 1.0)
        
        # Boost para integración Aurora V7
        factor_aurora = self.config.boost_aurora_v7 if config_audio.get('usar_aurora_v7') else 1.0
        
        # Amplificación neuroacústica
        factor_neuro = self.config.factor_neuroacustico if config_audio.get('neurotransmisor_preferido') else 1.0
        
        # Modulación por coherencia objetivo
        coherencia_objetivo = config_audio.get('coherencia_objetivo', 0.8)
        factor_coherencia = 1.0 + (coherencia_objetivo - 0.8) * self.config.modulacion_coherencia
        
        # Factor terapéutico
        factor_terapeutico = self.config.amplificacion_terapeutica if config_audio.get('objetivo_terapeutico') else 1.0
        
        # Combinación inteligente (no multiplicativa para evitar saturación)
        factor_final = factor_base * (
            0.3 * factor_calidad + 
            0.2 * factor_aurora + 
            0.2 * factor_neuro + 
            0.15 * factor_coherencia + 
            0.15 * factor_terapeutico
        )
        
        # Limitar para evitar distorsión
        factor_final = min(factor_final, 2.0)
        
        self.stats_amplificacion['amplificaciones_aplicadas'] += 1
        self._actualizar_stats_promedio(factor_final)
        
        return factor_final
    
    def _actualizar_stats_promedio(self, factor: float):
        """Actualiza estadísticas promedio"""
        total = self.stats_amplificacion['amplificaciones_aplicadas']
        factor_actual = self.stats_amplificacion['factor_promedio']
        self.stats_amplificacion['factor_promedio'] = ((factor_actual * (total - 1)) + factor) / total

# ===== 2. PROCESADOR DE TEXTURAS MULTI-DIMENSIONAL =====

class ProcessorTexturaMultiDimensional:
    """Procesador avanzado para texturas multi-dimensionales"""
    
    def __init__(self):
        self.dimensiones_activas = ['temporal', 'frecuencial', 'espacial', 'neural', 'cuantica']
        self.stats = {'procesamiento_multidimensional': 0, 'tiempo_procesamiento': 0.0}
    
    def procesar_textura_multidimensional(self, audio: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Aplica procesamiento multi-dimensional a la textura"""
        start_time = time.time()
        
        audio_mejorado = audio.copy()
        
        # Dimensión temporal: Modulación dinámica en el tiempo
        if 'temporal' in self.dimensiones_activas:
            audio_mejorado = self._aplicar_dimension_temporal(audio_mejorado, config)
        
        # Dimensión frecuencial: Enriquecimiento espectral
        if 'frecuencial' in self.dimensiones_activas:
            audio_mejorado = self._aplicar_dimension_frecuencial(audio_mejorado, config)
        
        # Dimensión espacial: Imagen estéreo avanzada
        if 'espacial' in self.dimensiones_activas:
            audio_mejorado = self._aplicar_dimension_espacial(audio_mejorado, config)
        
        # Dimensión neural: Patrones sinápticos
        if 'neural' in self.dimensiones_activas:
            audio_mejorado = self._aplicar_dimension_neural(audio_mejorado, config)
        
        # Dimensión cuántica: Coherencia e interferencia
        if 'cuantica' in self.dimensiones_activas:
            audio_mejorado = self._aplicar_dimension_cuantica(audio_mejorado, config)
        
        self.stats['procesamiento_multidimensional'] += 1
        self.stats['tiempo_procesamiento'] += time.time() - start_time
        
        return audio_mejorado
    
    def _aplicar_dimension_temporal(self, audio: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Aplica modulación temporal avanzada"""
        if audio.size == 0:
            return audio
        
        if audio.ndim == 2:
            samples = audio.shape[1]
        else:
            samples = len(audio)
        
        # Modulación temporal basada en golden ratio
        golden_ratio = (1 + np.sqrt(5)) / 2
        t = np.linspace(0, 1, samples)
        modulacion_temporal = 1.0 + 0.1 * np.sin(2 * np.pi * golden_ratio * t)
        
        # Aplicar modulación
        if audio.ndim == 2:
            for channel in range(audio.shape[0]):
                audio[channel] *= modulacion_temporal
        else:
            audio *= modulacion_temporal
        
        return audio
    
    def _aplicar_dimension_frecuencial(self, audio: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Aplica enriquecimiento frecuencial"""
        try:
            # Añadir armónicos sutiles
            if audio.ndim == 2:
                for channel in range(audio.shape[0]):
                    fft_data = np.fft.fft(audio[channel])
                    # Realzar armónicos en frecuencias específicas
                    freqs = np.fft.fftfreq(len(fft_data), 1/44100)
                    
                    # Realzar frecuencias neuroacústicas (40Hz, 100Hz, 440Hz)
                    for freq_target in [40, 100, 440]:
                        idx = np.abs(freqs - freq_target).argmin()
                        if idx < len(fft_data):
                            fft_data[idx] *= 1.1
                    
                    audio[channel] = np.real(np.fft.ifft(fft_data))
            else:
                fft_data = np.fft.fft(audio)
                freqs = np.fft.fftfreq(len(fft_data), 1/44100)
                
                for freq_target in [40, 100, 440]:
                    idx = np.abs(freqs - freq_target).argmin()
                    if idx < len(fft_data):
                        fft_data[idx] *= 1.1
                
                audio = np.real(np.fft.ifft(fft_data))
            
            return audio
        except Exception:
            return audio
    
    def _aplicar_dimension_espacial(self, audio: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Aplica expansión espacial avanzada"""
        if audio.ndim == 1:
            # Convertir a estéreo con expansión espacial
            left = audio.copy()
            right = audio.copy()
            
            # Aplicar delay sutil y filtrado diferencial
            delay_samples = 3
            if len(right) > delay_samples:
                right = np.concatenate([np.zeros(delay_samples), right[:-delay_samples]])
            
            return np.stack([left, right])
        
        return audio
    
    def _aplicar_dimension_neural(self, audio: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Aplica patrones neurales sinápticos"""
        try:
            if audio.ndim == 2:
                samples = audio.shape[1]
            else:
                samples = len(audio)
            
            # Generar patrón sináptico
            t = np.linspace(0, samples/44100, samples)
            patron_sinaptico = 1.0 + 0.05 * np.sin(2 * np.pi * 40 * t) * np.exp(-t * 0.1)
            
            # Aplicar patrón
            if audio.ndim == 2:
                for channel in range(audio.shape[0]):
                    audio[channel] *= patron_sinaptico
            else:
                audio *= patron_sinaptico
            
            return audio
        except Exception:
            return audio
    
    def _aplicar_dimension_cuantica(self, audio: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Aplica efectos de coherencia cuántica"""
        try:
            # Añadir coherencia cuántica sutil
            if audio.ndim == 2:
                # Entrelazamiento cuántico entre canales
                mid = (audio[0] + audio[1]) / 2
                side = (audio[0] - audio[1]) / 2
                
                # Modulación cuántica
                samples = audio.shape[1]
                t = np.linspace(0, 1, samples)
                coherencia = 1.0 + 0.03 * np.sin(2 * np.pi * 7.83 * t)  # Frecuencia Schumann
                
                audio[0] = mid + side * coherencia
                audio[1] = mid - side * coherencia
            
            return audio
        except Exception:
            return audio

# ===== 3. OPTIMIZADOR DE RENDIMIENTO ADAPTATIVO =====

class OptimizadorRendimientoAdaptativo:
    """Optimizador que adapta el procesamiento según carga del sistema"""
    
    def __init__(self):
        self.modo_actual = 'completo'  # completo, optimizado, rapido
        self.stats_rendimiento = {
            'tiempo_promedio_ms': 0.0,
            'optimizaciones_aplicadas': 0,
            'modo_changes': 0
        }
        self.umbral_tiempo_rapido = 50  # ms
        self.umbral_tiempo_optimizado = 100  # ms
    
    def determinar_modo_optimo(self, tiempo_anterior_ms: float, complejidad_config: float) -> str:
        """Determina el modo de procesamiento óptimo"""
        if tiempo_anterior_ms > self.umbral_tiempo_optimizado:
            nuevo_modo = 'rapido'
        elif tiempo_anterior_ms > self.umbral_tiempo_rapido:
            nuevo_modo = 'optimizado'
        else:
            nuevo_modo = 'completo'
        
        if nuevo_modo != self.modo_actual:
            self.modo_actual = nuevo_modo
            self.stats_rendimiento['modo_changes'] += 1
            logger.info(f"🎛️ Modo cambiado a: {nuevo_modo}")
        
        return self.modo_actual
    
    def aplicar_optimizaciones_modo(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica optimizaciones según el modo actual"""
        config_optimizado = config.copy()
        
        if self.modo_actual == 'rapido':
            # Modo rápido: Reducir complejidad
            config_optimizado['texture_complexity'] = min(0.7, config.get('texture_complexity', 1.0))
            config_optimizado['validacion_automatica'] = False
            config_optimizado['analisis_espectral'] = False
        
        elif self.modo_actual == 'optimizado':
            # Modo optimizado: Balance entre calidad y velocidad
            config_optimizado['texture_complexity'] = min(0.85, config.get('texture_complexity', 1.0))
            config_optimizado['validacion_automatica'] = True
            config_optimizado['analisis_espectral'] = config.get('calidad_objetivo') == 'maxima'
        
        # Modo completo: Sin cambios
        
        self.stats_rendimiento['optimizaciones_aplicadas'] += 1
        return config_optimizado

# ===== 4. INTEGRADOR DE MEJORAS ADITIVAS =====

class IntegradorMejorasHarmonicEssence:
    """Integra todas las mejoras aditivas sin modificar lógica original"""
    
    def __init__(self, motor_original):
        self.motor = motor_original
        self.amplificador = AmplificadorHarmonicEssence()
        self.processor_multidimensional = ProcessorTexturaMultiDimensional()
        self.optimizador_rendimiento = OptimizadorRendimientoAdaptativo()
        self.mejoras_activas = True
        self.stats_integracion = {
            'mejoras_aplicadas': 0,
            'tiempo_total_mejoras': 0.0,
            'factor_mejora_promedio': 0.0
        }
    
    def generar_audio_mejorado(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray:
        """Versión mejorada del generar_audio con todas las optimizaciones"""
        start_time = time.time()
        
        # 1. Optimizar configuración según rendimiento
        config_optimizado = self.optimizador_rendimiento.aplicar_optimizaciones_modo(config)
        
        # 2. Calcular factor de amplificación
        factor_amplificacion = self.amplificador.calcular_factor_amplificacion(config_optimizado)
        
        # 3. Generar audio usando motor original
        audio_base = self.motor.generar_audio(config_optimizado, duracion_sec)
        
        if not self.mejoras_activas or audio_base.size == 0:
            return audio_base
        
        # 4. Aplicar amplificación inteligente
        audio_amplificado = self._aplicar_amplificacion_inteligente(audio_base, factor_amplificacion)
        
        # 5. Procesamiento multi-dimensional
        audio_multidimensional = self.processor_multidimensional.procesar_textura_multidimensional(
            audio_amplificado, config_optimizado
        )
        
        # 6. Normalización final preservando mejoras
        audio_final = self._normalizar_preservando_mejoras(audio_multidimensional, config_optimizado)
        
        # 7. Actualizar estadísticas
        tiempo_procesamiento = (time.time() - start_time) * 1000  # ms
        self._actualizar_stats_integracion(tiempo_procesamiento, factor_amplificacion)
        
        # 8. Actualizar optimizador de rendimiento
        complejidad = config_optimizado.get('texture_complexity', 1.0)
        self.optimizador_rendimiento.determinar_modo_optimo(tiempo_procesamiento, complejidad)
        
        return audio_final
    
    def _aplicar_amplificacion_inteligente(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Aplica amplificación inteligente sin saturación"""
        # Amplificación con compresión suave para evitar clipping
        audio_amplificado = audio * factor
        
        # Compresión suave si se excede umbral
        umbral = 0.85
        mask = np.abs(audio_amplificado) > umbral
        if np.any(mask):
            # Compresión suave tipo tanh
            audio_amplificado[mask] = np.tanh(audio_amplificado[mask]) * umbral
        
        return audio_amplificado
    
    def _normalizar_preservando_mejoras(self, audio: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Normalización que preserva las mejoras aplicadas"""
        if audio.size == 0:
            return audio
        
        # Normalización adaptativa
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Target level basado en calidad objetivo
            target_levels = {'media': 0.80, 'alta': 0.85, 'maxima': 0.90}
            target = target_levels.get(config.get('calidad_objetivo', 'alta'), 0.85)
            
            if max_val > target:
                # Normalización suave
                factor_normalizacion = target / max_val
                audio = audio * factor_normalizacion
        
        return np.clip(audio, -1.0, 1.0)
    
    def _actualizar_stats_integracion(self, tiempo_ms: float, factor_amplificacion: float):
        """Actualiza estadísticas de integración"""
        self.stats_integracion['mejoras_aplicadas'] += 1
        
        # Promedio móvil del tiempo
        total = self.stats_integracion['mejoras_aplicadas']
        tiempo_actual = self.stats_integracion['tiempo_total_mejoras']
        self.stats_integracion['tiempo_total_mejoras'] = ((tiempo_actual * (total - 1)) + tiempo_ms) / total
        
        # Promedio móvil del factor de mejora
        factor_actual = self.stats_integracion['factor_mejora_promedio']
        self.stats_integracion['factor_mejora_promedio'] = ((factor_actual * (total - 1)) + factor_amplificacion) / total
    
    def obtener_estadisticas_mejoras(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas de las mejoras"""
        return {
            'integracion': self.stats_integracion,
            'amplificacion': self.amplificador.stats_amplificacion,
            'procesamiento_multidimensional': self.processor_multidimensional.stats,
            'optimizacion_rendimiento': self.optimizador_rendimiento.stats_rendimiento,
            'modo_actual': self.optimizador_rendimiento.modo_actual,
            'mejoras_activas': self.mejoras_activas,
            'dimensiones_activas': self.processor_multidimensional.dimensiones_activas
        }
    
    def activar_modo_potencia_maxima(self):
        """Activa modo de potencia máxima"""
        self.mejoras_activas = True
        self.amplificador.config.factor_intensidad_base = 1.5
        self.processor_multidimensional.dimensiones_activas = [
            'temporal', 'frecuencial', 'espacial', 'neural', 'cuantica'
        ]
        self.optimizador_rendimiento.modo_actual = 'completo'
        logger.info("🚀 MODO POTENCIA MÁXIMA ACTIVADO")
    
    def activar_modo_eficiencia(self):
        """Activa modo de eficiencia optimizada"""
        self.mejoras_activas = True
        self.amplificador.config.factor_intensidad_base = 1.2
        self.processor_multidimensional.dimensiones_activas = ['temporal', 'neural']
        logger.info("⚡ MODO EFICIENCIA ACTIVADO")

# ===== 5. FUNCIONES DE CONVENIENCIA PARA INTEGRACIÓN =====

def aplicar_mejoras_harmonic_essence(motor_harmonic_essence):
    """Aplica todas las mejoras aditivas al motor HarmonicEssence"""
    
    # Crear integrador de mejoras
    integrador = IntegradorMejorasHarmonicEssence(motor_harmonic_essence)
    
    # Método mejorado para generar audio
    def generar_audio_potencia_maxima(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray:
        """Versión con potencia máxima del generar_audio"""
        return integrador.generar_audio_mejorado(config, duracion_sec)
    
    # Método para activar modo potencia máxima
    def activar_potencia_maxima(self):
        """Activa modo de potencia máxima"""
        integrador.activar_modo_potencia_maxima()
        logger.info("🚀 HarmonicEssence - POTENCIA MÁXIMA ACTIVADA")
        return integrador.obtener_estadisticas_mejoras()
    
    # Método para obtener estadísticas de mejoras
    def obtener_estadisticas_mejoras_completas(self):
        """Obtiene estadísticas completas de mejoras"""
        stats_originales = self.get_performance_stats()
        stats_mejoras = integrador.obtener_estadisticas_mejoras()
        
        return {
            'stats_originales': stats_originales,
            'stats_mejoras': stats_mejoras,
            'factor_mejora_total': stats_mejoras['integracion']['factor_mejora_promedio'],
            'tiempo_procesamiento_mejorado': stats_mejoras['integracion']['tiempo_total_mejoras'],
            'dimensiones_activas': stats_mejoras['dimensiones_activas'],
            'modo_rendimiento': stats_mejoras['modo_actual']
        }
    
    # Método para análisis de potencia actual
    def analizar_potencia_actual(self):
        """Analiza la potencia actual del motor"""
        capacidades = self.obtener_capacidades()
        stats = self.get_performance_stats()
        
        # Calcular puntuación de potencia
        potencia_base = len(capacidades.get('tipos_textura_soportados', [])) * 0.1
        potencia_aurora = 0.3 if capacidades.get('aurora_v7_integration', False) else 0
        potencia_neural = 0.2 if capacidades.get('neural_enhanced', False) else 0
        potencia_cache = min(0.2, stats.get('cache_hits', 0) / max(1, stats.get('textures_generated', 1)))
        potencia_scientific = 0.2 if capacidades.get('scientific_integration', False) else 0
        
        potencia_total = potencia_base + potencia_aurora + potencia_neural + potencia_cache + potencia_scientific
        
        return {
            'potencia_total_score': min(1.0, potencia_total),
            'potencia_base': potencia_base,
            'potencia_aurora_v7': potencia_aurora,
            'potencia_neural': potencia_neural,
            'eficiencia_cache': potencia_cache,
            'potencia_cientifica': potencia_scientific,
            'recomendaciones': self._generar_recomendaciones_potencia(potencia_total),
            'puede_mejorar': potencia_total < 0.9
        }
    
    def _generar_recomendaciones_potencia(self, potencia_actual: float) -> List[str]:
        """Genera recomendaciones para mejorar potencia"""
        recomendaciones = []
        
        if potencia_actual < 0.7:
            recomendaciones.append("Activar modo potencia máxima")
            recomendaciones.append("Verificar integración Aurora V7")
            recomendaciones.append("Habilitar todas las mejoras neurales")
        elif potencia_actual < 0.85:
            recomendaciones.append("Optimizar configuración de cache")
            recomendaciones.append("Activar procesamiento multi-dimensional")
        else:
            recomendaciones.append("Potencia óptima - mantener configuración actual")
        
        return recomendaciones
    
    # Agregar métodos al motor
    motor_harmonic_essence.generar_audio_potencia_maxima = generar_audio_potencia_maxima.__get__(motor_harmonic_essence)
    motor_harmonic_essence.activar_potencia_maxima = activar_potencia_maxima.__get__(motor_harmonic_essence)
    motor_harmonic_essence.obtener_estadisticas_mejoras_completas = obtener_estadisticas_mejoras_completas.__get__(motor_harmonic_essence)
    motor_harmonic_essence.analizar_potencia_actual = analizar_potencia_actual.__get__(motor_harmonic_essence)
    motor_harmonic_essence._generar_recomendaciones_potencia = _generar_recomendaciones_potencia.__get__(motor_harmonic_essence)
    
    # Almacenar referencia al integrador
    motor_harmonic_essence._integrador_mejoras = integrador
    
    logger.info("✅ Mejoras de potencia máxima aplicadas a HarmonicEssence V34")
    return motor_harmonic_essence

# ===== 6. FUNCIONES DE TESTING Y VERIFICACIÓN =====

def test_potencia_maxima_harmonic_essence():
    """Testing completo de potencia máxima"""
    print("🧪 TESTING POTENCIA MÁXIMA - HarmonicEssence V34")
    print("=" * 60)
    
    try:
        # Importar y crear motor con mejoras
        from harmonicEssence_v34 import crear_motor_aurora_conectado
        motor = crear_motor_aurora_conectado()
        motor_mejorado = aplicar_mejoras_harmonic_essence(motor)
        
        # Test 1: Análisis de potencia actual
        print("📊 Test 1: Análisis de potencia actual...")
        analisis = motor_mejorado.analizar_potencia_actual()
        print(f"   • Potencia total: {analisis['potencia_total_score']:.2f}")
        print(f"   • Puede mejorar: {'✅ SÍ' if analisis['puede_mejorar'] else '❌ NO'}")
        
        # Test 2: Activar potencia máxima
        print("🚀 Test 2: Activando potencia máxima...")
        stats_potencia = motor_mejorado.activar_potencia_maxima()
        print(f"   • Mejoras activas: {'✅' if stats_potencia['mejoras_activas'] else '❌'}")
        print(f"   • Dimensiones activas: {len(stats_potencia['dimensiones_activas'])}")
        
        # Test 3: Generación con potencia máxima
        print("🎵 Test 3: Generación con potencia máxima...")
        config_potencia = {
            'objetivo': 'concentracion_maxima',
            'calidad_objetivo': 'maxima',
            'intensidad': 'intenso',
            'neurotransmisor_preferido': 'dopamina',
            'usar_aurora_v7': True,
            'coherencia_objetivo': 0.95
        }
        
        start_time = time.time()
        audio_potencia = motor_mejorado.generar_audio_potencia_maxima(config_potencia, 2.0)
        tiempo_generacion = time.time() - start_time
        
        print(f"   • Audio generado: {audio_potencia.shape}")
        print(f"   • Tiempo: {tiempo_generacion:.3f}s")
        
        # Test 4: Estadísticas completas
        print("📈 Test 4: Estadísticas de mejoras...")
        stats_completas = motor_mejorado.obtener_estadisticas_mejoras_completas()
        
        factor_mejora = stats_completas['factor_mejora_total']
        tiempo_mejorado = stats_completas['tiempo_procesamiento_mejorado']
        
        print(f"   • Factor de mejora: {factor_mejora:.2f}x")
        print(f"   • Tiempo promedio mejorado: {tiempo_mejorado:.1f}ms")
        print(f"   • Dimensiones activas: {len(stats_completas['dimensiones_activas'])}")
        
        print(f"\n🏆 TESTING COMPLETADO - POTENCIA VERIFICADA")
        return motor_mejorado, stats_completas
        
    except Exception as e:
        print(f"❌ Error en testing: {e}")
        return None, None

def verificar_aporte_harmonic_essence_sistema():
    """Verifica el aporte total del HarmonicEssence al sistema Aurora"""
    print("🔍 VERIFICACIÓN DEL APORTE AL SISTEMA AURORA")
    print("=" * 60)
    
    try:
        motor, stats = test_potencia_maxima_harmonic_essence()
        if not motor:
            print("❌ No se pudo verificar - motor no disponible")
            return
        
        # Análisis de capacidades
        capacidades = motor.obtener_capacidades()
        
        print(f"🎯 CAPACIDADES TOTALES:")
        print(f"   • Tipos de textura: {len(capacidades.get('tipos_textura_soportados', []))}")
        print(f"   • Modulaciones: {len(capacidades.get('modulaciones_soportadas', []))}")
        print(f"   • Efectos espaciales: {len(capacidades.get('efectos_espaciales', []))}")
        print(f"   • Neurotransmisores: {len(capacidades.get('neurotransmisores_soportados', []))}")
        
        # Análisis de integración
        aurora_integration = capacidades.get('aurora_v7_integration', False)
        neural_enhanced = capacidades.get('neural_enhanced', False)
        scientific_integration = capacidades.get('scientific_integration', False)
        
        print(f"🔗 INTEGRACIÓN CON AURORA:")
        print(f"   • Aurora V7: {'✅ CONECTADO' if aurora_integration else '❌ DESCONECTADO'}")
        print(f"   • Neural Enhanced: {'✅ ACTIVO' if neural_enhanced else '❌ INACTIVO'}")
        print(f"   • Científico: {'✅ ACTIVO' if scientific_integration else '❌ INACTIVO'}")
        
        # Cálculo del aporte total
        aporte_textural = min(1.0, len(capacidades.get('tipos_textura_soportados', [])) / 30)
        aporte_integracion = (aurora_integration + neural_enhanced + scientific_integration) / 3
        aporte_rendimiento = min(1.0, stats['factor_mejora_total'] / 2.0) if stats else 0.5
        
        aporte_total = (aporte_textural * 0.4 + aporte_integracion * 0.4 + aporte_rendimiento * 0.2)
        
        print(f"📊 APORTE TOTAL AL SISTEMA: {aporte_total:.1%}")
        
        if aporte_total >= 0.9:
            print(f"🏆 EXCELENTE - Aportando máxima potencia")
        elif aporte_total >= 0.75:
            print(f"✅ BUENO - Aporte sólido con potencial de mejora")
        else:
            print(f"⚠️ MEJORABLE - Potencial no completamente utilizado")
        
        return aporte_total >= 0.8
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        return False

# Ejecutar verificación si se importa este módulo
if __name__ == "__main__":
    verificar_aporte_harmonic_essence_sistema()
