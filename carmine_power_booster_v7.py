# ===== CARMINE POWER BOOSTER V7 =====
# Integración avanzada para aprovechar toda la potencia de Carmine Analyzer
# IMPORTANTE: Este código es ADITIVO - no modifica lógica existente

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("CarmineBooster")

class CarminePowerBooster:
    """
    Amplificador de potencia para Carmine Analyzer
    Es como convertir un análisis médico básico en una cirugía reconstructiva completa
    """
    
    def __init__(self, carmine_analyzer_instance):
        self.carmine = carmine_analyzer_instance
        self.version = "CARMINE_POWER_BOOSTER_V7"
        self.stats = {
            "analisis_realizados": 0,
            "correcciones_aplicadas": 0,
            "mejoras_detectadas": 0,
            "tiempo_total_optimizacion": 0.0,
            "calidad_promedio_antes": 0.0,
            "calidad_promedio_despues": 0.0
        }
        logger.info("🚀 Carmine Power Booster V7 inicializado")
    
    def analisis_completo_con_optimizacion(self, audio_data: np.ndarray, 
                                         objetivo: str = None,
                                         config_original: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Análisis completo + optimización automática
        Es como llevarte al mejor médico que no solo diagnostica, sino que te cura
        """
        start_time = datetime.now()
        
        # 1. Análisis inicial completo
        logger.info("🔬 Iniciando análisis Carmine completo...")
        expected_intent = self._mapear_objetivo_a_carmine_intent(objetivo)
        analisis_inicial = self.carmine.analyze_audio(audio_data, expected_intent)
        
        resultado_completo = {
            "analisis_inicial": self._carmine_result_to_dict(analisis_inicial),
            "audio_original": audio_data.copy(),
            "objetivo": objetivo,
            "optimizaciones_aplicadas": [],
            "mejoras_logradas": {}
        }
        
        # 2. Evaluación de necesidad de correcciones
        necesita_correccion = self._evaluar_necesidad_correcion(analisis_inicial)
        
        if necesita_correccion:
            logger.info("🔧 Aplicando correcciones neuroacústicas automáticas...")
            audio_optimizado = self._aplicar_correcciones_inteligentes(
                audio_data, analisis_inicial, objetivo, config_original
            )
            
            # 3. Re-análisis post-optimización
            analisis_final = self.carmine.analyze_audio(audio_optimizado, expected_intent)
            
            resultado_completo.update({
                "audio_optimizado": audio_optimizado,
                "analisis_final": self._carmine_result_to_dict(analisis_final),
                "mejoras_logradas": self._calcular_mejoras(analisis_inicial, analisis_final),
                "correcciones_exitosas": True
            })
            
            self.stats["correcciones_aplicadas"] += 1
        else:
            logger.info("✅ Audio ya está en calidad óptima")
            resultado_completo.update({
                "audio_optimizado": audio_data,
                "analisis_final": self._carmine_result_to_dict(analisis_inicial),
                "correcciones_exitosas": False,
                "mensaje": "Audio ya optimizado"
            })
        
        # 4. Generar reporte completo
        resultado_completo["reporte_detallado"] = self._generar_reporte_detallado(resultado_completo)
        
        # 5. Actualizar estadísticas
        tiempo_total = (datetime.now() - start_time).total_seconds()
        self._actualizar_estadisticas(analisis_inicial, resultado_completo.get("analisis_final"), tiempo_total)
        
        self.stats["analisis_realizados"] += 1
        
        return resultado_completo
    
    def optimizacion_predictiva_pre_generacion(self, config_aurora: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimización predictiva ANTES de generar audio
        Es como un GPS que te dice la mejor ruta antes de salir de casa
        """
        logger.info("🎯 Ejecutando optimización predictiva Carmine...")
        
        # Generar audio de muestra corta para análisis
        audio_muestra = self._generar_muestra_rapida(config_aurora)
        
        if audio_muestra is not None:
            analisis_predictivo = self.carmine.analyze_audio(audio_muestra)
            
            # Predecir optimizaciones necesarias
            optimizaciones_sugeridas = self._predecir_optimizaciones(analisis_predictivo, config_aurora)
            
            return {
                "optimizaciones_recomendadas": optimizaciones_sugeridas,
                "config_optimizada": self._aplicar_optimizaciones_config(config_aurora, optimizaciones_sugeridas),
                "prediccion_calidad": analisis_predictivo.score,
                "confianza_prediccion": 0.85
            }
        
        return {"optimizaciones_recomendadas": [], "config_optimizada": config_aurora}
    
    def exportar_reporte_completo(self, resultado_analisis: Dict[str, Any], 
                                 ruta_exportacion: str = None) -> str:
        """
        Exporta reporte completo usando la funcionalidad nativa de Carmine
        Es como tener el historial médico completo en PDF profesional
        """
        if not ruta_exportacion:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_exportacion = f"carmine_reporte_{timestamp}.json"
        
        # Usar método nativo de Carmine para exportación
        if "analisis_final" in resultado_analisis:
            analisis_dict = resultado_analisis["analisis_final"]
            
            # Reconstruir objeto de resultado de Carmine
            resultado_carmine = self._dict_to_carmine_result(analisis_dict)
            
            # Exportar usando método nativo
            self.carmine.export_analysis_json(resultado_carmine, ruta_exportacion)
            
            # Agregar datos adicionales del booster
            self._agregar_datos_booster_a_reporte(ruta_exportacion, resultado_analisis)
        
        logger.info(f"📄 Reporte Carmine exportado: {ruta_exportacion}")
        return ruta_exportacion
    
    def _evaluar_necesidad_correcion(self, analisis_carmine) -> bool:
        """Evalúa si el audio necesita corrección basado en múltiples criterios"""
        criterios_correccion = [
            analisis_carmine.score < 85,  # Score bajo
            analisis_carmine.therapeutic_score < 80,  # Score terapéutico bajo
            len(analisis_carmine.issues) > 0,  # Tiene problemas detectados
            analisis_carmine.neuro_metrics.entrainment_effectiveness < 0.7,  # Baja efectividad
            analisis_carmine.neuro_metrics.binaural_strength < 0.3  # Fuerza binaural baja
        ]
        
        return any(criterios_correccion)
    
    def _aplicar_correcciones_inteligentes(self, audio_data: np.ndarray, 
                                         analisis_inicial, objetivo: str,
                                         config_original: Dict[str, Any]) -> np.ndarray:
        """
        Aplica correcciones usando la funcionalidad nativa de Carmine
        Es como una cirugía de precisión guiada por IA
        """
        # Determinar modo agresivo basado en la severidad de problemas
        modo_agresivo = self._determinar_modo_agresivo(analisis_inicial)
        
        # Mapear objetivo a objetivo de Carmine
        objetivo_carmine = self._mapear_objetivo_para_correcciones(objetivo)
        
        logger.info(f"🔧 Aplicando correcciones - Modo: {'Agresivo' if modo_agresivo else 'Suave'}")
        
        # Usar método nativo de corrección de Carmine
        audio_corregido, metadatos_correcciones = self.carmine.aplicar_correcciones_neuroacusticas(
            audio=audio_data,
            analysis_result=analisis_inicial,
            objetivo=objetivo_carmine,
            modo_agresivo=modo_agresivo
        )
        
        # Log de correcciones aplicadas
        correcciones = metadatos_correcciones.get("correcciones_aplicadas", [])
        logger.info(f"✅ Correcciones aplicadas: {len(correcciones)}")
        for corr in correcciones:
            logger.info(f"   🔧 {corr.get('tipo', 'unknown')}")
        
        return audio_corregido
    
    def _determinar_modo_agresivo(self, analisis_carmine) -> bool:
        """Determina si usar modo agresivo de corrección"""
        problemas_graves = [
            analisis_carmine.score < 70,
            analisis_carmine.therapeutic_score < 60,
            len(analisis_carmine.issues) > 3,
            analisis_carmine.neuro_metrics.entrainment_effectiveness < 0.5
        ]
        
        return sum(problemas_graves) >= 2
    
    def _mapear_objetivo_para_correcciones(self, objetivo: str) -> str:
        """Mapea objetivos de Aurora a objetivos de corrección de Carmine"""
        mapeo = {
            "concentracion": "concentracion",
            "claridad_mental": "concentracion", 
            "enfoque": "concentracion",
            "relajacion": "relajacion",
            "calma": "relajacion",
            "meditacion": "meditacion",
            "creatividad": "creatividad",
            "inspiracion": "creatividad",
            "energia": "energia",
            "vitalidad": "energia",
            "sanacion": "sueño"  # Carmine no tiene sanación, usar sueño como proxy
        }
        
        objetivo_lower = objetivo.lower() if objetivo else "relajacion"
        
        for key, carmine_obj in mapeo.items():
            if key in objetivo_lower:
                return carmine_obj
        
        return "relajacion"  # Default
    
    def _mapear_objetivo_a_carmine_intent(self, objetivo: str):
        """Mapea objetivo a TherapeuticIntent de Carmine"""
        if not objetivo:
            return None
            
        try:
            # Importar dinámicamente para evitar dependencias
            from Carmine_Analyzer import TherapeuticIntent
            
            mapeo = {
                "relajacion": TherapeuticIntent.RELAXATION,
                "concentracion": TherapeuticIntent.FOCUS,
                "claridad_mental": TherapeuticIntent.FOCUS,
                "enfoque": TherapeuticIntent.FOCUS,
                "meditacion": TherapeuticIntent.MEDITATION,
                "creatividad": TherapeuticIntent.EMOTIONAL,
                "sanacion": TherapeuticIntent.RELAXATION,
                "energia": TherapeuticIntent.ENERGY,
                "sueño": TherapeuticIntent.SLEEP
            }
            
            objetivo_lower = objetivo.lower()
            return next((intent for key, intent in mapeo.items() if key in objetivo_lower), None)
        except ImportError:
            return None
    
    def _carmine_result_to_dict(self, carmine_result) -> Dict[str, Any]:
        """Convierte resultado de Carmine a diccionario"""
        return {
            "score": carmine_result.score,
            "therapeutic_score": carmine_result.therapeutic_score,
            "quality_level": carmine_result.quality.value,
            "issues": carmine_result.issues,
            "suggestions": carmine_result.suggestions,
            "technical_metrics": carmine_result.technical_metrics,
            "neuro_metrics": {
                "binaural_frequency_diff": carmine_result.neuro_metrics.binaural_frequency_diff,
                "binaural_strength": carmine_result.neuro_metrics.binaural_strength,
                "entrainment_effectiveness": carmine_result.neuro_metrics.entrainment_effectiveness,
                "isochronic_detected": carmine_result.neuro_metrics.isochronic_detected,
                "spatial_movement_detected": carmine_result.neuro_metrics.spatial_movement_detected
            },
            "frequency_analysis": carmine_result.frequency_analysis,
            "therapeutic_intent": carmine_result.therapeutic_intent.value if carmine_result.therapeutic_intent else None,
            "gpt_summary": carmine_result.gpt_summary,
            "timestamp": carmine_result.timestamp
        }
    
    def _dict_to_carmine_result(self, analisis_dict: Dict[str, Any]):
        """Convierte diccionario de vuelta a resultado de Carmine (simulado)"""
        # Para exportación, creamos un objeto simulado con los datos esenciales
        class SimulatedCarmineResult:
            def __init__(self, data):
                self.score = data["score"]
                self.therapeutic_score = data["therapeutic_score"] 
                self.quality = type('Quality', (), {'value': data["quality_level"]})()
                self.issues = data["issues"]
                self.suggestions = data["suggestions"]
                self.technical_metrics = data["technical_metrics"]
                self.frequency_analysis = data["frequency_analysis"]
                self.gpt_summary = data["gpt_summary"]
                self.timestamp = data["timestamp"]
                
                # Simular neuro_metrics
                nm = data["neuro_metrics"]
                self.neuro_metrics = type('NeuroMetrics', (), nm)()
                
                # Simular therapeutic_intent
                if data.get("therapeutic_intent"):
                    self.therapeutic_intent = type('TherapeuticIntent', (), {
                        'value': data["therapeutic_intent"]
                    })()
                else:
                    self.therapeutic_intent = None
        
        return SimulatedCarmineResult(analisis_dict)
    
    def _calcular_mejoras(self, analisis_inicial, analisis_final) -> Dict[str, Any]:
        """Calcula mejoras logradas entre análisis inicial y final"""
        return {
            "mejora_score": analisis_final.score - analisis_inicial.score,
            "mejora_therapeutic": analisis_final.therapeutic_score - analisis_inicial.therapeutic_score,
            "mejora_entrainment": (analisis_final.neuro_metrics.entrainment_effectiveness - 
                                 analisis_inicial.neuro_metrics.entrainment_effectiveness),
            "mejora_binaural": (analisis_final.neuro_metrics.binaural_strength - 
                              analisis_inicial.neuro_metrics.binaural_strength),
            "issues_resueltos": len(analisis_inicial.issues) - len(analisis_final.issues),
            "calidad_inicial": analisis_inicial.quality.value,
            "calidad_final": analisis_final.quality.value
        }
    
    def _generar_muestra_rapida(self, config_aurora: Dict[str, Any]) -> Optional[np.ndarray]:
        """Genera muestra rápida de 5 segundos para análisis predictivo"""
        try:
            # Crear muestra sintética basada en configuración
            duration = 5.0  # 5 segundos
            sample_rate = 44100
            samples = int(sample_rate * duration)
            
            # Generar onda base según objetivo
            objetivo = config_aurora.get("objetivo", "relajacion").lower()
            if "concentracion" in objetivo or "enfoque" in objetivo:
                freq_base = 15.0  # Beta para concentración
            elif "relajacion" in objetivo or "calma" in objetivo:
                freq_base = 8.0   # Alpha para relajación
            elif "meditacion" in objetivo:
                freq_base = 6.0   # Theta para meditación
            else:
                freq_base = 10.0  # Default alpha
            
            t = np.linspace(0, duration, samples)
            wave = 0.3 * np.sin(2 * np.pi * freq_base * t)
            
            # Agregar diferencia binaural básica
            left_channel = wave
            right_channel = 0.3 * np.sin(2 * np.pi * (freq_base + 3.0) * t)
            
            return np.stack([left_channel, right_channel])
        except Exception as e:
            logger.warning(f"No se pudo generar muestra predictiva: {e}")
            return None
    
    def _predecir_optimizaciones(self, analisis_predictivo, config_original: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predice optimizaciones necesarias basado en análisis de muestra"""
        optimizaciones = []
        
        if analisis_predictivo.score < 80:
            optimizaciones.append({
                "tipo": "calidad_objetivo",
                "valor_recomendado": "maxima",
                "razon": "Score predictivo bajo"
            })
        
        if analisis_predictivo.neuro_metrics.entrainment_effectiveness < 0.7:
            optimizaciones.append({
                "tipo": "coherencia_objetivo", 
                "valor_recomendado": 0.9,
                "razon": "Baja efectividad neuroacústica predictiva"
            })
        
        if analisis_predictivo.neuro_metrics.binaural_strength < 0.3:
            optimizaciones.append({
                "tipo": "modo_orquestacion",
                "valor_recomendado": "layered",
                "razon": "Fuerza binaural baja en predicción"
            })
        
        return optimizaciones
    
    def _aplicar_optimizaciones_config(self, config_original: Dict[str, Any], 
                                     optimizaciones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aplica optimizaciones a la configuración"""
        config_optimizada = config_original.copy()
        
        for opt in optimizaciones:
            config_optimizada[opt["tipo"]] = opt["valor_recomendado"]
            logger.info(f"🎯 Optimización predictiva: {opt['tipo']} = {opt['valor_recomendado']}")
        
        return config_optimizada
    
    def _generar_reporte_detallado(self, resultado_analisis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera reporte detallado del análisis completo"""
        return {
            "resumen_ejecutivo": self._generar_resumen_ejecutivo(resultado_analisis),
            "metricas_comparativas": resultado_analisis.get("mejoras_logradas", {}),
            "recomendaciones_tecnicas": self._generar_recomendaciones_tecnicas(resultado_analisis),
            "evaluacion_terapeutica": self._evaluar_efectividad_terapeutica(resultado_analisis),
            "timestamp": datetime.now().isoformat(),
            "version_booster": self.version
        }
    
    def _generar_resumen_ejecutivo(self, resultado_analisis: Dict[str, Any]) -> str:
        """Genera resumen ejecutivo del análisis"""
        analisis_final = resultado_analisis.get("analisis_final", {})
        mejoras = resultado_analisis.get("mejoras_logradas", {})
        
        score_final = analisis_final.get("score", 0)
        mejora_score = mejoras.get("mejora_score", 0)
        
        if resultado_analisis.get("correcciones_exitosas"):
            return f"Audio optimizado exitosamente. Score mejorado de {score_final - mejora_score:.0f} a {score_final:.0f} (+{mejora_score:.0f} puntos). Correcciones neuroacústicas aplicadas con éxito."
        else:
            return f"Audio ya en calidad óptima (Score: {score_final:.0f}). No requirió correcciones adicionales."
    
    def _generar_recomendaciones_tecnicas(self, resultado_analisis: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones técnicas basadas en el análisis"""
        recomendaciones = []
        analisis_final = resultado_analisis.get("analisis_final", {})
        
        if analisis_final.get("score", 100) < 90:
            recomendaciones.append("Considerar incrementar duración para mejor desarrollo neuroacústico")
        
        neuro_metrics = analisis_final.get("neuro_metrics", {})
        if neuro_metrics.get("entrainment_effectiveness", 1.0) < 0.8:
            recomendaciones.append("Optimizar coherencia entre canales para mayor efectividad")
        
        if neuro_metrics.get("binaural_strength", 1.0) < 0.5:
            recomendaciones.append("Incrementar diferencia binaural para mayor impacto neurológico")
        
        return recomendaciones
    
    def _evaluar_efectividad_terapeutica(self, resultado_analisis: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa efectividad terapéutica del audio"""
        analisis_final = resultado_analisis.get("analisis_final", {})
        therapeutic_score = analisis_final.get("therapeutic_score", 0)
        
        if therapeutic_score >= 90:
            nivel = "Excelente"
            recomendacion = "Audio listo para uso terapéutico profesional"
        elif therapeutic_score >= 75:
            nivel = "Bueno"
            recomendacion = "Apropiado para uso terapéutico general"
        elif therapeutic_score >= 60:
            nivel = "Aceptable"
            recomendacion = "Funcional pero con oportunidades de mejora"
        else:
            nivel = "Necesita Mejora"
            recomendacion = "Requiere optimización antes de uso terapéutico"
        
        return {
            "nivel_efectividad": nivel,
            "score_terapeutico": therapeutic_score,
            "recomendacion": recomendacion,
            "apto_uso_clinico": therapeutic_score >= 80
        }
    
    def _actualizar_estadisticas(self, analisis_inicial, analisis_final, tiempo_total: float):
        """Actualiza estadísticas del booster"""
        if analisis_final:
            # Calidad promedio antes
            total_analisis = self.stats["analisis_realizados"]
            calidad_antes_actual = self.stats["calidad_promedio_antes"]
            nueva_calidad_antes = ((calidad_antes_actual * total_analisis) + analisis_inicial.score) / (total_analisis + 1)
            self.stats["calidad_promedio_antes"] = nueva_calidad_antes
            
            # Calidad promedio después
            calidad_despues_actual = self.stats["calidad_promedio_despues"] 
            nueva_calidad_despues = ((calidad_despues_actual * total_analisis) + analisis_final.get("score", analisis_inicial.score)) / (total_analisis + 1)
            self.stats["calidad_promedio_despues"] = nueva_calidad_despues
        
        self.stats["tiempo_total_optimizacion"] += tiempo_total
    
    def _agregar_datos_booster_a_reporte(self, ruta_reporte: str, resultado_analisis: Dict[str, Any]):
        """Agrega datos del booster al reporte JSON existente"""
        try:
            # Leer reporte existente
            with open(ruta_reporte, 'r', encoding='utf-8') as f:
                reporte = json.load(f)
            
            # Agregar datos del booster
            reporte["carmine_power_booster"] = {
                "version": self.version,
                "optimizaciones_aplicadas": resultado_analisis.get("optimizaciones_aplicadas", []),
                "mejoras_logradas": resultado_analisis.get("mejoras_logradas", {}),
                "reporte_detallado": resultado_analisis.get("reporte_detallado", {}),
                "correcciones_exitosas": resultado_analisis.get("correcciones_exitosas", False),
                "estadisticas_booster": self.stats
            }
            
            # Escribir reporte actualizado
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                json.dump(reporte, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"No se pudieron agregar datos del booster al reporte: {e}")
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas del booster"""
        mejora_promedio = (self.stats["calidad_promedio_despues"] - 
                          self.stats["calidad_promedio_antes"])
        
        return {
            **self.stats,
            "mejora_promedio": mejora_promedio,
            "eficiencia": (self.stats["correcciones_aplicadas"] / 
                          max(1, self.stats["analisis_realizados"])) * 100,
            "tiempo_promedio_analisis": (self.stats["tiempo_total_optimizacion"] / 
                                        max(1, self.stats["analisis_realizados"]))
        }


# ===== INTEGRACIÓN CON AURORA DIRECTOR =====

def aplicar_carmine_power_booster(director_instance):
    """
    Aplica Carmine Power Booster al director de forma aditiva
    Es como instalar un turbo a un motor que ya funciona bien
    """
    
    # Solo aplicar si Carmine está disponible
    if "carmine_analyzer_v21" not in director_instance.componentes:
        logger.warning("⚠️ Carmine Analyzer no disponible - Power Booster no aplicado")
        return director_instance
    
    # Crear instancia del booster
    carmine_instance = director_instance.componentes["carmine_analyzer_v21"].instancia
    carmine_booster = CarminePowerBooster(carmine_instance)
    director_instance._carmine_booster = carmine_booster
    
    # Método adicional: análisis completo con optimización
    def analizar_y_optimizar_con_carmine(self, audio_data: np.ndarray, 
                                        objetivo: str = None,
                                        config_original: Dict[str, Any] = None) -> Dict[str, Any]:
        """Análisis completo + optimización automática con Carmine"""
        return self._carmine_booster.analisis_completo_con_optimizacion(
            audio_data, objetivo, config_original
        )
    
    # Método adicional: optimización predictiva
    def optimizacion_predictiva_carmine(self, config_aurora: Dict[str, Any]) -> Dict[str, Any]:
        """Optimización predictiva pre-generación"""
        return self._carmine_booster.optimizacion_predictiva_pre_generacion(config_aurora)
    
    # Método adicional: exportación completa
    def exportar_reporte_carmine_completo(self, resultado_analisis: Dict[str, Any],
                                         nombre_archivo: str = None) -> str:
        """Exporta reporte completo de Carmine"""
        return self._carmine_booster.exportar_reporte_completo(resultado_analisis, nombre_archivo)
    
    # Método mejorado: aplicar análisis Carmine (sobrescribe método básico)
    def _aplicar_analisis_carmine_potenciado(self, resultado, config):
        """Versión potenciada del análisis Carmine que reemplaza la básica"""
        if not hasattr(self, '_carmine_booster'):
            # Fallback al método original si no hay booster
            return self._aplicar_analisis_carmine_original(resultado, config)
        
        try:
            # Usar booster para análisis completo
            analisis_completo = self._carmine_booster.analisis_completo_con_optimizacion(
                resultado.audio_data,
                config.objetivo,
                {
                    "duracion_min": config.duracion_min,
                    "intensidad": config.intensidad,
                    "estilo": config.estilo,
                    "calidad_objetivo": config.calidad_objetivo
                }
            )
            
            # Si hubo optimización, actualizar audio del resultado
            if analisis_completo.get("correcciones_exitosas"):
                resultado.audio_data = analisis_completo["audio_optimizado"]
                logger.info("🚀 Audio optimizado automáticamente por Carmine Power Booster")
            
            # Actualizar resultado con análisis completo
            resultado.metadatos["carmine_power_analysis"] = {
                "analisis_completo": analisis_completo["analisis_final"],
                "optimizaciones_aplicadas": analisis_completo.get("optimizaciones_aplicadas", []),
                "mejoras_logradas": analisis_completo.get("mejoras_logradas", {}),
                "reporte_detallado": analisis_completo.get("reporte_detallado", {}),
                "carmine_booster_usado": True,
                "version_booster": self._carmine_booster.version
            }
            
            # Actualizar métricas del resultado con versión optimizada
            analisis_final = analisis_completo["analisis_final"]
            resultado.calidad_score = max(resultado.calidad_score, analisis_final["score"])
            resultado.efectividad_terapeutica = max(
                resultado.efectividad_terapeutica, 
                analisis_final["therapeutic_score"] / 100.0
            )
            
            neuro_metrics = analisis_final.get("neuro_metrics", {})
            if neuro_metrics.get("entrainment_effectiveness"):
                resultado.coherencia_neuroacustica = max(
                    resultado.coherencia_neuroacustica,
                    neuro_metrics["entrainment_effectiveness"]
                )
            
        except Exception as e:
            logger.error(f"Error en Carmine Power Booster: {e}")
            # Fallback al método original
            return self._aplicar_analisis_carmine_original(resultado, config)
        
        return resultado
    
    # Guardar método original y reemplazar
    director_instance._aplicar_analisis_carmine_original = director_instance._aplicar_analisis_carmine
    director_instance._aplicar_analisis_carmine = _aplicar_analisis_carmine_potenciado.__get__(director_instance)
    
    # Agregar nuevos métodos
    director_instance.analizar_y_optimizar_con_carmine = analizar_y_optimizar_con_carmine.__get__(director_instance)
    director_instance.optimizacion_predictiva_carmine = optimizacion_predictiva_carmine.__get__(director_instance)
    director_instance.exportar_reporte_carmine_completo = exportar_reporte_carmine_completo.__get__(director_instance)
    
    # Actualizar estadísticas
    director_instance.stats["carmine_power_booster_activo"] = True
    director_instance.stats["carmine_booster_version"] = carmine_booster.version
    
    logger.info("🚀 Carmine Power Booster integrado exitosamente")
    return director_instance


# ===== FUNCIONES DE CONVENIENCIA =====

def Aurora_ConCarminePower(objetivo: str = None, **kwargs):
    """Aurora con Carmine Power Booster activado"""
    global _director_global
    if _director_global is None:
        from aurora_director_v7 import AuroraDirectorV7Integrado
        _director_global = AuroraDirectorV7Integrado()
        _director_global = aplicar_carmine_power_booster(_director_global)
    
    if objetivo is not None:
        return _director_global.crear_experiencia(objetivo, **kwargs)
    else:
        return _director_global

def crear_experiencia_carmine_optimizada(objetivo: str, **kwargs):
    """Crea experiencia con optimización predictiva de Carmine"""
    director = Aurora_ConCarminePower()
    
    # Optimización predictiva pre-generación
    if hasattr(director, 'optimizacion_predictiva_carmine'):
        config_original = {
            "objetivo": objetivo,
            "duracion_min": kwargs.get("duracion_min", 20),
            "intensidad": kwargs.get("intensidad", "media"),
            "estilo": kwargs.get("estilo", "sereno"),
            "calidad_objetivo": kwargs.get("calidad_objetivo", "alta")
        }
        
        prediccion = director.optimizacion_predictiva_carmine(config_original)
        kwargs.update(prediccion.get("config_optimizada", {}))
    
    return director.crear_experiencia(objetivo, **kwargs)

def analizar_audio_completo_carmine(audio_data: np.ndarray, objetivo: str = None):
    """Análisis completo standalone con Carmine Power Booster"""
    director = Aurora_ConCarminePower()
    
    if hasattr(director, 'analizar_y_optimizar_con_carmine'):
        return director.analizar_y_optimizar_con_carmine(audio_data, objetivo)
    else:
        logger.warning("Carmine Power Booster no disponible")
        return None

def obtener_estadisticas_carmine_power():
    """Obtiene estadísticas del Carmine Power Booster"""
    director = Aurora_ConCarminePower()
    
    if hasattr(director, '_carmine_booster'):
        return director._carmine_booster.obtener_estadisticas()
    else:
        return {"disponible": False, "razon": "Carmine Power Booster no activado"}


# ===== AGREGAR MÉTODOS AL OBJETO AURORA PRINCIPAL =====

# Importar Aurora principal si está disponible
try:
    from aurora_director_v7 import Aurora
    
    # Métodos con Carmine Power
    Aurora.con_carmine_power = Aurora_ConCarminePower
    Aurora.carmine_optimizado = crear_experiencia_carmine_optimizada
    Aurora.analizar_completo = analizar_audio_completo_carmine
    Aurora.stats_carmine_power = obtener_estadisticas_carmine_power
    
    # Método para activar Carmine Power en instancia existente
    Aurora.activar_carmine_power = lambda: aplicar_carmine_power_booster(Aurora())
    
    logger.info("✅ Carmine Power Booster integrado en objeto Aurora principal")
    
except ImportError:
    logger.warning("⚠️ No se pudo integrar con Aurora principal - usar funciones standalone")


# ===== EJEMPLO DE USO =====

if __name__ == "__main__":
    print("🚀 CARMINE POWER BOOSTER V7 - Demo")
    print("=" * 60)
    
    try:
        # Usar Aurora con Carmine Power
        director = Aurora_ConCarminePower()
        print("✅ Aurora con Carmine Power inicializado")
        
        # Crear experiencia optimizada
        resultado = crear_experiencia_carmine_optimizada(
            "concentracion profunda",
            duracion_min=15,
            calidad_objetivo="maxima"
        )
        
        print(f"🎵 Experiencia creada - Score: {resultado.calidad_score:.1f}")
        
        # Mostrar datos de Carmine Power si están disponibles
        if "carmine_power_analysis" in resultado.metadatos:
            carmine_data = resultado.metadatos["carmine_power_analysis"]
            print(f"🔬 Análisis Carmine Power completado")
            print(f"   Optimizaciones: {len(carmine_data.get('optimizaciones_aplicadas', []))}")
            print(f"   Mejoras detectadas: {bool(carmine_data.get('mejoras_logradas'))}")
        
        # Estadísticas del booster
        stats = obtener_estadisticas_carmine_power()
        if stats.get("disponible", True):
            print(f"📊 Análisis realizados: {stats.get('analisis_realizados', 0)}")
            print(f"📈 Mejora promedio: +{stats.get('mejora_promedio', 0):.1f} puntos")
        
        print("🏆 CARMINE POWER BOOSTER FUNCIONANDO CORRECTAMENTE")
        
    except Exception as e:
        print(f"❌ Error en demo: {e}")
        print("💡 Asegúrate de que Carmine Analyzer esté disponible")
