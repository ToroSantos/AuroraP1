#!/usr/bin/env python3
"""
🚀 AURORA STARTUP FIX - Arranque Controlado
==========================================

Script para solucionar el problema de arranque del Sistema Aurora V7
refactorizado. Controla el orden de inicialización y registro de componentes.

"""

import sys
import logging
import time
from typing import Dict, Any, Optional

# Configurar logging para debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Aurora.Startup.Fix")

def init_aurora_system_controlled():
    """
    Inicializa el sistema Aurora de manera controlada
    
    Returns:
        Tuple[bool, Any]: (éxito, director_instance)
    """
    
    logger.info("🚀 Iniciando sistema Aurora V7 con arranque controlado")
    
    try:
        # PASO 1: Importar interfaces y factory primero
        logger.info("📦 Paso 1: Importando interfaces y factory...")
        from aurora_interfaces import TipoComponenteAurora, EstadoComponente
        from aurora_factory import aurora_factory
        
        # PASO 2: Registrar componentes manualmente en el orden correcto
        logger.info("🏭 Paso 2: Registrando componentes en factory...")
        
        # Registrar NeuroMix Super Unificado
        success_neuromix = aurora_factory.registrar_motor(
            "neuromix_super_unificado",
            "neuromix_definitivo_v7",
            "NeuroMixSuperUnificadoRefactored"
        )
        logger.info(f"   NeuroMix Super: {'✅' if success_neuromix else '❌'}")
        
        # Registrar HarmonicEssence 
        success_harmonic = aurora_factory.registrar_motor(
            "harmonicEssence_v34",
            "harmonicEssence_v34", 
            "HarmonicEssenceV34Refactored"
        )
        logger.info(f"   HarmonicEssence: {'✅' if success_harmonic else '❌'}")
        
        # Registrar Emotion Style Profiles
        success_emotion = aurora_factory.registrar_gestor(
            "emotion_style_profiles",
            "emotion_style_profiles",
            "GestorPresetsEmocionalesRefactored"
        )
        logger.info(f"   Emotion Profiles: {'✅' if success_emotion else '❌'}")
        
        # Registrar otros componentes opcionales
        componentes_opcionales = [
            ("neuromix_aurora_v27", "motor", "neuromix_aurora_v27", "AuroraNeuroAcousticEngineV27"),
            ("hypermod_v32", "motor", "hypermod_v32", "HyperModV32"),
            ("objective_manager", "gestor", "objective_manager", "ObjectiveManagerUnificado"),
            ("carmine_analyzer", "analizador", "Carmine_Analyzer", "CarmineAuroraAnalyzer"),
            ("aurora_quality_pipeline", "analizador", "aurora_quality_pipeline", "AuroraQualityPipeline")
        ]
        
        for nombre, tipo, modulo, clase in componentes_opcionales:
            try:
                if tipo == "motor":
                    success = aurora_factory.registrar_motor(nombre, modulo, clase)
                elif tipo == "gestor":
                    success = aurora_factory.registrar_gestor(nombre, modulo, clase)
                else:  # analizador
                    success = aurora_factory.registrar_analizador(nombre, modulo, clase)
                
                logger.info(f"   {nombre}: {'✅' if success else '❌'}")
            except Exception as e:
                logger.warning(f"   {nombre}: ❌ Error: {e}")
        
        # PASO 3: Verificar factory antes de crear director
        logger.info("🔍 Paso 3: Verificando factory...")
        stats_factory = aurora_factory.obtener_estadisticas()
        logger.info(f"   Componentes registrados: {stats_factory['total_registrados']}")
        logger.info(f"   Motores disponibles: {len(aurora_factory.listar_motores_disponibles())}")
        logger.info(f"   Gestores disponibles: {len(aurora_factory.listar_gestores_disponibles())}")
        
        # PASO 4: Crear director CON detección deshabilitada inicialmente
        logger.info("🎯 Paso 4: Creando Aurora Director (modo pasivo)...")
        
        # Importar director DESPUÉS de que factory esté listo
        from aurora_director_v7 import AuroraDirectorV7Refactored
        
        # Crear director - ahora ya no hace detección automática
        director = AuroraDirectorV7Refactored()
        
        # PASO 5: Inicializar detección desde el Director (ÚNICO PUNTO)
        logger.info("⚙️ Paso 5: Inicializando detección desde Director (punto único)...")
        
        exito_deteccion = director.inicializar_deteccion_completa()
        
        if not exito_deteccion:
            logger.error("❌ Detección falló")
            return False, None
        
        # PASO 6: Configurar modo según componentes cargados
        logger.info("🎛️ Paso 6: Configurando modo de operación...")
        
        num_motores = len(director.motores)
        num_gestores = len(director.gestores)
        
        logger.info(f"   ✅ Detección completada: {num_motores} motores, {num_gestores} gestores")
        logger.info(f"   🎯 Modo: {director.modo_orquestacion.value}")
        logger.info(f"   🎵 Estrategia: {director.estrategia_default.value}")
        
        logger.info("✅ Sistema Aurora V7 inicializado correctamente con detección única")
        
        return True, director
        
    except Exception as e:
        logger.error(f"❌ Error en arranque controlado: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_aurora_startup():
    """Test del arranque controlado"""
    
    print("🧪 Testeando arranque controlado de Aurora V7...")
    
    success, director = init_aurora_system_controlled()
    
    if success and director:
        print("✅ Sistema iniciado correctamente")
        
        # Test estado
        try:
            estado = director.obtener_estado_sistema()
            print(f"✅ Estado obtenido: {estado['director']['modo_orquestacion']}")
        except Exception as e:
            print(f"❌ Error obteniendo estado: {e}")
        
        # Test generación básica
        try:
            from aurora_director_v7 import ConfiguracionAuroraUnificada
            
            config = ConfiguracionAuroraUnificada(
                objetivo="test",
                duracion_min=0.05,  # 3 segundos
                intensidad=0.3
            )
            
            print("🎵 Probando generación de audio...")
            resultado = director.crear_experiencia(config)
            
            print(f"✅ Audio generado: {resultado.exito}")
            print(f"   Estrategia: {resultado.estrategia_usada.value}")
            print(f"   Motores usados: {resultado.motores_utilizados}")
            print(f"   Duración: {resultado.duracion_real:.1f}s")
            
        except Exception as e:
            print(f"❌ Error en test de generación: {e}")
        
        return director
    else:
        print("❌ Sistema NO se pudo inicializar")
        return None

if __name__ == "__main__":
    # Ejecutar test si se ejecuta directamente
    test_aurora_startup()
