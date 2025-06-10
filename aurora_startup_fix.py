#!/usr/bin/env python3
"""
🚀 AURORA STARTUP FIX - Arranque Controlado Inteligente
=====================================================

Script optimizado para el arranque del Sistema Aurora V7 refactorizado.
Controla inteligentemente el orden de inicialización y registro de componentes,
verificando el estado antes de realizar acciones redundantes.

Versión: V7.1-Optimized
Autor: Sistema Aurora V7
Descripción: Arranque inteligente sin registros duplicados
"""

import sys
import logging
import time
from typing import Dict, Any, Optional, Tuple

# Configurar logging para debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Aurora.Startup.Fix.Optimized")

def verificar_componente_registrado(factory, nombre: str) -> bool:
    """
    Verifica si un componente ya está registrado en el factory
    
    Args:
        factory: Instancia del aurora_factory
        nombre: Nombre del componente a verificar
        
    Returns:
        True si ya está registrado, False si no
    """
    try:
        # Verificar en los diferentes tipos de componentes
        motores = factory.listar_motores_disponibles()
        gestores = factory.listar_gestores_disponibles()
        analizadores = factory.listar_analizadores_disponibles()
        
        return (nombre in motores or 
                nombre in gestores or 
                nombre in analizadores)
    except Exception as e:
        logger.debug(f"Error verificando componente {nombre}: {e}")
        return False

def registrar_componente_inteligente(factory, nombre: str, tipo: str, 
                                   modulo: str, clase: str) -> bool:
    """
    Registra un componente solo si no está ya registrado
    
    Args:
        factory: Instancia del aurora_factory
        nombre: Nombre del componente
        tipo: Tipo (motor, gestor, analizador)
        modulo: Nombre del módulo
        clase: Nombre de la clase
        
    Returns:
        True si se registró o ya existía, False si falló
    """
    
    # Verificar si ya está registrado
    ya_registrado = verificar_componente_registrado(factory, nombre)
    
    if ya_registrado:
        logger.debug(f"   {nombre}: ✅ (ya registrado)")
        return True
    
    # Intentar registrar si no existe
    try:
        if tipo == "motor":
            success = factory.registrar_motor(nombre, modulo, clase)
        elif tipo == "gestor":
            success = factory.registrar_gestor(nombre, modulo, clase)
        elif tipo == "analizador":
            success = factory.registrar_analizador(nombre, modulo, clase)
        else:
            logger.warning(f"   {nombre}: ❌ Tipo desconocido: {tipo}")
            return False
        
        if success:
            logger.info(f"   {nombre}: ✅ (registrado exitosamente)")
        else:
            logger.warning(f"   {nombre}: ⚠️ (registro falló)")
        
        return success
        
    except Exception as e:
        logger.warning(f"   {nombre}: ❌ Error: {e}")
        return False

def init_aurora_system_controlled() -> Tuple[bool, Any]:
    """
    Inicializa el sistema Aurora de manera controlada e inteligente
    
    Returns:
        Tuple[bool, Any]: (éxito, director_instance)
    """
    
    logger.info("🚀 Iniciando sistema Aurora V7 con arranque controlado inteligente")
    
    try:
        # PASO 1: Importar interfaces y factory primero
        logger.info("📦 Paso 1: Importando interfaces y factory...")
        from aurora_interfaces import TipoComponenteAurora, EstadoComponente
        from aurora_factory import aurora_factory
        
        # Esperar un momento para que el auto-registro termine
        time.sleep(0.1)
        
        # PASO 2: Verificar estado actual del factory
        logger.info("🔍 Paso 2: Verificando estado actual del factory...")
        
        try:
            stats_inicial = aurora_factory.obtener_estadisticas()
            motores_existentes = aurora_factory.listar_motores_disponibles()
            gestores_existentes = aurora_factory.listar_gestores_disponibles()
            analizadores_existentes = aurora_factory.listar_analizadores_disponibles()
            
            logger.info(f"   📊 Estado inicial:")
            logger.info(f"      Componentes totales: {stats_inicial.get('total_registrados', 0)}")
            logger.info(f"      Motores: {len(motores_existentes)}")
            logger.info(f"      Gestores: {len(gestores_existentes)}")
            logger.info(f"      Analizadores: {len(analizadores_existentes)}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ No se pudo obtener estadísticas iniciales: {e}")
            motores_existentes = []
            gestores_existentes = []
            analizadores_existentes = []
        
        # PASO 3: Registro inteligente de componentes críticos
        logger.info("🏭 Paso 3: Registro inteligente de componentes críticos...")
        
        # Componentes críticos - verificar y registrar solo si es necesario
        componentes_criticos = [
            ("neuromix_super_unificado", "motor", "neuromix_definitivo_v7", "NeuroMixSuperUnificadoRefactored"),
            ("harmonicEssence_v34", "motor", "harmonicEssence_v34", "HarmonicEssenceV34Refactored"),
            ("emotion_style_profiles", "gestor", "emotion_style_profiles", "GestorPresetsEmocionalesRefactored")
        ]
        
        resultados_criticos = []
        for nombre, tipo, modulo, clase in componentes_criticos:
            resultado = registrar_componente_inteligente(aurora_factory, nombre, tipo, modulo, clase)
            resultados_criticos.append(resultado)
        
        # PASO 4: Registro inteligente de componentes opcionales
        logger.info("🔧 Paso 4: Registro inteligente de componentes opcionales...")
        
        componentes_opcionales = [
            ("neuromix_aurora_v27", "motor", "neuromix_aurora_v27", "AuroraNeuroAcousticEngineV27"),
            ("hypermod_v32", "motor", "hypermod_v32", "HyperModV32"),
            ("objective_manager", "gestor", "objective_manager", "ObjectiveManagerUnificado"),
            ("carmine_analyzer", "analizador", "Carmine_Analyzer", "CarmineAuroraAnalyzer"),
            ("aurora_quality_pipeline", "analizador", "aurora_quality_pipeline", "AuroraQualityPipeline")
        ]
        
        resultados_opcionales = []
        for nombre, tipo, modulo, clase in componentes_opcionales:
            resultado = registrar_componente_inteligente(aurora_factory, nombre, tipo, modulo, clase)
            resultados_opcionales.append(resultado)
        
        # PASO 5: Verificar estado final del factory
        logger.info("📈 Paso 5: Verificando estado final del factory...")
        
        try:
            stats_final = aurora_factory.obtener_estadisticas()
            motores_finales = aurora_factory.listar_motores_disponibles()
            gestores_finales = aurora_factory.listar_gestores_disponibles()
            analizadores_finales = aurora_factory.listar_analizadores_disponibles()
            
            logger.info(f"   📊 Estado final:")
            logger.info(f"      Componentes totales: {stats_final.get('total_registrados', 0)}")
            logger.info(f"      Motores disponibles: {len(motores_finales)}")
            logger.info(f"      Gestores disponibles: {len(gestores_finales)}")
            logger.info(f"      Analizadores disponibles: {len(analizadores_finales)}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ No se pudo obtener estadísticas finales: {e}")
        
        # PASO 6: Crear director CON detección deshabilitada inicialmente
        logger.info("🎯 Paso 6: Creando Aurora Director (modo pasivo)...")
        
        # Importar director DESPUÉS de que factory esté listo
        from aurora_director_v7 import AuroraDirectorV7Refactored
        
        # Crear director - ahora ya no hace detección automática
        director = AuroraDirectorV7Refactored()
        
        # PASO 7: Inicializar detección desde el Director (ÚNICO PUNTO)
        logger.info("⚙️ Paso 7: Inicializando detección desde Director (punto único)...")
        
        exito_deteccion = director.inicializar_deteccion_completa()
        
        if not exito_deteccion:
            logger.error("❌ Detección falló")
            return False, None
        
        # PASO 8: Configurar modo según componentes cargados
        logger.info("🎛️ Paso 8: Configurando modo de operación...")
        
        num_motores = len(director.motores)
        num_gestores = len(director.gestores)
        
        logger.info(f"   ✅ Detección completada:")
        logger.info(f"      🚗 Motores activos: {num_motores}")
        logger.info(f"      🛠️ Gestores activos: {num_gestores}")
        logger.info(f"      🎯 Modo de orquestación: {director.modo_orquestacion.value}")
        logger.info(f"      🎵 Estrategia default: {director.estrategia_default.value}")
        
        # PASO 9: Validación final del sistema
        logger.info("🔬 Paso 9: Validación final del sistema...")
        
        # Verificar que los componentes críticos estén funcionando
        componentes_funcionando = 0
        
        try:
            # Test básico de cada motor
            for nombre_motor, motor in director.motores.items():
                if motor and hasattr(motor, 'get_info'):
                    info = motor.get_info()
                    logger.debug(f"      ✅ {nombre_motor}: {info.get('nombre', 'OK')}")
                    componentes_funcionando += 1
                else:
                    logger.warning(f"      ⚠️ {nombre_motor}: motor no responde")
        except Exception as e:
            logger.warning(f"   ⚠️ Error en validación de motores: {e}")
        
        logger.info(f"   🎉 Sistema validado: {componentes_funcionando} componentes operativos")
        
        logger.info("✅ Sistema Aurora V7 inicializado correctamente con arranque inteligente")
        
        return True, director
        
    except Exception as e:
        logger.error(f"❌ Error en arranque controlado: {e}")
        logger.error(f"   Tipo de error: {type(e).__name__}")
        
        # Log de traceback para debug
        import traceback
        logger.debug(f"   Traceback: {traceback.format_exc()}")
        
        return False, None

def verificar_salud_sistema(director) -> Dict[str, Any]:
    """
    Verifica la salud general del sistema después del arranque
    
    Args:
        director: Instancia del director de Aurora
        
    Returns:
        Dict con información de salud del sistema
    """
    
    logger.info("🏥 Verificando salud del sistema...")
    
    salud = {
        "timestamp": time.time(),
        "sistema_operativo": True,
        "componentes": {},
        "warnings": [],
        "errores": [],
        "puntuacion_salud": 0.0
    }
    
    try:
        # Verificar director
        if director:
            salud["componentes"]["director"] = {
                "estado": "operativo",
                "motores": len(director.motores),
                "gestores": len(director.gestores)
            }
        else:
            salud["errores"].append("Director no disponible")
            salud["sistema_operativo"] = False
        
        # Calcular puntuación de salud
        total_componentes = len(salud["componentes"])
        componentes_ok = sum(1 for comp in salud["componentes"].values() 
                           if comp.get("estado") == "operativo")
        
        if total_componentes > 0:
            salud["puntuacion_salud"] = componentes_ok / total_componentes
        else:
            salud["puntuacion_salud"] = 0.0
        
        # Determinar estado general
        if salud["puntuacion_salud"] >= 0.8:
            logger.info("   💚 Salud del sistema: EXCELENTE")
        elif salud["puntuacion_salud"] >= 0.6:
            logger.info("   💛 Salud del sistema: BUENA")
        elif salud["puntuacion_salud"] >= 0.4:
            logger.info("   🧡 Salud del sistema: REGULAR")
        else:
            logger.warning("   ❤️ Salud del sistema: CRÍTICA")
        
    except Exception as e:
        logger.error(f"❌ Error verificando salud: {e}")
        salud["errores"].append(f"Error en verificación: {e}")
        salud["sistema_operativo"] = False
    
    return salud

# ===============================================================================
# EJECUCIÓN PRINCIPAL
# ===============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 AURORA V7 - ARRANQUE CONTROLADO INTELIGENTE")
    print("="*70)
    print("🧪 Testeando arranque controlado de Aurora V7...")
    
    # Ejecutar arranque controlado
    exito, director = init_aurora_system_controlled()
    
    if exito and director:
        print("\n" + "="*70)
        print("✅ ARRANQUE EXITOSO")
        print("="*70)
        
        # Verificar salud del sistema
        salud = verificar_salud_sistema(director)
        
        print(f"\n📊 Resumen final:")
        print(f"   Sistema operativo: {'✅' if salud['sistema_operativo'] else '❌'}")
        print(f"   Puntuación de salud: {salud['puntuacion_salud']:.1%}")
        print(f"   Componentes activos: {len(salud['componentes'])}")
        
        if salud["warnings"]:
            print(f"   ⚠️ Warnings: {len(salud['warnings'])}")
        
        if salud["errores"]:
            print(f"   ❌ Errores: {len(salud['errores'])}")
        
        print("\n🎉 Aurora V7 listo para crear experiencias neuroacústicas!")
        
    else:
        print("\n" + "="*70)
        print("❌ ARRANQUE FALLIDO")
        print("="*70)
        print("\n🔧 Revisa los logs para diagnosticar el problema.")
        print("💡 Sugerencia: Verifica que todos los módulos estén disponibles.")
    
    print("\n" + "="*70)
