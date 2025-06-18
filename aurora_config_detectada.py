#!/usr/bin/env python3
"""
Configuración automática de motores Aurora V7 detectados
Generado automáticamente por el detector
"""

import os
import sys
import importlib.util
from pathlib import Path

# Agregar rutas necesarias
if os.path.exists('aurora_system'):
    sys.path.insert(0, os.path.abspath('aurora_system'))

def verificar_motores_reales_detectados():
    """Configuración automática basada en detección de tu sistema"""
    motores_reales = {}
    errores_importacion = []
    
    print("🔍 VERIFICANDO MOTORES DETECTADOS AUTOMÁTICAMENTE...")
    print("=" * 60)

    # HARMONIC - harmonicEssence_v34
    try:
        # Importar desde: aurora_system/harmonicEssence_v34.py
        spec = importlib.util.spec_from_file_location('harmonicEssence_v34', 'aurora_system/harmonicEssence_v34.py')
        if spec and spec.loader:
            modulo_harmonic = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo_harmonic)
            
            if hasattr(modulo_harmonic, 'HarmonicEssenceV34'):
                clase_harmonic = getattr(modulo_harmonic, 'HarmonicEssenceV34')
                
                # Intentar crear instancia
                try:
                    instancia_harmonic = clase_harmonic()
                    instancia_creada = True
                except:
                    # Algunas clases pueden necesitar parámetros
                    instancia_harmonic = clase_harmonic
                    instancia_creada = False
                
                motores_reales['harmonic'] = {
                    'instancia': instancia_harmonic,
                    'modulo': 'harmonicEssence_v34',
                    'clase': 'HarmonicEssenceV34',
                    'metodos': ['generar_audio'],
                    'archivo': 'aurora_system/harmonicEssence_v34.py',
                    'es_real': True,
                    'instancia_creada': instancia_creada
                }
                print(f"✅ HARMONIC: HarmonicEssenceV34 - MOTOR REAL DETECTADO")
                print(f"   📍 Archivo: aurora_system/harmonicEssence_v34.py")
                print(f"   🔧 Métodos: generar_audio")
            else:
                errores_importacion.append("harmonic: Clase HarmonicEssenceV34 no encontrada")
        else:
            errores_importacion.append("harmonic: No se pudo cargar aurora_system/harmonicEssence_v34.py")
            
    except Exception as e:
        errores_importacion.append("harmonic: {str(e)}")
        print(f"❌ HARMONIC: Error - {e}")

    # NEUROMIX - neuromix_aurora_v27
    try:
        # Importar desde: aurora_system/neuromix_aurora_v27.py
        spec = importlib.util.spec_from_file_location('neuromix_aurora_v27', 'aurora_system/neuromix_aurora_v27.py')
        if spec and spec.loader:
            modulo_neuromix = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo_neuromix)
            
            if hasattr(modulo_neuromix, 'AuroraNeuroAcousticEngineV27'):
                clase_neuromix = getattr(modulo_neuromix, 'AuroraNeuroAcousticEngineV27')
                
                # Intentar crear instancia
                try:
                    instancia_neuromix = clase_neuromix()
                    instancia_creada = True
                except:
                    # Algunas clases pueden necesitar parámetros
                    instancia_neuromix = clase_neuromix
                    instancia_creada = False
                
                motores_reales['neuromix'] = {
                    'instancia': instancia_neuromix,
                    'modulo': 'neuromix_aurora_v27',
                    'clase': 'AuroraNeuroAcousticEngineV27',
                    'metodos': ['generar_audio'],
                    'archivo': 'aurora_system/neuromix_aurora_v27.py',
                    'es_real': True,
                    'instancia_creada': instancia_creada
                }
                print(f"✅ NEUROMIX: AuroraNeuroAcousticEngineV27 - MOTOR REAL DETECTADO")
                print(f"   📍 Archivo: aurora_system/neuromix_aurora_v27.py")
                print(f"   🔧 Métodos: generar_audio")
            else:
                errores_importacion.append("neuromix: Clase AuroraNeuroAcousticEngineV27 no encontrada")
        else:
            errores_importacion.append("neuromix: No se pudo cargar aurora_system/neuromix_aurora_v27.py")
            
    except Exception as e:
        errores_importacion.append("neuromix: {str(e)}")
        print(f"❌ NEUROMIX: Error - {e}")

    # NEUROMIX - neuromix_definitivo_v7
    try:
        # Importar desde: aurora_system/neuromix_definitivo_v7.py
        spec = importlib.util.spec_from_file_location('neuromix_definitivo_v7', 'aurora_system/neuromix_definitivo_v7.py')
        if spec and spec.loader:
            modulo_neuromix = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo_neuromix)
            
            if hasattr(modulo_neuromix, 'AuroraNeuroAcousticEngineV27Connected'):
                clase_neuromix = getattr(modulo_neuromix, 'AuroraNeuroAcousticEngineV27Connected')
                
                # Intentar crear instancia
                try:
                    instancia_neuromix = clase_neuromix()
                    instancia_creada = True
                except:
                    # Algunas clases pueden necesitar parámetros
                    instancia_neuromix = clase_neuromix
                    instancia_creada = False
                
                motores_reales['neuromix'] = {
                    'instancia': instancia_neuromix,
                    'modulo': 'neuromix_definitivo_v7',
                    'clase': 'AuroraNeuroAcousticEngineV27Connected',
                    'metodos': ['generar_audio'],
                    'archivo': 'aurora_system/neuromix_definitivo_v7.py',
                    'es_real': True,
                    'instancia_creada': instancia_creada
                }
                print(f"✅ NEUROMIX: AuroraNeuroAcousticEngineV27Connected - MOTOR REAL DETECTADO")
                print(f"   📍 Archivo: aurora_system/neuromix_definitivo_v7.py")
                print(f"   🔧 Métodos: generar_audio")
            else:
                errores_importacion.append("neuromix: Clase AuroraNeuroAcousticEngineV27Connected no encontrada")
        else:
            errores_importacion.append("neuromix: No se pudo cargar aurora_system/neuromix_definitivo_v7.py")
            
    except Exception as e:
        errores_importacion.append("neuromix: {str(e)}")
        print(f"❌ NEUROMIX: Error - {e}")

    # DIRECTOR - aurora_director_v7
    try:
        # Importar desde: aurora_system/aurora_director_v7.py
        spec = importlib.util.spec_from_file_location('aurora_director_v7', 'aurora_system/aurora_director_v7.py')
        if spec and spec.loader:
            modulo_director = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo_director)
            
            if hasattr(modulo_director, 'AuroraDirectorOptimizado'):
                clase_director = getattr(modulo_director, 'AuroraDirectorOptimizado')
                
                # Intentar crear instancia
                try:
                    instancia_director = clase_director()
                    instancia_creada = True
                except:
                    # Algunas clases pueden necesitar parámetros
                    instancia_director = clase_director
                    instancia_creada = False
                
                motores_reales['director'] = {
                    'instancia': instancia_director,
                    'modulo': 'aurora_director_v7',
                    'clase': 'AuroraDirectorOptimizado',
                    'metodos': ['crear_experiencia'],
                    'archivo': 'aurora_system/aurora_director_v7.py',
                    'es_real': True,
                    'instancia_creada': instancia_creada
                }
                print(f"✅ DIRECTOR: AuroraDirectorOptimizado - MOTOR REAL DETECTADO")
                print(f"   📍 Archivo: aurora_system/aurora_director_v7.py")
                print(f"   🔧 Métodos: crear_experiencia")
            else:
                errores_importacion.append("director: Clase AuroraDirectorOptimizado no encontrada")
        else:
            errores_importacion.append("director: No se pudo cargar aurora_system/aurora_director_v7.py")
            
    except Exception as e:
        errores_importacion.append("director: {str(e)}")
        print(f"❌ DIRECTOR: Error - {e}")

    # DIRECTOR - aurora_interfaces
    try:
        # Importar desde: aurora_system/aurora_interfaces.py
        spec = importlib.util.spec_from_file_location('aurora_interfaces', 'aurora_system/aurora_interfaces.py')
        if spec and spec.loader:
            modulo_director = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo_director)
            
            if hasattr(modulo_director, 'DirectorAuroraInterface'):
                clase_director = getattr(modulo_director, 'DirectorAuroraInterface')
                
                # Intentar crear instancia
                try:
                    instancia_director = clase_director()
                    instancia_creada = True
                except:
                    # Algunas clases pueden necesitar parámetros
                    instancia_director = clase_director
                    instancia_creada = False
                
                motores_reales['director'] = {
                    'instancia': instancia_director,
                    'modulo': 'aurora_interfaces',
                    'clase': 'DirectorAuroraInterface',
                    'metodos': ['crear_experiencia'],
                    'archivo': 'aurora_system/aurora_interfaces.py',
                    'es_real': True,
                    'instancia_creada': instancia_creada
                }
                print(f"✅ DIRECTOR: DirectorAuroraInterface - MOTOR REAL DETECTADO")
                print(f"   📍 Archivo: aurora_system/aurora_interfaces.py")
                print(f"   🔧 Métodos: crear_experiencia")
            else:
                errores_importacion.append("director: Clase DirectorAuroraInterface no encontrada")
        else:
            errores_importacion.append("director: No se pudo cargar aurora_system/aurora_interfaces.py")
            
    except Exception as e:
        errores_importacion.append("director: {str(e)}")
        print(f"❌ DIRECTOR: Error - {e}")

    # HYPERMOD - hypermod_v32
    try:
        # Importar desde: aurora_system/hypermod_v32.py
        spec = importlib.util.spec_from_file_location('hypermod_v32', 'aurora_system/hypermod_v32.py')
        if spec and spec.loader:
            modulo_hypermod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo_hypermod)
            
            if hasattr(modulo_hypermod, 'HyperModAuroraConnected'):
                clase_hypermod = getattr(modulo_hypermod, 'HyperModAuroraConnected')
                
                # Intentar crear instancia
                try:
                    instancia_hypermod = clase_hypermod()
                    instancia_creada = True
                except:
                    # Algunas clases pueden necesitar parámetros
                    instancia_hypermod = clase_hypermod
                    instancia_creada = False
                
                motores_reales['hypermod'] = {
                    'instancia': instancia_hypermod,
                    'modulo': 'hypermod_v32',
                    'clase': 'HyperModAuroraConnected',
                    'metodos': ['generar_audio'],
                    'archivo': 'aurora_system/hypermod_v32.py',
                    'es_real': True,
                    'instancia_creada': instancia_creada
                }
                print(f"✅ HYPERMOD: HyperModAuroraConnected - MOTOR REAL DETECTADO")
                print(f"   📍 Archivo: aurora_system/hypermod_v32.py")
                print(f"   🔧 Métodos: generar_audio")
            else:
                errores_importacion.append("hypermod: Clase HyperModAuroraConnected no encontrada")
        else:
            errores_importacion.append("hypermod: No se pudo cargar aurora_system/hypermod_v32.py")
            
    except Exception as e:
        errores_importacion.append("hypermod: {str(e)}")
        print(f"❌ HYPERMOD: Error - {e}")

    print("=" * 60)
    print(f"📊 RESUMEN FINAL:")
    print(f"   🎵 Motores reales detectados: {len(motores_reales)}")
    print(f"   ❌ Errores de importación: {len(errores_importacion)}")
    
    if len(motores_reales) > 0:
        print(f"\n🎉 SISTEMA LISTO CON {len(motores_reales)} MOTORES REALES")
        for nombre, info in motores_reales.items():
            print(f"   🔥 {nombre.upper()}: {info['clase']} ({info['modulo']})")
    else:
        print("\n💥 CRÍTICO: NO SE DETECTARON MOTORES REALES")
        for error in errores_importacion:
            print(f"   • {error}")
    
    return motores_reales, errores_importacion

# Variables globales para uso en Flask
MOTORES_REALES, ERRORES_IMPORTACION = verificar_motores_reales_detectados()

# Función auxiliar para verificar disponibilidad
def tiene_motores_reales():
    return len(MOTORES_REALES) > 0

def obtener_motor(tipo_motor):
    """Obtiene un motor específico por tipo"""
    if tipo_motor in MOTORES_REALES:
        return MOTORES_REALES[tipo_motor]['instancia']
    return None

def listar_motores_disponibles():
    """Lista todos los motores disponibles"""
    return list(MOTORES_REALES.keys())

def obtener_info_motor(tipo_motor):
    """Obtiene información detallada de un motor"""
    if tipo_motor in MOTORES_REALES:
        return MOTORES_REALES[tipo_motor]
    return None
