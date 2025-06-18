#!/usr/bin/env python3
"""
🔍 DETECTOR DE ARCHIVOS AURORA REALES
=====================================

Este script detecta automáticamente qué archivos de Aurora V7 
están realmente disponibles en tu sistema.
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path

def detectar_archivos_aurora():
    """Detecta todos los archivos Aurora disponibles"""
    print("🔍 DETECTANDO ARCHIVOS AURORA REALES...")
    print("=" * 60)
    
    # Rutas corregidas - enfoque en aurora_system
    rutas_busqueda = [
        "aurora_system",  # Carpeta principal de motores
        ".",              # Directorio actual (para bridge, flask, etc.)
    ]
    
    # Agregar rutas al PYTHONPATH
    for ruta in rutas_busqueda:
        if os.path.exists(ruta):
            ruta_absoluta = os.path.abspath(ruta)
            if ruta_absoluta not in sys.path:
                sys.path.insert(0, ruta_absoluta)
    
    print("📁 Rutas de búsqueda configuradas:")
    for ruta in rutas_busqueda:
        if os.path.exists(ruta):
            archivos_py = list(Path(ruta).glob("*.py"))
            print(f"   ✅ {os.path.abspath(ruta)} ({len(archivos_py)} archivos .py)")
        else:
            print(f"   ❌ {ruta} (no existe)")
    
    print("\n🔍 BUSCANDO ARCHIVOS PYTHON DE AURORA...")
    print("-" * 60)
    
    archivos_encontrados = []
    
    # Buscar en ambas rutas
    for ruta in rutas_busqueda:
        if not os.path.exists(ruta):
            continue
            
        print(f"\n📂 Explorando: {ruta}/")
        
        for archivo in Path(ruta).glob("*.py"):
            nombre = archivo.stem
            
            # Filtrar archivos relacionados con Aurora
            if any(palabra in nombre.lower() for palabra in [
                'aurora', 'neuromix', 'hypermod', 'harmonic', 'carmine', 'director'
            ]):
                try:
                    print(f"🔍 Analizando: {archivo.name}")
                    
                    # Leer el archivo y buscar clases sin importar
                    with open(archivo, 'r', encoding='utf-8') as f:
                        contenido = f.read()
                    
                    # Buscar definiciones de clases
                    clases_encontradas = []
                    for linea in contenido.split('\n'):
                        if linea.strip().startswith('class ') and ':' in linea:
                            clase_def = linea.strip().split('class ')[1].split('(')[0].split(':')[0].strip()
                            if clase_def and not clase_def.startswith('_'):
                                clases_encontradas.append(clase_def)
                    
                    # Intentar importación simple
                    importable = False
                    error_importacion = None
                    try:
                        spec = importlib.util.spec_from_file_location(nombre, archivo)
                        if spec and spec.loader:
                            modulo = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(modulo)
                            
                            # Verificar clases reales en el módulo
                            clases_reales = []
                            for attr_name in dir(modulo):
                                if not attr_name.startswith('_'):
                                    attr = getattr(modulo, attr_name)
                                    if inspect.isclass(attr) and attr.__module__ == modulo.__name__:
                                        clases_reales.append(attr_name)
                            
                            clases_encontradas = clases_reales
                            importable = True
                            
                    except Exception as e:
                        error_importacion = str(e)
                        importable = False
                    
                    archivos_encontrados.append({
                        'archivo': str(archivo),
                        'nombre_modulo': nombre,
                        'clases': clases_encontradas,
                        'tamaño_kb': archivo.stat().st_size / 1024,
                        'importable': importable,
                        'error': error_importacion,
                        'ruta': ruta
                    })
                    
                    if importable:
                        print(f"   ✅ IMPORTABLE")
                        print(f"   🏗️  Clases: {len(clases_encontradas)} encontradas")
                        if clases_encontradas:
                            print(f"   📋 Lista: {', '.join(clases_encontradas[:3])}")
                    else:
                        print(f"   ❌ ERROR DE IMPORTACIÓN")
                        print(f"   ⚠️  Pero encontré: {len(clases_encontradas)} clases")
                        if clases_encontradas:
                            print(f"   📋 Clases: {', '.join(clases_encontradas[:3])}")
                        if error_importacion:
                            print(f"   🐛 Error: {error_importacion[:60]}...")
                    
                    print(f"   💾 Tamaño: {archivo.stat().st_size / 1024:.1f} KB")
                    print()
                        
                except Exception as e:
                    print(f"   💥 ERROR CRÍTICO: {str(e)[:60]}...")
                    print()
    
    print("=" * 60)
    print(f"📊 RESUMEN:")
    print(f"   📁 Archivos Aurora encontrados: {len(archivos_encontrados)}")
    
    importables = [a for a in archivos_encontrados if a['importable']]
    print(f"   ✅ Archivos importables: {len(importables)}")
    
    con_clases = [a for a in archivos_encontrados if a['clases']]
    print(f"   🏗️ Con clases detectadas: {len(con_clases)}")
    
    # Mostrar detalles por ruta
    aurora_system_files = [a for a in archivos_encontrados if a['ruta'] == 'aurora_system']
    main_files = [a for a in archivos_encontrados if a['ruta'] == '.']
    
    print(f"   📂 En aurora_system/: {len(aurora_system_files)} archivos")
    print(f"   📂 En directorio principal: {len(main_files)} archivos")
    
    return archivos_encontrados

def buscar_motores_potenciales(archivos_encontrados):
    """Busca motores potenciales en los archivos encontrados"""
    print("\n🎵 BUSCANDO MOTORES POTENCIALES...")
    print("-" * 60)
    
    motores_potenciales = []
    
    for archivo_info in archivos_encontrados:
        if not archivo_info['importable']:
            continue
            
        nombre = archivo_info['nombre_modulo'].lower()
        clases = archivo_info['clases']
        
        # Detectar NeuroMix
        if 'neuromix' in nombre or any('neuromix' in c.lower() for c in clases):
            motores_potenciales.append({
                'tipo': 'neuromix',
                'archivo': archivo_info['archivo'],
                'modulo': archivo_info['nombre_modulo'],
                'clases_candidatas': [c for c in clases if 'neuromix' in c.lower() or 'neuro' in c.lower()]
            })
            print(f"🎵 NEUROMIX detectado: {archivo_info['nombre_modulo']}")
            print(f"   📦 Archivo: {archivo_info['archivo']}")
            print(f"   🏗️ Clases: {[c for c in clases if 'neuromix' in c.lower() or 'neuro' in c.lower()]}")
            print()
        
        # Detectar HyperMod
        if 'hypermod' in nombre or 'hyper' in nombre or any('hypermod' in c.lower() for c in clases):
            motores_potenciales.append({
                'tipo': 'hypermod',
                'archivo': archivo_info['archivo'],
                'modulo': archivo_info['nombre_modulo'],
                'clases_candidatas': [c for c in clases if 'hypermod' in c.lower() or 'hyper' in c.lower()]
            })
            print(f"🔧 HYPERMOD detectado: {archivo_info['nombre_modulo']}")
            print(f"   📦 Archivo: {archivo_info['archivo']}")
            print(f"   🏗️ Clases: {[c for c in clases if 'hypermod' in c.lower() or 'hyper' in c.lower()]}")
            print()
        
        # Detectar HarmonicEssence
        if 'harmonic' in nombre or 'essence' in nombre or any('harmonic' in c.lower() for c in clases):
            motores_potenciales.append({
                'tipo': 'harmonic',
                'archivo': archivo_info['archivo'],
                'modulo': archivo_info['nombre_modulo'],
                'clases_candidatas': [c for c in clases if 'harmonic' in c.lower() or 'essence' in c.lower()]
            })
            print(f"🎶 HARMONIC detectado: {archivo_info['nombre_modulo']}")
            print(f"   📦 Archivo: {archivo_info['archivo']}")
            print(f"   🏗️ Clases: {[c for c in clases if 'harmonic' in c.lower() or 'essence' in c.lower()]}")
            print()
        
        # Detectar Aurora Director
        if 'director' in nombre or 'aurora' in nombre:
            if any('director' in c.lower() for c in clases):
                motores_potenciales.append({
                    'tipo': 'director',
                    'archivo': archivo_info['archivo'],
                    'modulo': archivo_info['nombre_modulo'],
                    'clases_candidatas': [c for c in clases if 'director' in c.lower() or 'aurora' in c.lower()]
                })
                print(f"🎯 DIRECTOR detectado: {archivo_info['nombre_modulo']}")
                print(f"   📦 Archivo: {archivo_info['archivo']}")
                print(f"   🏗️ Clases: {[c for c in clases if 'director' in c.lower() or 'aurora' in c.lower()]}")
                print()
    
    print("=" * 60)
    print(f"🎵 MOTORES POTENCIALES ENCONTRADOS: {len(motores_potenciales)}")
    
    tipos_encontrados = set(m['tipo'] for m in motores_potenciales)
    print(f"🔥 Tipos disponibles: {', '.join(tipos_encontrados)}")
    
    return motores_potenciales

def test_motores_reales(motores_potenciales):
    """Testea que los motores realmente funcionen"""
    print("\n🧪 TESTEANDO MOTORES REALES...")
    print("-" * 60)
    
    motores_funcionales = []
    
    for motor in motores_potenciales:
        try:
            print(f"🧪 Testeando {motor['tipo'].upper()}: {motor['modulo']}")
            
            # Importar módulo usando la ruta del archivo
            try:
                # Método 1: Importación directa si es posible
                if motor['modulo'] in sys.modules:
                    modulo = sys.modules[motor['modulo']]
                else:
                    modulo = importlib.import_module(motor['modulo'])
                print(f"   ✅ Módulo importado correctamente")
            except ImportError:
                # Método 2: Importar desde archivo específico
                try:
                    spec = importlib.util.spec_from_file_location(motor['modulo'], motor['archivo'])
                    if spec and spec.loader:
                        modulo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(modulo)
                        print(f"   ✅ Módulo cargado desde archivo")
                    else:
                        raise ImportError("No se pudo crear spec")
                except Exception as e2:
                    print(f"   ❌ Error cargando módulo: {e2}")
                    continue
            
            # Probar cada clase candidata
            for clase_nombre in motor['clases_candidatas']:
                try:
                    if not hasattr(modulo, clase_nombre):
                        print(f"   ⚠️  Clase {clase_nombre} no encontrada en módulo")
                        continue
                        
                    clase = getattr(modulo, clase_nombre)
                    
                    # Verificar que es una clase
                    if not inspect.isclass(clase):
                        print(f"   ⚠️  {clase_nombre} no es una clase")
                        continue
                    
                    # Intentar crear instancia
                    try:
                        instancia = clase()
                        print(f"   ✅ Instancia de {clase_nombre} creada")
                    except Exception as e:
                        print(f"   ⚠️  Error creando instancia de {clase_nombre}: {str(e)[:40]}...")
                        # Algunas clases pueden requerir parámetros, las marcamos como disponibles pero sin instancia
                        instancia = None
                    
                    # Verificar métodos útiles
                    metodos_importantes = []
                    metodos_a_verificar = ['generar_audio', 'generate_audio', 'crear_experiencia', 'generar', 'run', 'process']
                    
                    for metodo in metodos_a_verificar:
                        if hasattr(clase, metodo) or (instancia and hasattr(instancia, metodo)):
                            metodos_importantes.append(metodo)
                    
                    if metodos_importantes:
                        motores_funcionales.append({
                            'tipo': motor['tipo'],
                            'modulo': motor['modulo'],
                            'clase': clase_nombre,
                            'instancia': instancia,
                            'clase_objeto': clase,
                            'metodos': metodos_importantes,
                            'archivo': motor['archivo']
                        })
                        
                        print(f"   ✅ Clase {clase_nombre} FUNCIONAL")
                        print(f"   🔧 Métodos: {', '.join(metodos_importantes)}")
                        print()
                        break
                    else:
                        print(f"   ⚠️  Clase {clase_nombre} sin métodos de audio")
                        
                except Exception as e:
                    print(f"   ❌ Error con clase {clase_nombre}: {str(e)[:50]}...")
                    
        except Exception as e:
            print(f"   ❌ Error general con {motor['modulo']}: {str(e)[:50]}...")
            print()
    
    print("=" * 60)
    print(f"🎉 MOTORES FUNCIONALES: {len(motores_funcionales)}")
    
    for motor in motores_funcionales:
        print(f"✅ {motor['tipo'].upper()}: {motor['modulo']}.{motor['clase']}")
        print(f"   🔧 Métodos: {', '.join(motor['metodos'])}")
        print(f"   📁 Archivo: {motor['archivo']}")
    
    return motores_funcionales

def generar_configuracion_flask(motores_funcionales):
    """Genera código de configuración para Flask con los motores reales"""
    print("\n🔧 GENERANDO CONFIGURACIÓN PARA FLASK...")
    print("-" * 60)
    
    if not motores_funcionales:
        print("❌ NO HAY MOTORES FUNCIONALES PARA CONFIGURAR")
        return ""
    
    config_code = f"""#!/usr/bin/env python3
\"\"\"
Configuración automática de motores Aurora V7 detectados
Generado automáticamente por el detector
\"\"\"

import os
import sys
import importlib.util
from pathlib import Path

# Agregar rutas necesarias
if os.path.exists('aurora_system'):
    sys.path.insert(0, os.path.abspath('aurora_system'))

def verificar_motores_reales_detectados():
    \"\"\"Configuración automática basada en detección de tu sistema\"\"\"
    motores_reales = {{}}
    errores_importacion = []
    
    print("🔍 VERIFICANDO MOTORES DETECTADOS AUTOMÁTICAMENTE...")
    print("=" * 60)
"""
    
    for motor in motores_funcionales:
        config_code += f"""
    # {motor['tipo'].upper()} - {motor['modulo']}
    try:
        # Importar desde: {motor['archivo']}
        spec = importlib.util.spec_from_file_location('{motor['modulo']}', '{motor['archivo']}')
        if spec and spec.loader:
            modulo_{motor['tipo']} = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo_{motor['tipo']})
            
            if hasattr(modulo_{motor['tipo']}, '{motor['clase']}'):
                clase_{motor['tipo']} = getattr(modulo_{motor['tipo']}, '{motor['clase']}')
                
                # Intentar crear instancia
                try:
                    instancia_{motor['tipo']} = clase_{motor['tipo']}()
                    instancia_creada = True
                except:
                    # Algunas clases pueden necesitar parámetros
                    instancia_{motor['tipo']} = clase_{motor['tipo']}
                    instancia_creada = False
                
                motores_reales['{motor['tipo']}'] = {{
                    'instancia': instancia_{motor['tipo']},
                    'modulo': '{motor['modulo']}',
                    'clase': '{motor['clase']}',
                    'metodos': {motor['metodos']},
                    'archivo': '{motor['archivo']}',
                    'es_real': True,
                    'instancia_creada': instancia_creada
                }}
                print(f"✅ {motor['tipo'].upper()}: {motor['clase']} - MOTOR REAL DETECTADO")
                print(f"   📍 Archivo: {motor['archivo']}")
                print(f"   🔧 Métodos: {', '.join(motor['metodos'])}")
            else:
                errores_importacion.append("{motor['tipo']}: Clase {motor['clase']} no encontrada")
        else:
            errores_importacion.append("{motor['tipo']}: No se pudo cargar {motor['archivo']}")
            
    except Exception as e:
        errores_importacion.append("{motor['tipo']}: {{str(e)}}")
        print(f"❌ {motor['tipo'].upper()}: Error - {{e}}")
"""
    
    config_code += f"""
    print("=" * 60)
    print(f"📊 RESUMEN FINAL:")
    print(f"   🎵 Motores reales detectados: {{len(motores_reales)}}")
    print(f"   ❌ Errores de importación: {{len(errores_importacion)}}")
    
    if len(motores_reales) > 0:
        print(f"\\n🎉 SISTEMA LISTO CON {{len(motores_reales)}} MOTORES REALES")
        for nombre, info in motores_reales.items():
            print(f"   🔥 {{nombre.upper()}}: {{info['clase']}} ({{info['modulo']}})")
    else:
        print("\\n💥 CRÍTICO: NO SE DETECTARON MOTORES REALES")
        for error in errores_importacion:
            print(f"   • {{error}}")
    
    return motores_reales, errores_importacion

# Variables globales para uso en Flask
MOTORES_REALES, ERRORES_IMPORTACION = verificar_motores_reales_detectados()

# Función auxiliar para verificar disponibilidad
def tiene_motores_reales():
    return len(MOTORES_REALES) > 0

def obtener_motor(tipo_motor):
    \"\"\"Obtiene un motor específico por tipo\"\"\"
    if tipo_motor in MOTORES_REALES:
        return MOTORES_REALES[tipo_motor]['instancia']
    return None

def listar_motores_disponibles():
    \"\"\"Lista todos los motores disponibles\"\"\"
    return list(MOTORES_REALES.keys())

def obtener_info_motor(tipo_motor):
    \"\"\"Obtiene información detallada de un motor\"\"\"
    if tipo_motor in MOTORES_REALES:
        return MOTORES_REALES[tipo_motor]
    return None
"""
    
    print("✅ Configuración generada:")
    print(f"   🎵 Motores incluidos: {len(motores_funcionales)}")
    for motor in motores_funcionales:
        print(f"   🔥 {motor['tipo'].upper()}: {motor['clase']}")
    
    # Guardar en archivo
    try:
        with open("aurora_config_detectada.py", "w", encoding="utf-8") as f:
            f.write(config_code)
        print(f"\n💾 Configuración guardada en: aurora_config_detectada.py")
    except Exception as e:
        print(f"\n❌ Error guardando configuración: {e}")
    
    return config_code

def main():
    """Función principal del detector"""
    print("🚀 INICIANDO DETECCIÓN COMPLETA DE AURORA V7")
    print("=" * 70)
    
    # Paso 1: Detectar archivos
    archivos = detectar_archivos_aurora()
    
    if not archivos:
        print("❌ NO SE ENCONTRARON ARCHIVOS DE AURORA")
        print("\n💡 VERIFICA:")
        print("1. Que estés en el directorio correcto")
        print("2. Que exista la carpeta aurora_system/")
        print("3. Que los archivos .py estén presentes")
        return
    
    # Paso 2: Buscar motores
    motores = buscar_motores_potenciales(archivos)
    
    if not motores:
        print("❌ NO SE ENCONTRARON MOTORES POTENCIALES")
        return
    
    # Paso 3: Testear motores
    motores_funcionales = test_motores_reales(motores)
    
    if not motores_funcionales:
        print("❌ NO SE ENCONTRARON MOTORES FUNCIONALES")
        return
    
    # Paso 4: Generar configuración
    generar_configuracion_flask(motores_funcionales)
    
    print("\n🎉 DETECCIÓN COMPLETADA EXITOSAMENTE!")
    print(f"✅ Motores funcionales encontrados: {len(motores_funcionales)}")
    print("💡 Usa la configuración generada en aurora_config_detectada.py")

if __name__ == "__main__":
    main()
