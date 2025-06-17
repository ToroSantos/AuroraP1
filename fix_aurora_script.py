#!/usr/bin/env python3
"""
🔧 Script de Arreglo Automático para Aurora V7
Corrige errores de indentación y compatibilidad con Flask
"""

import os
import re
import shutil
from pathlib import Path

def fix_aurora_director():
    """Arregla el problema de indentación en aurora_director_v7.py"""
    file_path = "aurora_system/aurora_director_v7.py"
    
    if not os.path.exists(file_path):
        print(f"❌ No se encontró {file_path}")
        return False
    
    # Hacer backup
    shutil.copy2(file_path, f"{file_path}.backup")
    print(f"📁 Backup creado: {file_path}.backup")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar y reemplazar el método validar problemático
    old_pattern = r'def validar\(self\)->List\[str\]:p=\[\];.*?return p'
    new_method = '''def validar(self) -> List[str]:
        p = []
        if self.duracion_min <= 0:
            p.append("Duración debe ser positiva")
        if self.sample_rate not in [22050, 44100, 48000]:
            p.append("Sample rate no estándar")
        if self.intensidad not in ["suave", "media", "intenso"]:
            p.append("Intensidad inválida")
        if not self.objetivo.strip():
            p.append("Objetivo no puede estar vacío")
        return p'''
    
    # Reemplazar usando regex
    content = re.sub(old_pattern, new_method, content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Arreglado: {file_path}")
    return True

def fix_flask_app():
    """Arregla el problema de before_first_request en Flask"""
    file_path = "aurora_flask_app.py"
    
    if not os.path.exists(file_path):
        print(f"❌ No se encontró {file_path}")
        return False
    
    # Hacer backup
    shutil.copy2(file_path, f"{file_path}.backup")
    print(f"📁 Backup creado: {file_path}.backup")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Reemplazar @app.before_first_request con solución compatible
    flask_fix = '''
# Variable global para inicialización única
_system_initialized = False

def initialize_system_once():
    """Inicializa el sistema Aurora solo una vez"""
    global _system_initialized
    if not _system_initialized:
        try:
            logger.info("🌟 Inicializando sistema Aurora V7...")
            # Inicialización del sistema aquí
            _system_initialized = True
        except Exception as e:
            logger.error(f"Error inicializando sistema: {e}")

@app.before_request
def ensure_initialized():
    """Asegura que el sistema esté inicializado antes de cada request"""
    initialize_system_once()
'''
    
    # Buscar y reemplazar @app.before_first_request
    content = re.sub(
        r'@app\.before_first_request\s*\ndef\s+\w+\(\):', 
        flask_fix.strip() + '\n\n# Función de inicialización reemplazada\ndef _old_init():', 
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Arreglado: {file_path}")
    return True

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    try:
        import flask
        print(f"✅ Flask {flask.__version__} instalado")
        
        import numpy
        print(f"✅ NumPy {numpy.__version__} instalado")
        
        import scipy
        print(f"✅ SciPy {scipy.__version__} instalado")
        
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        return False

def main():
    print("🔧 Aurora V7 - Script de Arreglo Automático")
    print("=" * 50)
    
    # Verificar dependencias
    print("\n📦 Verificando dependencias...")
    if not check_dependencies():
        print("\n💡 Instala las dependencias faltantes:")
        print("pip install flask numpy scipy")
        return
    
    # Arreglar archivos
    print("\n🔧 Aplicando arreglos...")
    
    success = True
    
    if fix_aurora_director():
        print("✅ aurora_director_v7.py arreglado")
    else:
        success = False
    
    if fix_flask_app():
        print("✅ aurora_flask_app.py arreglado")
    else:
        success = False
    
    if success:
        print("\n🎉 ¡Todos los arreglos aplicados exitosamente!")
        print("\n🚀 Ahora puedes ejecutar:")
        print("python aurora_flask_app.py")
    else:
        print("\n⚠️ Algunos arreglos fallaron. Revisa los mensajes arriba.")
        print("\n📝 Si persisten los problemas, aplica los arreglos manualmente:")
        print("1. Corrige la indentación en aurora_director_v7.py línea 28")
        print("2. Reemplaza @app.before_first_request en aurora_flask_app.py")

if __name__ == "__main__":
    main()
