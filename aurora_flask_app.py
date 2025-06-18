#!/usr/bin/env python3
"""
Aurora V7 - Backend Flask SOLO MOTORES REALES
Sistema neuroacústico sin fallbacks

✅ CONFIGURADO: Solo usa motores reales de Aurora V7
✅ VERIFICADO: Detecta y reporta motores disponibles
✅ SEGURO: Falla si no hay motores reales disponibles
❌ ELIMINADO: Todos los fallbacks
"""

import os
import uuid
import time
import threading
import atexit
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

# Configurar logging optimizado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Aurora.Flask.RealEngines")

# ===============================================================================
# VERIFICACIÓN ESTRICTA DE MOTORES REALES
# ===============================================================================

def verificar_motores_reales():
    """Verifica que los motores reales de Aurora estén disponibles"""
    motores_reales = {}
    errores_importacion = []
    
    # Lista de motores reales requeridos
    motores_requeridos = {
        'neuromix': {
            'modulo': 'neuromix_definitivo_v7',
            'clase': 'NeuroMixSuperUnificado',
            'alternativas': ['neuromix_aurora_v27', 'AuroraNeuroAcousticEngineV27']
        },
        'hypermod': {
            'modulo': 'hypermod_v32',
            'clase': 'HyperModEngineV32AuroraConnected',
            'alternativas': ['HyperModV32', 'HyperModEngine']
        },
        'harmonic': {
            'modulo': 'harmonicEssence_v34',
            'clase': 'HarmonicEssenceV34AuroraConnected',
            'alternativas': ['HarmonicEssenceV34', 'HarmonicEngine']
        }
    }
    
    print("🔍 VERIFICANDO MOTORES REALES DE AURORA V7...")
    print("=" * 60)
    
    for nombre_motor, config in motores_requeridos.items():
        try:
            # Intentar importar módulo principal
            try:
                modulo = __import__(config['modulo'], fromlist=[config['clase']])
                clase_motor = getattr(modulo, config['clase'])
                
                # Verificar que sea una clase real y no un mock
                if hasattr(clase_motor, '__module__') and clase_motor.__module__ != '__main__':
                    # Intentar crear instancia para verificar que funciona
                    instancia = clase_motor()
                    
                    # Verificar métodos críticos
                    if hasattr(instancia, 'generar_audio') or hasattr(instancia, 'generate_audio'):
                        motores_reales[nombre_motor] = {
                            'instancia': instancia,
                            'modulo': config['modulo'],
                            'clase': config['clase'],
                            'metodos': [m for m in dir(instancia) if not m.startswith('_')],
                            'es_real': True
                        }
                        print(f"✅ {nombre_motor.upper()}: {config['clase']} - MOTOR REAL DETECTADO")
                        print(f"   📍 Módulo: {config['modulo']}")
                        print(f"   🔧 Métodos: {len([m for m in dir(instancia) if not m.startswith('_')])}")
                    else:
                        errores_importacion.append(f"{nombre_motor}: Clase sin método generar_audio")
                        print(f"❌ {nombre_motor.upper()}: Sin métodos de generación")
                else:
                    errores_importacion.append(f"{nombre_motor}: Clase no es real")
                    print(f"❌ {nombre_motor.upper()}: Clase mock detectada")
                    
            except (ImportError, AttributeError) as e:
                # Intentar alternativas
                motor_encontrado = False
                for alt_clase in config['alternativas']:
                    try:
                        # Buscar en diferentes módulos
                        posibles_modulos = [config['modulo'], alt_clase.lower(), f"{alt_clase.lower()}_v7"]
                        
                        for mod_name in posibles_modulos:
                            try:
                                alt_modulo = __import__(mod_name, fromlist=[alt_clase])
                                alt_clase_motor = getattr(alt_modulo, alt_clase)
                                alt_instancia = alt_clase_motor()
                                
                                if hasattr(alt_instancia, 'generar_audio') or hasattr(alt_instancia, 'generate_audio'):
                                    motores_reales[nombre_motor] = {
                                        'instancia': alt_instancia,
                                        'modulo': mod_name,
                                        'clase': alt_clase,
                                        'metodos': [m for m in dir(alt_instancia) if not m.startswith('_')],
                                        'es_real': True
                                    }
                                    print(f"✅ {nombre_motor.upper()}: {alt_clase} - MOTOR REAL (alternativo)")
                                    print(f"   📍 Módulo: {mod_name}")
                                    motor_encontrado = True
                                    break
                            except:
                                continue
                        if motor_encontrado:
                            break
                    except:
                        continue
                
                if not motor_encontrado:
                    errores_importacion.append(f"{nombre_motor}: {str(e)}")
                    print(f"❌ {nombre_motor.upper()}: NO DISPONIBLE - {str(e)}")
                    
        except Exception as e:
            errores_importacion.append(f"{nombre_motor}: Error crítico - {str(e)}")
            print(f"💥 {nombre_motor.upper()}: ERROR CRÍTICO - {str(e)}")
    
    print("=" * 60)
    print(f"📊 RESUMEN:")
    print(f"   🎵 Motores reales detectados: {len(motores_reales)}")
    print(f"   ❌ Errores de importación: {len(errores_importacion)}")
    
    if len(motores_reales) == 0:
        print("\n💥 CRÍTICO: NO SE DETECTARON MOTORES REALES")
        print("🚫 El sistema NO PUEDE FUNCIONAR sin motores reales")
        print("\n💡 SOLUCIONES:")
        print("1. Verificar que los módulos estén en el PYTHONPATH")
        print("2. Instalar dependencias faltantes")
        print("3. Verificar estructura de carpetas aurora_system/")
        
        for error in errores_importacion:
            print(f"   • {error}")
        
        return None, errores_importacion
    
    print(f"\n🎉 SISTEMA LISTO CON {len(motores_reales)} MOTORES REALES")
    return motores_reales, []

# Verificar motores al inicio
MOTORES_REALES, ERRORES_IMPORTACION = verificar_motores_reales()

if MOTORES_REALES is None:
    print("\n🚫 DETENIENDO APLICACIÓN: No hay motores reales disponibles")
    exit(1)

# ===============================================================================
# IMPORTAR AURORA BRIDGE SOLO SI HAY MOTORES REALES
# ===============================================================================

try:
    from aurora_bridge import AuroraBridge, GenerationStatus
    AURORA_BRIDGE_AVAILABLE = True
    logger.info("✅ Aurora Bridge importado - Usando motores reales")
except ImportError as e:
    logger.error(f"❌ Aurora Bridge no disponible: {e}")
    print("💥 CRÍTICO: Aurora Bridge es requerido para motores reales")
    exit(1)

# ===============================================================================
# VERIFICADOR DE MOTORES REALES EN TIEMPO DE EJECUCIÓN
# ===============================================================================

class VerificadorMotoresReales:
    """Verifica y monitorea que solo se usen motores reales"""
    
    def __init__(self):
        self.motores_verificados = MOTORES_REALES
        self.contadores = {
            'generaciones_reales': 0,
            'fallbacks_bloqueados': 0,
            'verificaciones_realizadas': 0
        }
        
    def verificar_motor_es_real(self, motor_instance, nombre_motor="unknown"):
        """Verifica que un motor sea real y no fallback"""
        self.contadores['verificaciones_realizadas'] += 1
        
        if motor_instance is None:
            logger.error(f"🚫 BLOQUEADO: Motor {nombre_motor} es None - NO SE PERMITE")
            self.contadores['fallbacks_bloqueados'] += 1
            return False
            
        # Verificar que no sea una función de fallback
        if hasattr(motor_instance, '__name__') and 'fallback' in motor_instance.__name__.lower():
            logger.error(f"🚫 BLOQUEADO: Motor {nombre_motor} es fallback - NO SE PERMITE")
            self.contadores['fallbacks_bloqueados'] += 1
            return False
            
        # Verificar que tenga métodos reales
        if not (hasattr(motor_instance, 'generar_audio') or hasattr(motor_instance, 'generate_audio')):
            logger.error(f"🚫 BLOQUEADO: Motor {nombre_motor} sin métodos de generación")
            self.contadores['fallbacks_bloqueados'] += 1
            return False
            
        # Verificar que no sea un mock
        clase_motor = motor_instance.__class__
        if clase_motor.__module__ == '__main__' or 'mock' in clase_motor.__name__.lower():
            logger.error(f"🚫 BLOQUEADO: Motor {nombre_motor} es mock - NO SE PERMITE")
            self.contadores['fallbacks_bloqueados'] += 1
            return False
            
        logger.info(f"✅ VERIFICADO: Motor {nombre_motor} es REAL - {clase_motor.__name__}")
        self.contadores['generaciones_reales'] += 1
        return True
    
    def obtener_estadisticas(self):
        """Obtiene estadísticas de verificación"""
        return {
            'motores_reales_detectados': len(self.motores_verificados),
            'generaciones_con_motores_reales': self.contadores['generaciones_reales'],
            'fallbacks_bloqueados': self.contadores['fallbacks_bloqueados'],
            'verificaciones_realizadas': self.contadores['verificaciones_realizadas'],
            'motores_disponibles': list(self.motores_verificados.keys())
        }

# Crear verificador global
verificador_motores = VerificadorMotoresReales()

# ===============================================================================
# AURORA BRIDGE PERSONALIZADO SIN FALLBACKS
# ===============================================================================

class AuroraBridgeMotoresReales:
    """Aurora Bridge que SOLO usa motores reales - Sin fallbacks"""
    
    def __init__(self, audio_folder: str = "static/audio", sample_rate: int = 44100):
        self.audio_folder = Path(audio_folder)
        self.audio_folder.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.motores_reales = MOTORES_REALES
        
        # Verificar que tenemos motores reales
        if not self.motores_reales:
            raise Exception("❌ NO HAY MOTORES REALES DISPONIBLES")
            
        # Inicializar Aurora Bridge real si está disponible
        try:
            self.aurora_bridge_real = AuroraBridge(audio_folder=audio_folder, sample_rate=sample_rate)
            logger.info("✅ Aurora Bridge real inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando Aurora Bridge real: {e}")
            raise Exception("💥 Aurora Bridge real es requerido")
        
        logger.info(f"🎵 Sistema inicializado con {len(self.motores_reales)} motores reales")
        self._imprimir_motores_disponibles()
    
    def _imprimir_motores_disponibles(self):
        """Imprime información detallada de motores disponibles"""
        print("\n🎵 MOTORES REALES DISPONIBLES:")
        print("=" * 50)
        
        for nombre, info in self.motores_reales.items():
            print(f"🔥 {nombre.upper()}:")
            print(f"   📦 Módulo: {info['modulo']}")
            print(f"   🏗️  Clase: {info['clase']}")
            print(f"   ⚙️  Métodos: {len(info['metodos'])}")
            print(f"   ✅ Estado: MOTOR REAL VERIFICADO")
            print()
    
    def get_complete_system_status(self):
        """Estado del sistema solo con motores reales"""
        try:
            # Usar estado del Aurora Bridge real si está disponible
            estado_real = self.aurora_bridge_real.get_complete_system_status()
            
            # Agregar información de verificación
            estado_real['motores_reales'] = {
                'detectados': len(self.motores_reales),
                'verificados': True,
                'fallbacks_habilitados': False,
                'detalles': {nombre: {
                    'modulo': info['modulo'],
                    'clase': info['clase'],
                    'es_real': info['es_real']
                } for nombre, info in self.motores_reales.items()}
            }
            
            estado_real['verificador_stats'] = verificador_motores.obtener_estadisticas()
            
            return estado_real
            
        except Exception as e:
            logger.error(f"Error obteniendo estado: {e}")
            return {
                "summary": {
                    "status": "error",
                    "message": f"Error en sistema real: {str(e)}",
                    "motores_reales_disponibles": len(self.motores_reales),
                    "fallbacks_habilitados": False
                }
            }
    
    def validate_generation_params(self, params):
        """Validación usando Aurora Bridge real"""
        if not self.aurora_bridge_real:
            return {
                "valid": False,
                "errors": ["Aurora Bridge real no disponible"],
                "warnings": []
            }
        
        # Usar validación real
        resultado = self.aurora_bridge_real.validate_generation_params(params)
        
        # Verificar que no se esté intentando usar fallback
        if params.get('usar_fallback', False):
            resultado['valid'] = False
            resultado['errors'].append("Fallbacks están DESHABILITADOS - solo motores reales")
        
        return resultado
    
    def estimate_generation_time(self, params):
        """Estimación usando Aurora Bridge real"""
        if self.aurora_bridge_real:
            return self.aurora_bridge_real.estimate_generation_time(params)
        else:
            # Estimación básica sin fallback
            duracion_min = params.get('duracion_min', 20)
            return max(duracion_min * 2, 30)  # Mínimo 30 segundos
    
    def crear_experiencia_completa(self, params, progress_callback=None):
        """Creación usando SOLO motores reales"""
        try:
            logger.info("🎵 Iniciando generación con MOTORES REALES únicamente")
            
            if progress_callback:
                progress_callback(5, "Verificando que SOLO se usen motores reales...")
            
            # Verificar que tenemos motores reales
            if not self.motores_reales:
                raise Exception("❌ NO HAY MOTORES REALES DISPONIBLES")
            
            # BLOQUEAR cualquier intento de usar fallback
            if params.get('usar_fallback', False):
                raise Exception("🚫 FALLBACKS ESTÁN DESHABILITADOS")
            
            if params.get('modo_fallback', False):
                raise Exception("🚫 MODO FALLBACK NO PERMITIDO")
            
            # Forzar parámetros para solo motores reales
            params_reales = params.copy()
            params_reales.update({
                'usar_fallback': False,
                'modo_fallback': False,
                'solo_motores_reales': True,
                'verificar_motores': True
            })
            
            if progress_callback:
                progress_callback(10, f"Usando {len(self.motores_reales)} motores reales verificados")
            
            # Usar Aurora Bridge real
            if self.aurora_bridge_real:
                resultado = self.aurora_bridge_real.crear_experiencia_completa(
                    params_reales, progress_callback
                )
                
                # Verificar que el resultado sea de motores reales
                if resultado.get('success'):
                    metadata = resultado.get('metadata', {})
                    generator = metadata.get('generator', 'unknown')
                    
                    # Verificar que no sea fallback
                    if 'fallback' in generator.lower():
                        raise Exception(f"🚫 RESULTADO DE FALLBACK DETECTADO: {generator}")
                    
                    logger.info(f"✅ Audio generado con MOTORES REALES: {generator}")
                    
                    # Actualizar metadata con información de verificación
                    metadata.update({
                        'motores_reales_usados': list(self.motores_reales.keys()),
                        'fallbacks_habilitados': False,
                        'verificacion_motores': 'SOLO_MOTORES_REALES',
                        'stats_verificador': verificador_motores.obtener_estadisticas()
                    })
                    
                    resultado['metadata'] = metadata
                    return resultado
                else:
                    error_msg = resultado.get('error', 'Error desconocido')
                    raise Exception(f"❌ Error en motores reales: {error_msg}")
            else:
                raise Exception("💥 Aurora Bridge real no disponible")
                
        except Exception as e:
            logger.error(f"❌ Error en generación con motores reales: {e}")
            
            # NO USAR FALLBACK - Fallar explícitamente
            return {
                'success': False,
                'error': f"Falla en motores reales (sin fallback): {str(e)}",
                'metadata': {
                    'generator': 'ERROR_MOTORES_REALES',
                    'fallbacks_habilitados': False,
                    'motores_reales_disponibles': len(self.motores_reales),
                    'error_timestamp': datetime.now().isoformat()
                }
            }

# ===============================================================================
# CONFIGURACIÓN FLASK CON VERIFICACIÓN DE MOTORES REALES
# ===============================================================================

app = Flask(__name__)

# CORS optimizado
CORS(app, 
     origins=["*"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
     supports_credentials=True
)

# Configuración Flask
app.config.update({
    'SECRET_KEY': 'aurora-v7-real-engines-only',
    'MAX_CONTENT_LENGTH': 32 * 1024 * 1024,
    'AUDIO_FOLDER': 'static/audio',
    'GENERATION_TIMEOUT': 600,
    'SEND_FILE_MAX_AGE_DEFAULT': 0,
    'TEMPLATES_AUTO_RELOAD': True,
    'JSON_SORT_KEYS': False,
    'JSONIFY_PRETTYPRINT_REGULAR': True,
    'SOLO_MOTORES_REALES': True,  # Nuevo flag
    'FALLBACKS_HABILITADOS': False  # Explícitamente deshabilitados
})

# Crear carpetas
audio_folder = Path(app.config['AUDIO_FOLDER'])
audio_folder.mkdir(parents=True, exist_ok=True)

# Inicializar Aurora Bridge solo con motores reales
try:
    aurora_bridge = AuroraBridgeMotoresReales(
        audio_folder=app.config['AUDIO_FOLDER'],
        sample_rate=44100
    )
    logger.info("✅ Aurora Bridge con motores reales inicializado")
except Exception as e:
    logger.error(f"💥 CRÍTICO: No se pudo inicializar con motores reales: {e}")
    print(f"\n💥 FALLA CRÍTICA: {e}")
    print("🚫 El sistema requiere motores reales para funcionar")
    exit(1)

# Storage para sesiones
generation_sessions = {}
generation_lock = threading.RLock()

# Estados de generación
class GenerationStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class GenerationSession:
    def __init__(self, session_id: str, params: dict):
        self.session_id = session_id
        self.params = params
        self.status = GenerationStatus.PENDING
        self.progress = 0
        self.created_at = datetime.now()
        self.completed_at = None
        self.error_message = None
        self.audio_filename = None
        self.metadata = {}
        self.estimated_duration = 0
        self.download_count = 0
        self.file_size = 0
        self.motores_reales_usados = []
        
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'status': self.status,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'audio_filename': self.audio_filename,
            'metadata': self.metadata,
            'estimated_duration': self.estimated_duration,
            'download_count': self.download_count,
            'file_size': self.file_size,
            'motores_reales_usados': self.motores_reales_usados,
            'fallbacks_habilitados': False
        }

# ===============================================================================
# RUTAS PRINCIPALES
# ===============================================================================

@app.route('/')
def index():
    """Página principal mostrando motores reales"""
    motores_info = ""
    for nombre, info in MOTORES_REALES.items():
        motores_info += f"""
        <div style="background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #28a745;">
            <h4>🔥 {nombre.upper()} - MOTOR REAL</h4>
            <p><strong>Módulo:</strong> {info['modulo']}</p>
            <p><strong>Clase:</strong> {info['clase']}</p>
            <p><strong>Métodos:</strong> {len(info['metodos'])}</p>
        </div>"""
    
    verificador_stats = verificador_motores.obtener_estadisticas()
    
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aurora V7 - Solo Motores Reales</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
                min-height: 100vh; display: flex; align-items: center; justify-content: center;
                color: #333;
            }}
            .container {{ 
                max-width: 1000px; background: rgba(255,255,255,0.98); 
                padding: 40px; border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            }}
            h1 {{ color: #2E8B57; text-align: center; margin-bottom: 30px; }}
            .status {{ 
                background: linear-gradient(45deg, #90EE90, #98FB98); 
                padding: 20px; border-radius: 10px; margin: 20px 0; 
                border-left: 5px solid #228B22;
            }}
            .real-engines {{ 
                background: #f0fdf4; padding: 20px; border-radius: 10px; 
                border: 2px solid #22c55e; margin: 20px 0;
            }}
            .no-fallback {{ 
                background: #fef2f2; padding: 15px; border-radius: 8px; 
                border-left: 5px solid #ef4444; margin: 20px 0;
            }}
            .stats {{ 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px; margin: 20px 0; 
            }}
            .stat-card {{ 
                background: #f8fafc; padding: 15px; border-radius: 8px; 
                border: 1px solid #e2e8f0; text-align: center;
            }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #2E8B57; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎵 Aurora V7 - Solo Motores Reales</h1>
            
            <div class="status">
                <h3>✅ Sistema Operativo con Motores Reales</h3>
                <p><strong>Estado:</strong> Motores Aurora V7 reales verificados y activos</p>
                <p><strong>Tiempo:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Modo:</strong> SOLO MOTORES REALES</p>
                <p><strong>Fallbacks:</strong> ❌ COMPLETAMENTE DESHABILITADOS</p>
            </div>
            
            <div class="real-engines">
                <h3>🔥 Motores Reales Detectados ({len(MOTORES_REALES)})</h3>
                {motores_info}
            </div>
            
            <div class="no-fallback">
                <h3>🚫 Política de No-Fallbacks</h3>
                <p><strong>Este sistema NO utiliza fallbacks.</strong> Solo motores reales de Aurora V7.</p>
                <p>Si un motor real falla, la generación fallará explícitamente.</p>
                <p>Esto garantiza la máxima calidad y autenticidad del audio neuroacústico.</p>
            </div>
            
            <h3>📊 Estadísticas del Verificador:</h3>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{verificador_stats['motores_reales_detectados']}</div>
                    <div>Motores Reales</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{verificador_stats['generaciones_con_motores_reales']}</div>
                    <div>Generaciones Reales</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{verificador_stats['fallbacks_bloqueados']}</div>
                    <div>Fallbacks Bloqueados</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{verificador_stats['verificaciones_realizadas']}</div>
                    <div>Verificaciones</div>
                </div>
            </div>
            
            <h3>🔗 Enlaces de API:</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
                <a href="/test" style="background: #f8fafc; padding: 15px; border-radius: 8px; text-decoration: none; color: #374151;">
                    <strong>Test de Motores Reales</strong><br>
                    Verificar motores disponibles
                </a>
                <a href="/api/health" style="background: #f8fafc; padding: 15px; border-radius: 8px; text-decoration: none; color: #374151;">
                    <strong>Health Check Real</strong><br>
                    Estado de motores reales
                </a>
                <a href="/api/system/status" style="background: #f8fafc; padding: 15px; border-radius: 8px; text-decoration: none; color: #374151;">
                    <strong>Estado del Sistema</strong><br>
                    Diagnóstico completo
                </a>
                <a href="/api/real-engines" style="background: #f8fafc; padding: 15px; border-radius: 8px; text-decoration: none; color: #374151;">
                    <strong>Info Motores Reales</strong><br>
                    Detalles técnicos
                </a>
            </div>
            
            <div style="background: #fff3cd; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #ffc107;">
                <h3>🎵 Test de Generación:</h3>
                <p><strong>Endpoint:</strong> <code>POST /api/generate/experience</code></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">{{
    "objetivo": "concentración profunda",
    "duracion_min": 5,
    "intensidad": "media",
    "estilo": "crystalline",
    "solo_motores_reales": true
}}</pre>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/test')
def simple_test():
    """Test que verifica motores reales"""
    verificador_stats = verificador_motores.obtener_estadisticas()
    
    return jsonify({
        'message': '🎵 Aurora V7 funcionando con MOTORES REALES únicamente',
        'timestamp': datetime.now().isoformat(),
        'version': 'V7-RealEnginesOnly',
        'status': 'operational_real_engines',
        'motores_reales': {
            'detectados': len(MOTORES_REALES),
            'nombres': list(MOTORES_REALES.keys()),
            'detalles': {nombre: {
                'modulo': info['modulo'], 
                'clase': info['clase'],
                'es_real': True
            } for nombre, info in MOTORES_REALES.items()}
        },
        'fallbacks_status': {
            'habilitados': False,
            'bloqueados': verificador_stats['fallbacks_bloqueados'],
            'politica': 'SOLO_MOTORES_REALES'
        },
        'verificador_stats': verificador_stats
    })

@app.route('/api/real-engines')
def real_engines_info():
    """Información detallada de motores reales"""
    return jsonify({
        'motores_reales_detectados': len(MOTORES_REALES),
        'fallbacks_habilitados': False,
        'politica_sistema': 'SOLO_MOTORES_REALES',
        'motores_detalles': MOTORES_REALES,
        'errores_importacion': ERRORES_IMPORTACION,
        'verificador_stats': verificador_motores.obtener_estadisticas(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health_check():
    """Health check verificando solo motores reales"""
    try:
        start_time = time.time()
        
        # Verificar que todos los motores sean reales
        motores_verificados = 0
        for nombre, info in MOTORES_REALES.items():
            if verificador_motores.verificar_motor_es_real(info['instancia'], nombre):
                motores_verificados += 1
        
        with generation_lock:
            session_stats = {
                'total_sessions': len(generation_sessions),
                'active_sessions': len([s for s in generation_sessions.values() 
                                      if s.status in [GenerationStatus.PENDING, GenerationStatus.PROCESSING]]),
                'completed_sessions': len([s for s in generation_sessions.values() 
                                         if s.status == GenerationStatus.COMPLETED])
            }
        
        response_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'status': 'healthy_real_engines',
            'timestamp': datetime.now().isoformat(),
            'version': 'Aurora V7 - Real Engines Only',
            'response_time_ms': round(response_time, 2),
            'motores_reales': {
                'total_detectados': len(MOTORES_REALES),
                'verificados_activos': motores_verificados,
                'nombres': list(MOTORES_REALES.keys()),
                'fallbacks_habilitados': False
            },
            'components': {
                'aurora_bridge_real': 'ok',
                'verificador_motores': 'ok',
                'flask_app': 'ok',
                'audio_folder': audio_folder.exists(),
                'sessions': session_stats
            },
            'verificador_stats': verificador_motores.obtener_estadisticas()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'motores_reales_disponibles': len(MOTORES_REALES),
            'fallbacks_habilitados': False
        }), 500

@app.route('/api/system/status')
def system_status():
    """Estado completo del sistema con motores reales"""
    try:
        if aurora_bridge:
            status = aurora_bridge.get_complete_system_status()
        else:
            status = {"summary": {"status": "no_real_bridge", "message": "Aurora Bridge real no disponible"}}
        
        with generation_lock:
            status['session_stats'] = {
                'active_sessions': len([s for s in generation_sessions.values() 
                                     if s.status in [GenerationStatus.PENDING, GenerationStatus.PROCESSING]]),
                'completed_sessions': len([s for s in generation_sessions.values() 
                                        if s.status == GenerationStatus.COMPLETED]),
                'total_sessions': len(generation_sessions)
            }
        
        status['server_info'] = {
            'timestamp': datetime.now().isoformat(),
            'debug_mode': app.debug,
            'audio_files_count': len(list(audio_folder.glob("*.wav"))),
            'solo_motores_reales': True,
            'fallbacks_habilitados': False
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'error': 'Error interno del sistema',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'motores_reales_disponibles': len(MOTORES_REALES),
            'fallbacks_habilitados': False
        }), 500

@app.route('/api/generate/experience', methods=['POST'])
def generate_experience():
    """Generar experiencia usando SOLO motores reales"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type debe ser application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON válidos'}), 400
        
        # Verificar que no se solicite fallback
        if data.get('usar_fallback', False) or data.get('modo_fallback', False):
            return jsonify({
                'error': 'Fallbacks están DESHABILITADOS en este sistema',
                'politica': 'SOLO_MOTORES_REALES',
                'motores_disponibles': list(MOTORES_REALES.keys())
            }), 400
        
        # Validar parámetros básicos
        objetivo = data.get('objetivo', '').strip()
        if not objetivo:
            return jsonify({'error': 'El campo "objetivo" es requerido'}), 400
            
        duracion_min = data.get('duracion_min', 10)
        try:
            duracion_min = int(duracion_min)
            if duracion_min < 1 or duracion_min > 120:
                return jsonify({'error': 'Duración debe estar entre 1 y 120 minutos'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Duración debe ser un número válido'}), 400
        
        # Parámetros forzados para motores reales
        params = {
            'objetivo': objetivo,
            'duracion_min': duracion_min,
            'intensidad': data.get('intensidad', 'media'),
            'estilo': data.get('estilo', 'crystalline'),
            'calidad_objetivo': data.get('calidad_objetivo', 'alta'),
            'sample_rate': data.get('sample_rate', 44100),
            # Forzar uso de motores reales
            'solo_motores_reales': True,
            'usar_fallback': False,
            'modo_fallback': False,
            'verificar_motores': True
        }
        
        # Validar con Aurora Bridge real
        validation_result = aurora_bridge.validate_generation_params(params)
        if not validation_result['valid']:
            return jsonify({
                'error': 'Parámetros inválidos para motores reales',
                'details': validation_result['errors'],
                'motores_reales_disponibles': list(MOTORES_REALES.keys())
            }), 400
        
        # Crear sesión
        session_id = str(uuid.uuid4())
        session = GenerationSession(session_id, params)
        session.motores_reales_usados = list(MOTORES_REALES.keys())
        
        # Estimar duración
        try:
            session.estimated_duration = aurora_bridge.estimate_generation_time(params)
        except:
            session.estimated_duration = duracion_min * 3
        
        # Almacenar sesión
        with generation_lock:
            generation_sessions[session_id] = session
        
        # Iniciar generación asíncrona
        thread = threading.Thread(
            target=_generate_audio_async, 
            args=(session_id,),
            name=f"RealEngine-{session_id[:8]}",
            daemon=True
        )
        thread.start()
        
        logger.info(f"🎵 Generación REAL iniciada: {session_id} - {objetivo} ({duracion_min}min)")
        
        return jsonify({
            'session_id': session_id,
            'status': 'accepted',
            'estimated_duration': session.estimated_duration,
            'message': 'Generación iniciada con MOTORES REALES únicamente',
            'motores_reales_usados': session.motores_reales_usados,
            'fallbacks_habilitados': False,
            'endpoints': {
                'status': f'/api/generation/{session_id}/status',
                'download': f'/api/generation/{session_id}/download'
            }
        }), 202
        
    except Exception as e:
        logger.error(f"Error en generate_experience: {e}")
        return jsonify({
            'error': 'Error interno del servidor con motores reales',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'motores_reales_disponibles': len(MOTORES_REALES),
            'fallbacks_habilitados': False
        }), 500

@app.route('/api/generation/<session_id>/status')
def generation_status(session_id):
    """Estado de generación con información de motores reales"""
    with generation_lock:
        session = generation_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        status_dict = session.to_dict()
        status_dict['verificador_stats'] = verificador_motores.obtener_estadisticas()
        return jsonify(status_dict)

@app.route('/api/generation/<session_id>/download')
def download_audio(session_id):
    """Descargar audio generado con motores reales"""
    with generation_lock:
        session = generation_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        if session.status != GenerationStatus.COMPLETED:
            return jsonify({
                'error': 'Generación no completada',
                'current_status': session.status,
                'motores_reales_usados': session.motores_reales_usados
            }), 400
        
        if not session.audio_filename:
            return jsonify({'error': 'Archivo de audio no disponible'}), 404
        
        audio_path = audio_folder / session.audio_filename
        if not audio_path.exists():
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        session.download_count += 1
        
        try:
            return send_file(
                audio_path,
                as_attachment=True,
                download_name=f"aurora_real_{session_id[:8]}_{session.params['objetivo'][:20].replace(' ', '_')}.wav",
                mimetype='audio/wav'
            )
        except Exception as e:
            logger.error(f"Error enviando archivo: {e}")
            return jsonify({'error': 'Error enviando archivo'}), 500

# ===============================================================================
# GENERACIÓN ASÍNCRONA SOLO CON MOTORES REALES
# ===============================================================================

def _generate_audio_async(session_id: str):
    """Generar audio en background usando SOLO motores reales"""
    try:
        with generation_lock:
            session = generation_sessions.get(session_id)
            if not session:
                return
        
        session.status = GenerationStatus.PROCESSING
        
        def progress_callback(progress: int, status: str = None):
            with generation_lock:
                current_session = generation_sessions.get(session_id)
                if current_session:
                    current_session.progress = min(95, max(10, progress))
                    if status:
                        logger.info(f"🎵 REAL Session {session_id}: {status} ({progress}%)")
        
        # Verificar motores antes de generar
        motores_verificados = []
        for nombre, info in MOTORES_REALES.items():
            if verificador_motores.verificar_motor_es_real(info['instancia'], nombre):
                motores_verificados.append(nombre)
        
        if not motores_verificados:
            raise Exception("❌ NO HAY MOTORES REALES VERIFICADOS DISPONIBLES")
        
        logger.info(f"🎵 Usando motores reales verificados: {motores_verificados}")
        
        # Generar usando Aurora Bridge real únicamente
        resultado = aurora_bridge.crear_experiencia_completa(
            session.params,
            progress_callback=progress_callback
        )
        
        # Procesar resultado
        with generation_lock:
            current_session = generation_sessions.get(session_id)
            if not current_session:
                return
            
            if resultado.get('success', False):
                # Verificar que no sea fallback
                metadata = resultado.get('metadata', {})
                generator = metadata.get('generator', 'unknown')
                
                if 'fallback' in generator.lower():
                    raise Exception(f"🚫 FALLBACK DETECTADO EN RESULTADO: {generator}")
                
                current_session.status = GenerationStatus.COMPLETED
                current_session.progress = 100
                current_session.completed_at = datetime.now()
                current_session.audio_filename = resultado.get('filename')
                current_session.metadata = metadata
                current_session.motores_reales_usados = motores_verificados
                
                if current_session.audio_filename:
                    audio_path = audio_folder / current_session.audio_filename
                    if audio_path.exists():
                        current_session.file_size = audio_path.stat().st_size
                
                logger.info(f"✅ Generación REAL completada: {session_id} usando {motores_verificados}")
                
            else:
                error_msg = resultado.get('error', 'Error desconocido en motores reales')
                current_session.status = GenerationStatus.ERROR
                current_session.error_message = f"Error en motores reales: {error_msg}"
                logger.error(f"❌ Error en generación REAL {session_id}: {current_session.error_message}")
                
    except Exception as e:
        logger.error(f"❌ Error crítico en generación REAL {session_id}: {e}")
        with generation_lock:
            session = generation_sessions.get(session_id)
            if session:
                session.status = GenerationStatus.ERROR
                session.error_message = f"Error crítico en motores reales: {str(e)}"

# ===============================================================================
# MANEJO DE ERRORES Y CLEANUP
# ===============================================================================

def cleanup_old_sessions():
    """Limpiar sesiones antiguas"""
    try:
        current_time = datetime.now()
        with generation_lock:
            expired_sessions = []
            
            for session_id, session in generation_sessions.items():
                time_diff = (current_time - session.created_at).total_seconds()
                if time_diff > app.config['GENERATION_TIMEOUT']:
                    expired_sessions.append(session_id)
                    
                    if session.audio_filename:
                        audio_path = audio_folder / session.audio_filename
                        if audio_path.exists():
                            try:
                                audio_path.unlink()
                                logger.info(f"Archivo limpiado: {session.audio_filename}")
                            except Exception as e:
                                logger.warning(f"Error limpiando archivo: {e}")
            
            for session_id in expired_sessions:
                del generation_sessions[session_id]
            
            if expired_sessions:
                logger.info(f"🧹 Limpieza completada: {len(expired_sessions)} sesiones")
                
    except Exception as e:
        logger.error(f"Error en cleanup: {e}")

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint no encontrado',
        'message': f'La ruta "{request.path}" no existe',
        'sistema': 'SOLO_MOTORES_REALES',
        'available_endpoints': ['/', '/test', '/api/health', '/api/real-engines']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno: {error}")
    return jsonify({
        'error': 'Error interno del servidor',
        'message': 'Error en sistema de motores reales',
        'timestamp': datetime.now().isoformat(),
        'motores_reales_disponibles': len(MOTORES_REALES),
        'fallbacks_habilitados': False
    }), 500

# ===============================================================================
# FUNCIÓN DE INICIO CON VERIFICACIÓN DE MOTORES REALES
# ===============================================================================

def run_real_engines_server(host='0.0.0.0', port=5001, debug=True):
    """Ejecutar servidor con verificación de motores reales"""
    app.debug = debug
    app._start_time = time.time()
    
    print("\n" + "🔥" * 70)
    print("🎵 AURORA V7 BACKEND - SOLO MOTORES REALES")
    print("🔥" * 70)
    print(f"🌐 Servidor: http://{host}:{port}")
    print(f"🌐 Local: http://localhost:{port}")
    print(f"🔧 Debug: {'✅' if debug else '❌'}")
    print(f"🎵 Motores Reales: ✅ {len(MOTORES_REALES)} detectados")
    print(f"🚫 Fallbacks: ❌ COMPLETAMENTE DESHABILITADOS")
    print(f"📁 Audio: {audio_folder}")
    print("🔥" * 70)
    print("🎵 MOTORES REALES ACTIVOS:")
    for nombre, info in MOTORES_REALES.items():
        print(f"   🔥 {nombre.upper()}: {info['clase']} ({info['modulo']})")
    print("🔥" * 70)
    print(f"🔗 Test: http://localhost:{port}/test")
    print(f"🔗 Motores: http://localhost:{port}/api/real-engines")
    print(f"🔗 Health: http://localhost:{port}/api/health")
    print("🔥" * 70)
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\n💡 SOLUCIONES:")
        print(f"1. Puerto ocupado? Prueba: python aurora_flask_app.py --port 5002")
        print(f"2. Verificar que los motores reales estén instalados")
        print(f"3. Revisar el PYTHONPATH para módulos Aurora V7")

# ===============================================================================
# PUNTO DE ENTRADA
# ===============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Aurora V7 Real Engines Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5001, help='Port number')
    parser.add_argument('--debug', action='store_true', default=True, help='Debug mode')
    
    args = parser.parse_args()
    
    # Registrar cleanup al salir
    atexit.register(cleanup_old_sessions)
    
    # Ejecutar servidor solo si hay motores reales
    if MOTORES_REALES:
        run_real_engines_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    else:
        print("💥 NO SE PUEDE INICIAR: No hay motores reales disponibles")
        print("🔍 Ejecute primero la verificación de motores")
        exit(1)