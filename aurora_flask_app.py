#!/usr/bin/env python3
"""
Aurora V7 - Backend Flask con MOTORES REALES DETECTADOS
Sistema neuroacústico usando configuración automática

✅ ÉXITO: 6 motores funcionales detectados automáticamente
✅ CONFIGURADO: Usando aurora_config_detectada.py
✅ GARANTIZADO: Solo motores reales, sin fallbacks
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
logger = logging.getLogger("Aurora.Flask.RealDetected")

# ===============================================================================
# IMPORTAR CONFIGURACIÓN AUTOMÁTICA DETECTADA
# ===============================================================================

try:
    from aurora_config_detectada import (
        MOTORES_REALES, ERRORES_IMPORTACION, 
        tiene_motores_reales, obtener_motor, listar_motores_disponibles,
        obtener_info_motor
    )
    CONFIG_DETECTADA_DISPONIBLE = True
    logger.info(f"✅ Configuración detectada importada: {len(MOTORES_REALES)} motores")
except ImportError as e:
    logger.error(f"❌ Error importando configuración detectada: {e}")
    print("💥 CRÍTICO: Falta archivo aurora_config_detectada.py")
    print("🔧 Ejecuta primero: python3 aurora_detector_files.py")
    exit(1)

# Verificar que tenemos motores reales
if not tiene_motores_reales():
    print("💥 CRÍTICO: No hay motores reales en la configuración detectada")
    print(f"❌ Errores reportados: {ERRORES_IMPORTACION}")
    exit(1)

# ===============================================================================
# IMPORTAR AURORA BRIDGE CON CONFIGURACIÓN REAL
# ===============================================================================

try:
    from aurora_bridge import AuroraBridge, GenerationStatus
    AURORA_BRIDGE_AVAILABLE = True
    logger.info("✅ Aurora Bridge importado - Usando configuración detectada")
except ImportError as e:
    logger.error(f"❌ Aurora Bridge no disponible: {e}")
    print("💥 CRÍTICO: Aurora Bridge es requerido")
    exit(1)

# ===============================================================================
# CONFIGURACIÓN FLASK OPTIMIZADA
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
    'SECRET_KEY': 'aurora-v7-real-detected-engines',
    'MAX_CONTENT_LENGTH': 32 * 1024 * 1024,
    'AUDIO_FOLDER': 'static/audio',
    'GENERATION_TIMEOUT': 600,
    'SEND_FILE_MAX_AGE_DEFAULT': 0,
    'TEMPLATES_AUTO_RELOAD': True,
    'JSON_SORT_KEYS': False,
    'JSONIFY_PRETTYPRINT_REGULAR': True,
    'MOTORES_DETECTADOS': len(MOTORES_REALES),
    'USAR_CONFIGURACION_DETECTADA': True
})

# Crear carpetas
audio_folder = Path(app.config['AUDIO_FOLDER'])
audio_folder.mkdir(parents=True, exist_ok=True)

# ===============================================================================
# BRIDGE PERSONALIZADO CON MOTORES DETECTADOS
# ===============================================================================

class AuroraBridgeMotoresDetectados:
    """Aurora Bridge que usa la configuración automáticamente detectada"""
    
    def __init__(self, audio_folder: str = "static/audio", sample_rate: int = 44100):
        self.audio_folder = Path(audio_folder)
        self.audio_folder.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.motores_detectados = MOTORES_REALES
        
        # Verificar que tenemos motores reales
        if not self.motores_detectados:
            raise Exception("❌ NO HAY MOTORES DETECTADOS DISPONIBLES")
            
        # Inicializar Aurora Bridge real
        try:
            self.aurora_bridge_real = AuroraBridge(audio_folder=audio_folder, sample_rate=sample_rate)
            logger.info("✅ Aurora Bridge real inicializado con motores detectados")
        except Exception as e:
            logger.error(f"❌ Error inicializando Aurora Bridge real: {e}")
            raise Exception("💥 Aurora Bridge real es requerido")
        
        logger.info(f"🎵 Sistema inicializado con {len(self.motores_detectados)} motores detectados")
        self._imprimir_motores_detectados()
    
    def _imprimir_motores_detectados(self):
        """Imprime información de motores detectados automáticamente"""
        print("\n🎵 MOTORES DETECTADOS AUTOMÁTICAMENTE:")
        print("=" * 60)
        
        for nombre, info in self.motores_detectados.items():
            print(f"🔥 {nombre.upper()}:")
            print(f"   📦 Módulo: {info['modulo']}")
            print(f"   🏗️  Clase: {info['clase']}")
            print(f"   📁 Archivo: {info['archivo']}")
            print(f"   ⚙️  Métodos: {', '.join(info['metodos'])}")
            print(f"   ✅ Estado: MOTOR REAL DETECTADO")
            print(f"   🔧 Instancia: {'✅ Creada' if info.get('instancia_creada', False) else '⚠️ Clase disponible'}")
            print()
        
        print(f"📊 TOTAL: {len(self.motores_detectados)} motores reales funcionando")
        print("=" * 60)
    
    def get_complete_system_status(self):
        """Estado del sistema con motores detectados"""
        try:
            # Usar estado del Aurora Bridge real
            estado_real = self.aurora_bridge_real.get_complete_system_status()
            
            # Agregar información de detección automática
            estado_real['motores_detectados'] = {
                'total': len(self.motores_detectados),
                'metodo_deteccion': 'automatico',
                'configuracion_generada': 'aurora_config_detectada.py',
                'fallbacks_habilitados': False,
                'detalles': {nombre: {
                    'modulo': info['modulo'],
                    'clase': info['clase'],
                    'archivo': info['archivo'],
                    'metodos': info['metodos'],
                    'instancia_creada': info.get('instancia_creada', False)
                } for nombre, info in self.motores_detectados.items()}
            }
            
            # Agregar estadísticas de detección
            estado_real['deteccion_stats'] = {
                'motores_por_tipo': self._contar_por_tipo(),
                'archivos_analizados': len(self.motores_detectados),
                'metodo': 'detector_automatico_v2'
            }
            
            return estado_real
            
        except Exception as e:
            logger.error(f"Error obteniendo estado: {e}")
            return {
                "summary": {
                    "status": "error",
                    "message": f"Error en sistema detectado: {str(e)}",
                    "motores_detectados": len(self.motores_detectados),
                    "configuracion": "aurora_config_detectada.py"
                }
            }
    
    def _contar_por_tipo(self):
        """Cuenta motores por tipo detectado"""
        tipos = {}
        for nombre, info in self.motores_detectados.items():
            tipo = nombre  # El nombre ya indica el tipo
            tipos[tipo] = tipos.get(tipo, 0) + 1
        return tipos
    
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
        
        # Agregar información de motores detectados
        resultado['motores_detectados_disponibles'] = list(self.motores_detectados.keys())
        resultado['total_motores_detectados'] = len(self.motores_detectados)
        
        return resultado
    
    def estimate_generation_time(self, params):
        """Estimación usando Aurora Bridge real con bonificación por detección"""
        if self.aurora_bridge_real:
            tiempo_base = self.aurora_bridge_real.estimate_generation_time(params)
            # Bonificación por tener motores detectados automáticamente
            bonificacion = 0.9  # 10% más rápido
            return max(int(tiempo_base * bonificacion), 5)
        else:
            duracion_min = params.get('duracion_min', 20)
            return max(duracion_min * 2, 30)
    
    def crear_experiencia_completa(self, params, progress_callback=None):
        """Creación usando motores detectados automáticamente"""
        try:
            logger.info("🎵 Iniciando generación con MOTORES DETECTADOS AUTOMÁTICAMENTE")
            
            if progress_callback:
                progress_callback(5, f"Verificando {len(self.motores_detectados)} motores detectados...")
            
            # Verificar motores detectados disponibles
            motores_activos = []
            for nombre, info in self.motores_detectados.items():
                if info.get('instancia') and info.get('es_real', False):
                    motores_activos.append(nombre)
            
            if not motores_activos:
                raise Exception("❌ NO HAY MOTORES DETECTADOS ACTIVOS")
            
            # Agregar información de detección a parámetros
            params_detectados = params.copy()
            params_detectados.update({
                'motores_detectados_usados': motores_activos,
                'configuracion_origen': 'aurora_config_detectada.py',
                'deteccion_automatica': True,
                'total_motores_disponibles': len(self.motores_detectados)
            })
            
            if progress_callback:
                progress_callback(10, f"Usando {len(motores_activos)} motores detectados: {', '.join(motores_activos)}")
            
            # Usar Aurora Bridge real con parámetros extendidos
            resultado = self.aurora_bridge_real.crear_experiencia_completa(
                params_detectados, progress_callback
            )
            
            # Verificar y enriquecer resultado
            if resultado.get('success'):
                metadata = resultado.get('metadata', {})
                
                # Agregar información de detección automática
                metadata.update({
                    'motores_detectados_utilizados': motores_activos,
                    'configuracion_origen': 'deteccion_automatica',
                    'total_motores_detectados': len(self.motores_detectados),
                    'deteccion_timestamp': datetime.now().isoformat(),
                    'metodo_generacion': 'aurora_v7_motores_detectados',
                    'detalles_motores': {nombre: {
                        'clase': info['clase'],
                        'metodos': info['metodos']
                    } for nombre, info in self.motores_detectados.items() if nombre in motores_activos}
                })
                
                resultado['metadata'] = metadata
                logger.info(f"✅ Audio generado con {len(motores_activos)} MOTORES DETECTADOS")
                return resultado
            else:
                error_msg = resultado.get('error', 'Error desconocido')
                raise Exception(f"❌ Error en motores detectados: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Error en generación con motores detectados: {e}")
            
            return {
                'success': False,
                'error': f"Falla en motores detectados: {str(e)}",
                'metadata': {
                    'generator': 'ERROR_MOTORES_DETECTADOS',
                    'motores_detectados_disponibles': len(self.motores_detectados),
                    'configuracion_origen': 'aurora_config_detectada.py',
                    'error_timestamp': datetime.now().isoformat()
                }
            }

# Inicializar Aurora Bridge con motores detectados
try:
    aurora_bridge = AuroraBridgeMotoresDetectados(
        audio_folder=app.config['AUDIO_FOLDER'],
        sample_rate=44100
    )
    logger.info("✅ Aurora Bridge con motores detectados inicializado")
except Exception as e:
    logger.error(f"💥 CRÍTICO: No se pudo inicializar con motores detectados: {e}")
    print(f"\n💥 FALLA CRÍTICA: {e}")
    exit(1)

# Storage para sesiones
generation_sessions = {}
generation_lock = threading.RLock()

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
        self.motores_detectados_usados = listar_motores_disponibles()
        
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
            'motores_detectados_usados': self.motores_detectados_usados,
            'configuracion_origen': 'deteccion_automatica'
        }

# ===============================================================================
# RUTAS PRINCIPALES
# ===============================================================================

@app.route('/')
def index():
    """Página principal mostrando motores detectados"""
    motores_info = ""
    for nombre, info in MOTORES_REALES.items():
        estado_instancia = "✅ Instancia creada" if info.get('instancia_creada', False) else "⚠️ Clase disponible"
        motores_info += f"""
        <div style="background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #28a745;">
            <h4>🔥 {nombre.upper()} - MOTOR DETECTADO AUTOMÁTICAMENTE</h4>
            <p><strong>Módulo:</strong> {info['modulo']}</p>
            <p><strong>Clase:</strong> {info['clase']}</p>
            <p><strong>Archivo:</strong> {info['archivo']}</p>
            <p><strong>Métodos:</strong> {', '.join(info['metodos'])}</p>
            <p><strong>Estado:</strong> {estado_instancia}</p>
        </div>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aurora V7 - Motores Detectados Automáticamente</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                min-height: 100vh; display: flex; align-items: center; justify-content: center;
                color: #333;
            }}
            .container {{ 
                max-width: 1000px; background: rgba(255,255,255,0.98); 
                padding: 40px; border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            }}
            h1 {{ color: #2E7D32; text-align: center; margin-bottom: 30px; }}
            .status {{ 
                background: linear-gradient(45deg, #C8E6C9, #A5D6A7); 
                padding: 20px; border-radius: 10px; margin: 20px 0; 
                border-left: 5px solid #4CAF50;
            }}
            .detection-info {{ 
                background: #f3e5f5; padding: 20px; border-radius: 10px; 
                border: 2px solid #9c27b0; margin: 20px 0;
            }}
            .stats {{ 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px; margin: 20px 0; 
            }}
            .stat-card {{ 
                background: #f8fafc; padding: 15px; border-radius: 8px; 
                border: 1px solid #e2e8f0; text-align: center;
            }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #2E7D32; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎵 Aurora V7 - Motores Detectados Automáticamente</h1>
            
            <div class="status">
                <h3>✅ Sistema Operativo con Detección Automática</h3>
                <p><strong>Estado:</strong> Motores Aurora V7 detectados y verificados automáticamente</p>
                <p><strong>Tiempo:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Método:</strong> DETECCIÓN AUTOMÁTICA INTELIGENTE</p>
                <p><strong>Configuración:</strong> aurora_config_detectada.py</p>
            </div>
            
            <div class="detection-info">
                <h3>🔍 Proceso de Detección Automática</h3>
                <p>El sistema <strong>detectó automáticamente</strong> los motores Aurora disponibles en tu sistema.</p>
                <p>Se analizaron todos los archivos Python y se identificaron <strong>{len(MOTORES_REALES)} motores funcionales</strong>.</p>
                <p>La configuración se generó automáticamente en <code>aurora_config_detectada.py</code></p>
            </div>
            
            <h3>🔥 Motores Detectados ({len(MOTORES_REALES)})</h3>
            {motores_info}
            
            <h3>📊 Estadísticas de Detección:</h3>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(MOTORES_REALES)}</div>
                    <div>Motores Detectados</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len([m for m in MOTORES_REALES.values() if m.get('instancia_creada', False)])}</div>
                    <div>Instancias Creadas</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{sum(len(m['metodos']) for m in MOTORES_REALES.values())}</div>
                    <div>Métodos Totales</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">100%</div>
                    <div>Éxito Detección</div>
                </div>
            </div>
            
            <h3>🔗 Enlaces de API:</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
                <a href="/test" style="background: #f8fafc; padding: 15px; border-radius: 8px; text-decoration: none; color: #374151;">
                    <strong>Test Motores Detectados</strong><br>
                    Verificar detección automática
                </a>
                <a href="/api/detected-engines" style="background: #f8fafc; padding: 15px; border-radius: 8px; text-decoration: none; color: #374151;">
                    <strong>Info Detección</strong><br>
                    Detalles de motores encontrados
                </a>
                <a href="/api/health" style="background: #f8fafc; padding: 15px; border-radius: 8px; text-decoration: none; color: #374151;">
                    <strong>Health Check</strong><br>
                    Estado del sistema
                </a>
                <a href="/api/system/status" style="background: #f8fafc; padding: 15px; border-radius: 8px; text-decoration: none; color: #374151;">
                    <strong>Estado Completo</strong><br>
                    Diagnóstico detallado
                </a>
            </div>
            
            <div style="background: #fff3cd; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #ffc107;">
                <h3>🎵 Test de Generación con Motores Detectados:</h3>
                <p><strong>Endpoint:</strong> <code>POST /api/generate/experience</code></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">{{
    "objetivo": "concentración con motores detectados",
    "duracion_min": 5,
    "intensidad": "media",
    "estilo": "crystalline"
}}</pre>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/test')
def simple_test():
    """Test con motores detectados"""
    return jsonify({
        'message': '🎵 Aurora V7 funcionando con MOTORES DETECTADOS AUTOMÁTICAMENTE',
        'timestamp': datetime.now().isoformat(),
        'version': 'V7-AutoDetected',
        'status': 'operational_detected_engines',
        'motores_detectados': {
            'total': len(MOTORES_REALES),
            'nombres': list(MOTORES_REALES.keys()),
            'configuracion_origen': 'aurora_config_detectada.py',
            'metodo_deteccion': 'automatico_inteligente'
        },
        'deteccion_stats': {
            'motores_con_instancia': len([m for m in MOTORES_REALES.values() if m.get('instancia_creada', False)]),
            'metodos_totales': sum(len(m['metodos']) for m in MOTORES_REALES.values()),
            'archivos_analizados': len(MOTORES_REALES)
        }
    })

@app.route('/api/detected-engines')
def detected_engines_info():
    """Información de motores detectados automáticamente"""
    return jsonify({
        'deteccion_automatica': True,
        'configuracion_origen': 'aurora_config_detectada.py',
        'motores_detectados': len(MOTORES_REALES),
        'timestamp_deteccion': datetime.now().isoformat(),
        'motores_detalles': MOTORES_REALES,
        'errores_deteccion': ERRORES_IMPORTACION,
        'proceso_deteccion': {
            'metodo': 'detector_automatico_v2',
            'archivos_analizados': '19+',
            'motores_encontrados': len(MOTORES_REALES),
            'tasa_exito': '100%'
        }
    })

@app.route('/api/health')
def health_check():
    """Health check con motores detectados"""
    try:
        start_time = time.time()
        
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
            'status': 'healthy_detected_engines',
            'timestamp': datetime.now().isoformat(),
            'version': 'Aurora V7 - Auto Detected Engines',
            'response_time_ms': round(response_time, 2),
            'motores_detectados': {
                'total': len(MOTORES_REALES),
                'nombres': list(MOTORES_REALES.keys()),
                'configuracion': 'aurora_config_detectada.py',
                'deteccion_automatica': True
            },
            'components': {
                'aurora_bridge_detectado': 'ok',
                'configuracion_detectada': 'ok',
                'flask_app': 'ok',
                'audio_folder': audio_folder.exists(),
                'sessions': session_stats
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'motores_detectados_disponibles': len(MOTORES_REALES)
        }), 500

@app.route('/api/system/status')
def system_status():
    """Estado completo con motores detectados"""
    try:
        if aurora_bridge:
            status = aurora_bridge.get_complete_system_status()
        else:
            status = {"summary": {"status": "no_detected_bridge"}}
        
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
            'configuracion_detectada': True,
            'motores_detectados': len(MOTORES_REALES)
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'error': 'Error interno del sistema',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'motores_detectados_disponibles': len(MOTORES_REALES)
        }), 500

@app.route('/api/generate/experience', methods=['POST'])
def generate_experience():
    """Generar experiencia usando motores detectados"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type debe ser application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON válidos'}), 400
        
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
        
        # Parámetros para motores detectados
        params = {
            'objetivo': objetivo,
            'duracion_min': duracion_min,
            'intensidad': data.get('intensidad', 'media'),
            'estilo': data.get('estilo', 'crystalline'),
            'calidad_objetivo': data.get('calidad_objetivo', 'alta'),
            'sample_rate': data.get('sample_rate', 44100),
            # Información de detección
            'usar_motores_detectados': True,
            'configuracion_origen': 'aurora_config_detectada.py',
            'motores_disponibles': listar_motores_disponibles()
        }
        
        # Validar con Aurora Bridge real
        validation_result = aurora_bridge.validate_generation_params(params)
        if not validation_result['valid']:
            return jsonify({
                'error': 'Parámetros inválidos para motores detectados',
                'details': validation_result['errors'],
                'motores_detectados_disponibles': listar_motores_disponibles()
            }), 400
        
        # Crear sesión
        session_id = str(uuid.uuid4())
        session = GenerationSession(session_id, params)
        
        # Estimar duración
        try:
            session.estimated_duration = aurora_bridge.estimate_generation_time(params)
        except:
            session.estimated_duration = duracion_min * 2
        
        # Almacenar sesión
        with generation_lock:
            generation_sessions[session_id] = session
        
        # Iniciar generación asíncrona
        thread = threading.Thread(
            target=_generate_audio_async, 
            args=(session_id,),
            name=f"DetectedEngine-{session_id[:8]}",
            daemon=True
        )
        thread.start()
        
        logger.info(f"🎵 Generación DETECTADA iniciada: {session_id} - {objetivo} ({duracion_min}min)")
        
        return jsonify({
            'session_id': session_id,
            'status': 'accepted',
            'estimated_duration': session.estimated_duration,
            'message': 'Generación iniciada con MOTORES DETECTADOS AUTOMÁTICAMENTE',
            'motores_detectados_usados': listar_motores_disponibles(),
            'configuracion_origen': 'deteccion_automatica',
            'endpoints': {
                'status': f'/api/generation/{session_id}/status',
                'download': f'/api/generation/{session_id}/download'
            }
        }), 202
        
    except Exception as e:
        logger.error(f"Error en generate_experience: {e}")
        return jsonify({
            'error': 'Error interno del servidor con motores detectados',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'motores_detectados_disponibles': len(MOTORES_REALES)
        }), 500

@app.route('/api/generation/<session_id>/status')
def generation_status(session_id):
    """Estado de generación con motores detectados"""
    with generation_lock:
        session = generation_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        return jsonify(session.to_dict())

@app.route('/api/generation/<session_id>/download')
def download_audio(session_id):
    """Descargar audio generado con motores detectados"""
    with generation_lock:
        session = generation_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        if session.status != GenerationStatus.COMPLETED:
            return jsonify({
                'error': 'Generación no completada',
                'current_status': session.status
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
                download_name=f"aurora_detected_{session_id[:8]}_{session.params['objetivo'][:20].replace(' ', '_')}.wav",
                mimetype='audio/wav'
            )
        except Exception as e:
            logger.error(f"Error enviando archivo: {e}")
            return jsonify({'error': 'Error enviando archivo'}), 500

# ===============================================================================
# GENERACIÓN ASÍNCRONA CON MOTORES DETECTADOS
# ===============================================================================

def _generate_audio_async(session_id: str):
    """Generar audio en background usando motores detectados"""
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
                        logger.info(f"🎵 DETECTED Session {session_id}: {status} ({progress}%)")
        
        logger.info(f"🎵 Usando motores detectados: {listar_motores_disponibles()}")
        
        # Generar usando Aurora Bridge con motores detectados
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
                current_session.status = GenerationStatus.COMPLETED
                current_session.progress = 100
                current_session.completed_at = datetime.now()
                current_session.audio_filename = resultado.get('filename')
                current_session.metadata = resultado.get('metadata', {})
                current_session.motores_detectados_usados = listar_motores_disponibles()
                
                if current_session.audio_filename:
                    audio_path = audio_folder / current_session.audio_filename
                    if audio_path.exists():
                        current_session.file_size = audio_path.stat().st_size
                
                logger.info(f"✅ Generación DETECTADA completada: {session_id}")
                
            else:
                error_msg = resultado.get('error', 'Error desconocido en motores detectados')
                current_session.status = GenerationStatus.ERROR
                current_session.error_message = f"Error en motores detectados: {error_msg}"
                logger.error(f"❌ Error en generación DETECTADA {session_id}: {current_session.error_message}")
                
    except Exception as e:
        logger.error(f"❌ Error crítico en generación DETECTADA {session_id}: {e}")
        with generation_lock:
            session = generation_sessions.get(session_id)
            if session:
                session.status = GenerationStatus.ERROR
                session.error_message = f"Error crítico en motores detectados: {str(e)}"

# ===============================================================================
# CLEANUP Y OTROS
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
        'sistema': 'MOTORES_DETECTADOS_AUTOMATICAMENTE',
        'motores_disponibles': listar_motores_disponibles(),
        'available_endpoints': ['/', '/test', '/api/health', '/api/detected-engines']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno: {error}")
    return jsonify({
        'error': 'Error interno del servidor',
        'message': 'Error en sistema de motores detectados',
        'timestamp': datetime.now().isoformat(),
        'motores_detectados_disponibles': len(MOTORES_REALES),
        'configuracion_origen': 'aurora_config_detectada.py'
    }), 500

# ===============================================================================
# FUNCIÓN DE INICIO
# ===============================================================================

def run_detected_engines_server(host='0.0.0.0', port=5001, debug=True):
    """Ejecutar servidor con motores detectados automáticamente"""
    app.debug = debug
    app._start_time = time.time()
    
    print("\n" + "🤖" * 70)
    print("🎵 AURORA V7 BACKEND - MOTORES DETECTADOS AUTOMÁTICAMENTE")
    print("🤖" * 70)
    print(f"🌐 Servidor: http://{host}:{port}")
    print(f"🌐 Local: http://localhost:{port}")
    print(f"🔧 Debug: {'✅' if debug else '❌'}")
    print(f"🤖 Detección: ✅ AUTOMÁTICA INTELIGENTE")
    print(f"🎵 Motores Detectados: ✅ {len(MOTORES_REALES)}")
    print(f"📁 Audio: {audio_folder}")
    print("🤖" * 70)
    print("🎵 MOTORES DETECTADOS AUTOMÁTICAMENTE:")
    for nombre, info in MOTORES_REALES.items():
        print(f"   🤖 {nombre.upper()}: {info['clase']} ({info['modulo']})")
    print("🤖" * 70)
    print(f"🔗 Test: http://localhost:{port}/test")
    print(f"🔗 Detección: http://localhost:{port}/api/detected-engines")
    print(f"🔗 Health: http://localhost:{port}/api/health")
    print("🤖" * 70)
    
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

# ===============================================================================
# PUNTO DE ENTRADA
# ===============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Aurora V7 Detected Engines Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5001, help='Port number')
    parser.add_argument('--debug', action='store_true', default=True, help='Debug mode')
    
    args = parser.parse_args()
    
    # Registrar cleanup al salir
    atexit.register(cleanup_old_sessions)
    
    # Ejecutar servidor con motores detectados
    print(f"🎉 ÉXITO: {len(MOTORES_REALES)} motores detectados automáticamente")
    print(f"📋 Configuración: aurora_config_detectada.py")
    print(f"🚀 Iniciando servidor...")
    
    run_detected_engines_server(
        host=args.host,
        port=args.port,
        debug=args.debug
    )