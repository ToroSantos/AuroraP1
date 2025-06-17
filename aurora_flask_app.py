#!/usr/bin/env python3
"""
Aurora V7 - Backend Flask Principal Optimizado
Sistema neuroacústico modular avanzado

✅ OPTIMIZADO: Error HTTP 403 corregido + Todas las funcionalidades mejoradas
✅ CORREGIDO: Configuración de host y CORS optimizada
✅ AÑADIDO: Rutas de diagnóstico y manejo robusto de errores
"""

import os
import uuid
import time
import threading
import atexit
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS

# Configurar logging optimizado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Aurora.Flask.Backend")

# Importar Aurora Bridge
try:
    from aurora_bridge import AuroraBridge, GenerationStatus
    AURORA_BRIDGE_AVAILABLE = True
    logger.info("✅ Aurora Bridge importado correctamente")
except ImportError as e:
    logger.error(f"❌ Error importando Aurora Bridge: {e}")
    AURORA_BRIDGE_AVAILABLE = False
    
    # Fallback minimal para casos extremos
    class GenerationStatus:
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        ERROR = "error"
    
    class AuroraBridge:
        def __init__(self):
            self.initialized = False
        
        def get_complete_system_status(self):
            return {"summary": {"status": "error", "message": "Bridge no disponible"}}
        
        def crear_experiencia_completa(self, *args, **kwargs):
            return {"success": False, "error": "Sistema no disponible"}

# ===============================================================================
# CONFIGURACIÓN DE FLASK OPTIMIZADA
# ===============================================================================

# Crear aplicación Flask
app = Flask(__name__)

# ✅ CORRECCIÓN CRÍTICA: CORS optimizado para desarrollo
CORS(app, 
     origins=["http://localhost:5000", "http://127.0.0.1:5000", "http://0.0.0.0:5000", "http://localhost:*"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
     supports_credentials=True
)

# ✅ CONFIGURACIÓN FLASK MEJORADA
app.config.update({
    'SECRET_KEY': 'aurora-v7-development-key-2025',
    'MAX_CONTENT_LENGTH': 32 * 1024 * 1024,  # 32MB max (aumentado)
    'AUDIO_FOLDER': 'static/audio',
    'GENERATION_TIMEOUT': 600,  # 10 minutos timeout (aumentado)
    'SEND_FILE_MAX_AGE_DEFAULT': 0,  # No cache en desarrollo
    'TEMPLATES_AUTO_RELOAD': True,    # Auto-reload templates
    'EXPLAIN_TEMPLATE_LOADING': False, # Reduce logs verbosos
    'JSON_SORT_KEYS': False,          # Mantener orden JSON
    'JSONIFY_PRETTYPRINT_REGULAR': True # JSON bonito en desarrollo
})

# Crear carpetas necesarias
audio_folder = Path(app.config['AUDIO_FOLDER'])
audio_folder.mkdir(parents=True, exist_ok=True)

# Crear carpetas adicionales
(audio_folder.parent / 'logs').mkdir(exist_ok=True)
(audio_folder.parent / 'temp').mkdir(exist_ok=True)

# ===============================================================================
# INICIALIZACIÓN DE AURORA BRIDGE
# ===============================================================================

# Variable global para inicialización única
_system_initialized = False
_initialization_lock = threading.Lock()

def initialize_system_once():
    """Inicializa el sistema Aurora solo una vez de forma thread-safe"""
    global _system_initialized, aurora_bridge
    
    if _system_initialized:
        return
    
    with _initialization_lock:
        if _system_initialized:  # Double-check locking
            return
        
        try:
            logger.info("🌟 Inicializando sistema Aurora V7...")
            
            if AURORA_BRIDGE_AVAILABLE:
                aurora_bridge = AuroraBridge(
                    audio_folder=app.config['AUDIO_FOLDER'],
                    sample_rate=44100
                )
                logger.info("✅ Aurora Bridge inicializado correctamente")
            else:
                aurora_bridge = AuroraBridge()  # Fallback
                logger.warning("⚠️ Aurora Bridge en modo fallback")
            
            _system_initialized = True
            logger.info("✅ Sistema Aurora V7 inicializado completamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema Aurora: {e}")
            # Crear bridge dummy para evitar crashes
            aurora_bridge = AuroraBridge()

# Inicializar Aurora Bridge inmediatamente
aurora_bridge = None
initialize_system_once()

# Storage para sesiones de generación
generation_sessions = {}
generation_lock = threading.RLock()  # RLock para re-entrada

# ===============================================================================
# CLASE GENERATIONSESSION OPTIMIZADA
# ===============================================================================

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
        self.download_count = 0  # Nuevo: tracking de descargas
        self.file_size = 0       # Nuevo: tamaño del archivo
        
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'audio_filename': self.audio_filename,
            'metadata': self.metadata,
            'estimated_duration': self.estimated_duration,
            'download_count': self.download_count,
            'file_size': self.file_size
        }

# ===============================================================================
# FUNCIONES DE LIMPIEZA OPTIMIZADAS
# ===============================================================================

def cleanup_old_sessions():
    """Limpiar sesiones antiguas y archivos huérfanos - versión optimizada"""
    try:
        current_time = datetime.now()
        with generation_lock:
            expired_sessions = []
            total_size_cleaned = 0
            
            for session_id, session in generation_sessions.items():
                time_diff = (current_time - session.created_at).total_seconds()
                if time_diff > app.config['GENERATION_TIMEOUT']:
                    expired_sessions.append(session_id)
                    
                    # Eliminar archivo de audio si existe
                    if session.audio_filename:
                        audio_path = audio_folder / session.audio_filename
                        if audio_path.exists():
                            try:
                                file_size = audio_path.stat().st_size
                                audio_path.unlink()
                                total_size_cleaned += file_size
                                logger.info(f"Archivo limpiado: {session.audio_filename} ({file_size/1024/1024:.1f}MB)")
                            except Exception as e:
                                logger.warning(f"Error limpiando archivo {audio_path}: {e}")
            
            # Limpiar sesiones del diccionario
            for session_id in expired_sessions:
                del generation_sessions[session_id]
            
            if expired_sessions:
                logger.info(f"🧹 Limpieza completada: {len(expired_sessions)} sesiones, {total_size_cleaned/1024/1024:.1f}MB liberados")
            
            # Limpiar archivos huérfanos (sin sesión asociada)
            try:
                audio_files = set(f.name for f in audio_folder.glob("*.wav"))
                session_files = set(s.audio_filename for s in generation_sessions.values() if s.audio_filename)
                orphaned_files = audio_files - session_files
                
                for orphaned_file in orphaned_files:
                    try:
                        (audio_folder / orphaned_file).unlink()
                        logger.info(f"Archivo huérfano eliminado: {orphaned_file}")
                    except Exception as e:
                        logger.warning(f"Error eliminando archivo huérfano {orphaned_file}: {e}")
                        
            except Exception as e:
                logger.warning(f"Error en limpieza de archivos huérfanos: {e}")
                
    except Exception as e:
        logger.error(f"Error en cleanup de sesiones: {e}")

# ===============================================================================
# MIDDLEWARE Y HOOKS
# ===============================================================================

@app.before_request
def ensure_initialized():
    """Asegura que el sistema esté inicializado antes de cada request - versión mejorada"""
    try:
        # Solo inicializar para rutas que no son archivos estáticos
        if request.endpoint and not request.endpoint.startswith('static'):
            initialize_system_once()
            
        # Log de requests en debug
        if app.debug and request.endpoint:
            logger.debug(f"📨 {request.method} {request.path} desde {request.remote_addr}")
            
    except Exception as e:
        logger.warning(f"⚠️ Error en middleware de inicialización: {e}")
        # No bloquear el request por errores de inicialización

@app.after_request
def after_request(response):
    """Añadir headers de seguridad y desarrollo"""
    if app.debug:
        response.headers['X-Aurora-Version'] = 'V7.2-Optimized'
        response.headers['X-Debug-Mode'] = 'enabled'
    
    # Headers de seguridad básicos
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    
    return response

# ===============================================================================
# RUTAS PRINCIPALES OPTIMIZADAS
# ===============================================================================

@app.route('/')
def index():
    """Página principal - Interfaz GUI de Aurora V7 con fallback robusto"""
    try:
        # Verificar si existe el template
        template_path = Path(app.template_folder or 'templates') / 'index.html'
        
        if template_path.exists():
            return render_template('index.html')
        else:
            # ✅ FALLBACK: Página HTML completa si no existe template
            return f"""
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Aurora V7 - Sistema Neuroacústico</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{ 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh; display: flex; align-items: center; justify-content: center;
                    }}
                    .container {{ 
                        max-width: 900px; background: rgba(255,255,255,0.95); 
                        padding: 40px; border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                    }}
                    h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                    .status {{ 
                        background: linear-gradient(45deg, #a8e6cf, #dcedc1); 
                        padding: 20px; border-radius: 10px; margin: 20px 0; 
                        border-left: 5px solid #27ae60;
                    }}
                    .api-grid {{ 
                        display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                        gap: 15px; margin: 20px 0; 
                    }}
                    .api-link {{ 
                        background: #f8f9fa; padding: 15px; border-radius: 8px; 
                        border: 1px solid #e9ecef; transition: all 0.3s ease;
                    }}
                    .api-link:hover {{ background: #e9ecef; transform: translateY(-2px); }}
                    .api-link a {{ color: #495057; text-decoration: none; font-weight: 500; }}
                    .api-link a:hover {{ color: #007bff; }}
                    .test-section {{ 
                        background: #fff3cd; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #ffc107; margin: 20px 0;
                    }}
                    .footer {{ text-align: center; margin-top: 30px; color: #6c757d; }}
                    .status-indicator {{ 
                        display: inline-block; width: 12px; height: 12px; 
                        border-radius: 50%; background: #28a745; margin-right: 8px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎵 Aurora V7 - Sistema Neuroacústico</h1>
                    
                    <div class="status">
                        <h3><span class="status-indicator"></span>Sistema Operativo</h3>
                        <p><strong>Estado:</strong> El backend de Aurora V7 está funcionando correctamente</p>
                        <p><strong>Tiempo:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Versión:</strong> V7.2 Optimizada</p>
                    </div>
                    
                    <h3>🔗 Enlaces de API:</h3>
                    <div class="api-grid">
                        <div class="api-link">
                            <strong>Estado del Sistema:</strong><br>
                            <a href="/api/system/status" target="_blank">/api/system/status</a>
                        </div>
                        <div class="api-link">
                            <strong>Información General:</strong><br>
                            <a href="/api/info" target="_blank">/api/info</a>
                        </div>
                        <div class="api-link">
                            <strong>Test Simple:</strong><br>
                            <a href="/test" target="_blank">/test</a>
                        </div>
                        <div class="api-link">
                            <strong>Health Check:</strong><br>
                            <a href="/api/health" target="_blank">/api/health</a>
                        </div>
                    </div>
                    
                    <div class="test-section">
                        <h3>📝 Para generar audio neuroacústico:</h3>
                        <p><strong>Endpoint:</strong> <code>POST /api/generate/experience</code></p>
                        <p><strong>Ejemplo JSON:</strong></p>
                        <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">{{
    "objetivo": "relajación profunda",
    "duracion_min": 20,
    "intensidad": "media",
    "estilo": "sereno"
}}</pre>
                    </div>
                    
                    <div class="footer">
                        <p>⚠️ <strong>Nota:</strong> Template 'index.html' no encontrado. Usando página de fallback.</p>
                        <p>Aurora V7 - Sistema Neuroacústico Avanzado | 2025</p>
                    </div>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error en ruta principal: {e}")
        return f"""
        <h1>Error en Aurora V7</h1>
        <p>Error: {str(e)}</p>
        <p><a href="/api/health">Verificar estado del sistema</a></p>
        """, 500

# ===============================================================================
# RUTAS DE DIAGNÓSTICO Y SALUD
# ===============================================================================

@app.route('/test')
def simple_test():
    """Ruta de test súper simple para verificar conectividad"""
    return jsonify({
        'message': '¡Aurora V7 está funcionando perfectamente! ✅',
        'timestamp': datetime.now().isoformat(),
        'version': 'V7.2-Optimized',
        'status': 'operational',
        'server_info': {
            'host': request.host,
            'remote_addr': request.remote_addr,
            'method': request.method,
            'user_agent': request.headers.get('User-Agent', 'Unknown')[:100]
        }
    })

@app.route('/api/health')
def health_check():
    """Health check completo del sistema"""
    try:
        start_time = time.time()
        
        # Verificar Aurora Bridge
        bridge_status = 'ok' if aurora_bridge and hasattr(aurora_bridge, 'initialized') else 'error'
        
        # Verificar carpetas
        folders_status = {
            'audio_folder': audio_folder.exists(),
            'logs_folder': (audio_folder.parent / 'logs').exists(),
            'temp_folder': (audio_folder.parent / 'temp').exists()
        }
        
        # Estadísticas de sesiones
        with generation_lock:
            session_stats = {
                'total_sessions': len(generation_sessions),
                'active_sessions': len([s for s in generation_sessions.values() 
                                      if s.status in [GenerationStatus.PENDING, GenerationStatus.PROCESSING]]),
                'completed_sessions': len([s for s in generation_sessions.values() 
                                         if s.status == GenerationStatus.COMPLETED]),
                'failed_sessions': len([s for s in generation_sessions.values() 
                                      if s.status == GenerationStatus.ERROR])
            }
        
        # Información del sistema
        response_time = (time.time() - start_time) * 1000  # ms
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': 'Aurora V7.2 Optimized',
            'uptime': f"{time.time() - app._start_time:.1f}s" if hasattr(app, '_start_time') else 'unknown',
            'response_time_ms': round(response_time, 2),
            'components': {
                'aurora_bridge': bridge_status,
                'flask_app': 'ok',
                'folders': folders_status,
                'sessions': session_stats
            },
            'debug_mode': app.debug,
            'config': {
                'audio_folder': str(audio_folder),
                'max_content_length': app.config.get('MAX_CONTENT_LENGTH'),
                'generation_timeout': app.config.get('GENERATION_TIMEOUT')
            }
        }
        
        return jsonify(health_data)
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'version': 'Aurora V7.2 Optimized'
        }), 500

@app.route('/api/info')
def api_info():
    """Información detallada del sistema"""
    try:
        system_status = aurora_bridge.get_complete_system_status()
        
        return jsonify({
            'message': 'Aurora V7 Backend Activo',
            'version': 'V7.2 Optimized',
            'timestamp': datetime.now().isoformat(),
            'system_status': system_status.get('summary', {}),
            'endpoints': {
                'generate_experience': '/api/generate/experience',
                'system_status': '/api/system/status',
                'generation_status': '/api/generation/{session_id}/status',
                'download_audio': '/api/generation/{session_id}/download',
                'health_check': '/api/health',
                'simple_test': '/test'
            },
            'features': [
                'Generación neuroacústica avanzada',
                'Múltiples motores de audio',
                'Análisis Carmine integrado',
                'Pipeline de calidad',
                'Sistema de fallback robusto',
                'API REST completa',
                'Monitoreo en tiempo real'
            ]
        })
    except Exception as e:
        logger.error(f"Error en api_info: {e}")
        return jsonify({
            'message': 'Aurora V7 Backend (Modo Básico)',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ===============================================================================
# RUTAS DE API PRINCIPALES
# ===============================================================================

@app.route('/api/system/status')
def system_status():
    """Estado completo del sistema Aurora"""
    try:
        status = aurora_bridge.get_complete_system_status()
        status = aurora_bridge._clean_metadata(status)
        
        # Agregar estadísticas de sesiones
        with generation_lock:
            active_sessions = len([s for s in generation_sessions.values() 
                                 if s.status in [GenerationStatus.PENDING, GenerationStatus.PROCESSING]])
            completed_sessions = len([s for s in generation_sessions.values() 
                                    if s.status == GenerationStatus.COMPLETED])
            failed_sessions = len([s for s in generation_sessions.values() 
                                 if s.status == GenerationStatus.ERROR])
        
        status['session_stats'] = {
            'active_sessions': active_sessions,
            'completed_sessions': completed_sessions,
            'failed_sessions': failed_sessions,
            'total_sessions': len(generation_sessions)
        }
        
        # Agregar información del servidor
        status['server_info'] = {
            'timestamp': datetime.now().isoformat(),
            'debug_mode': app.debug,
            'audio_files_count': len(list(audio_folder.glob("*.wav"))),
            'audio_folder_size_mb': sum(f.stat().st_size for f in audio_folder.glob("*.wav")) / 1024 / 1024
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        return jsonify({
            'error': 'Error interno del sistema',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/generate/experience', methods=['POST'])
def generate_experience():
    """Generar experiencia neuroacústica - versión optimizada"""
    try:
        # Validar Content-Type
        if not request.is_json:
            return jsonify({'error': 'Content-Type debe ser application/json'}), 400
        
        # Validar datos de entrada
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON válidos'}), 400
        
        # Parámetros requeridos
        objetivo = data.get('objetivo', '').strip()
        if not objetivo:
            return jsonify({'error': 'El campo "objetivo" es requerido'}), 400
            
        duracion_min = data.get('duracion_min', 20)
        try:
            duracion_min = int(duracion_min)
            if duracion_min < 1 or duracion_min > 180:
                return jsonify({'error': 'Duración debe estar entre 1 y 180 minutos'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Duración debe ser un número válido'}), 400
        
        # Parámetros optimizados con valores por defecto mejorados
        params = {
            'objetivo': objetivo,
            'duracion_min': duracion_min,
            'intensidad': data.get('intensidad', 'media'),
            'estilo': data.get('estilo', 'sereno'),
            'calidad_objetivo': data.get('calidad_objetivo', 'alta'),
            'normalizar': data.get('normalizar', True),
            'aplicar_mastering': data.get('aplicar_mastering', True),
            'exportar_wav': True,
            'incluir_metadatos': True,
            'sample_rate': data.get('sample_rate', 44100),
            'neurotransmisor_preferido': data.get('neurotransmisor_preferido'),
            'contexto_uso': data.get('contexto_uso', 'general'),
            'modo_orquestacion': data.get('modo_orquestacion', 'hybrid'),
            'estrategia_preferida': data.get('estrategia_preferida'),
            'motores_preferidos': data.get('motores_preferidos', []),
            'usar_objective_manager': data.get('usar_objective_manager', True),
            'validacion_automatica': data.get('validacion_automatica', True),
            'usar_carmine_optimization': data.get('usar_carmine_optimization', True),
            'aplicar_quality_pipeline': data.get('aplicar_quality_pipeline', True)
        }
        
        # Validar parámetros con Aurora Bridge
        validation_result = aurora_bridge.validate_generation_params(params)
        if not validation_result['valid']:
            return jsonify({
                'error': 'Parámetros inválidos',
                'details': validation_result['errors'],
                'warnings': validation_result.get('warnings', [])
            }), 400
        
        # Crear sesión de generación
        session_id = str(uuid.uuid4())
        session = GenerationSession(session_id, params)
        
        # Estimar duración
        try:
            session.estimated_duration = aurora_bridge.estimate_generation_time(params)
        except Exception as e:
            logger.warning(f"Error estimando duración: {e}")
            session.estimated_duration = 60  # Fallback
        
        # Almacenar sesión
        with generation_lock:
            generation_sessions[session_id] = session
        
        # Iniciar generación asíncrona
        thread = threading.Thread(
            target=_generate_audio_async, 
            args=(session_id,),
            name=f"AudioGen-{session_id[:8]}",
            daemon=True
        )
        thread.start()
        
        logger.info(f"🎵 Generación iniciada: {session_id} - {objetivo} ({duracion_min}min)")
        
        return jsonify({
            'session_id': session_id,
            'status': 'accepted',
            'estimated_duration': session.estimated_duration,
            'message': 'Generación iniciada exitosamente',
            'endpoints': {
                'status': f'/api/generation/{session_id}/status',
                'download': f'/api/generation/{session_id}/download',
                'metadata': f'/api/generation/{session_id}/metadata'
            }
        }), 202
        
    except Exception as e:
        logger.error(f"Error en generate_experience: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/generation/<session_id>/status')
def generation_status(session_id):
    """Estado de una sesión de generación específica"""
    with generation_lock:
        session = generation_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        return jsonify(session.to_dict())

@app.route('/api/generation/<session_id>/download')
def download_audio(session_id):
    """Descargar archivo de audio generado"""
    with generation_lock:
        session = generation_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        if session.status != GenerationStatus.COMPLETED:
            return jsonify({
                'error': 'Generación no completada',
                'current_status': session.status.value if hasattr(session.status, 'value') else str(session.status)
            }), 400
        
        if not session.audio_filename:
            return jsonify({'error': 'Archivo de audio no disponible'}), 404
        
        audio_path = audio_folder / session.audio_filename
        if not audio_path.exists():
            return jsonify({'error': 'Archivo de audio no encontrado en el sistema'}), 404
        
        # Actualizar contador de descargas
        session.download_count += 1
        
        try:
            return send_file(
                audio_path,
                as_attachment=True,
                download_name=f"aurora_{session_id[:8]}_{session.params['objetivo'][:20].replace(' ', '_')}.wav",
                mimetype='audio/wav'
            )
        except Exception as e:
            logger.error(f"Error enviando archivo: {e}")
            return jsonify({'error': 'Error enviando archivo'}), 500

@app.route('/api/generation/<session_id>/metadata')
def generation_metadata(session_id):
    """Obtener metadatos detallados de la generación"""
    with generation_lock:
        session = generation_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        metadata = session.to_dict()
        
        # Añadir información adicional si está disponible
        if session.audio_filename:
            audio_path = audio_folder / session.audio_filename
            if audio_path.exists():
                stat = audio_path.stat()
                metadata['file_info'] = {
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
        
        return jsonify(metadata)

# ===============================================================================
# FUNCIÓN DE GENERACIÓN ASÍNCRONA OPTIMIZADA
# ===============================================================================

def _generate_audio_async(session_id: str):
    """Genera audio en background con manejo robusto de errores"""
    try:
        with generation_lock:
            session = generation_sessions.get(session_id)
            if not session:
                logger.error(f"Sesión {session_id} no encontrada para generación")
                return
        
        logger.info(f"🎵 Iniciando generación asíncrona: {session_id}")
        session.status = GenerationStatus.PROCESSING
        
        # Callback de progreso thread-safe
        def progress_callback(progress: int, status: str = None):
            with generation_lock:
                current_session = generation_sessions.get(session_id)
                if current_session:
                    current_session.progress = min(95, max(10, progress))
                    if status:
                        logger.info(f"Sesión {session_id}: {status} ({progress}%)")
        
        # Generar audio usando Aurora Bridge
        result = aurora_bridge.crear_experiencia_completa(
            session.params,
            progress_callback=progress_callback
        )
        
        # Procesar resultado
        with generation_lock:
            current_session = generation_sessions.get(session_id)
            if not current_session:
                return
            
            if result.get('success', False):
                current_session.status = GenerationStatus.COMPLETED
                current_session.progress = 100
                current_session.completed_at = datetime.now()
                current_session.audio_filename = result.get('filename')
                current_session.metadata = result.get('metadata', {})
                
                # Actualizar tamaño del archivo
                if current_session.audio_filename:
                    audio_path = audio_folder / current_session.audio_filename
                    if audio_path.exists():
                        current_session.file_size = audio_path.stat().st_size
                
                logger.info(f"✅ Generación completada: {session_id}")
                
            else:
                current_session.status = GenerationStatus.ERROR
                current_session.error_message = result.get('error', 'Error desconocido en generación')
                logger.error(f"❌ Error en generación {session_id}: {current_session.error_message}")
                
    except Exception as e:
        logger.error(f"❌ Error crítico en generación {session_id}: {e}")
        with generation_lock:
            session = generation_sessions.get(session_id)
            if session:
                session.status = GenerationStatus.ERROR
                session.error_message = f"Error crítico: {str(e)}"

# ===============================================================================
# RUTAS DE DESARROLLO Y DEBUG
# ===============================================================================

@app.route('/debug/sessions')
def debug_sessions():
    """Debug: ver todas las sesiones activas"""
    if not app.debug:
        return jsonify({'error': 'Solo disponible en modo debug'}), 403
    
    with generation_lock:
        sessions_info = {sid: session.to_dict() for sid, session in generation_sessions.items()}
    
    return jsonify({
        'total_sessions': len(sessions_info),
        'sessions': sessions_info,
        'server_info': {
            'audio_folder': str(audio_folder),
            'files_in_folder': len(list(audio_folder.glob("*.wav"))),
            'folder_size_mb': sum(f.stat().st_size for f in audio_folder.glob("*.wav")) / 1024 / 1024
        }
    })

@app.route('/debug/cleanup')
def debug_cleanup():
    """Debug: forzar cleanup manual"""
    if not app.debug:
        return jsonify({'error': 'Solo disponible en modo debug'}), 403
    
    cleanup_old_sessions()
    return jsonify({
        'message': 'Cleanup ejecutado exitosamente',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/debug/system')
def debug_system():
    """Debug: información detallada del sistema"""
    if not app.debug:
        return jsonify({'error': 'Solo disponible en modo debug'}), 403
    
    try:
        import psutil
        system_info = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
    except ImportError:
        system_info = {'note': 'psutil no disponible'}
    
    return jsonify({
        'flask_config': dict(app.config),
        'system_info': system_info,
        'aurora_bridge_status': {
            'available': AURORA_BRIDGE_AVAILABLE,
            'initialized': hasattr(aurora_bridge, 'initialized') and aurora_bridge.initialized if aurora_bridge else False
        },
        'threading_info': {
            'active_threads': threading.active_count(),
            'current_thread': threading.current_thread().name
        }
    })

# ===============================================================================
# MANEJO DE ERRORES OPTIMIZADO
# ===============================================================================

@app.errorhandler(403)
def forbidden(error):
    """Manejo específico de errores 403 - mejorado"""
    logger.warning(f"🚫 Acceso denegado 403: {request.url} desde {request.remote_addr}")
    return jsonify({
        'error': 'Acceso denegado',
        'message': 'No tienes permisos para acceder a este recurso',
        'timestamp': datetime.now().isoformat(),
        'requested_url': request.url,
        'method': request.method,
        'suggestions': [
            'Verifica que el servidor esté en modo debug para rutas de desarrollo',
            'Intenta acceder a /api/health para verificar conectividad',
            'Revisa los logs del servidor para más detalles',
            'Consulta la documentación de la API en /api/info'
        ],
        'available_endpoints': ['/api/health', '/test', '/api/info', '/api/system/status']
    }), 403

@app.errorhandler(404)
def not_found(error):
    """Manejo de errores 404 mejorado"""
    return jsonify({
        'error': 'Endpoint no encontrado',
        'message': f'La ruta "{request.path}" no existe',
        'timestamp': datetime.now().isoformat(),
        'available_endpoints': {
            'main': ['/', '/test', '/api/health', '/api/info'],
            'api': ['/api/system/status', '/api/generate/experience'],
            'debug': ['/debug/sessions', '/debug/cleanup'] if app.debug else []
        }
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Manejo de errores 405"""
    return jsonify({
        'error': 'Método no permitido',
        'message': f'El método {request.method} no está permitido para {request.path}',
        'timestamp': datetime.now().isoformat()
    }), 405

@app.errorhandler(413)
def too_large(error):
    """Manejo de errores 413"""
    return jsonify({
        'error': 'Archivo demasiado grande',
        'message': f'El tamaño máximo permitido es {app.config["MAX_CONTENT_LENGTH"] // 1024 // 1024}MB',
        'timestamp': datetime.now().isoformat()
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Manejo de errores 500 mejorado"""
    logger.error(f"💥 Error interno del servidor: {error}")
    return jsonify({
        'error': 'Error interno del servidor',
        'message': 'Se produjo un error inesperado en el servidor',
        'timestamp': datetime.now().isoformat(),
        'support': 'Revisa los logs del servidor para más detalles'
    }), 500

# ===============================================================================
# CLEANUP Y MANTENIMIENTO
# ===============================================================================

# Thread de cleanup periódico
cleanup_thread = None

def start_cleanup_thread():
    """Iniciar thread de cleanup automático"""
    global cleanup_thread
    if cleanup_thread is None or not cleanup_thread.is_alive():
        def cleanup_loop():
            while True:
                try:
                    time.sleep(1800)  # 30 minutos
                    cleanup_old_sessions()
                except Exception as e:
                    logger.error(f"Error en cleanup automático: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True, name="CleanupThread")
        cleanup_thread.start()
        logger.info("🧹 Thread de cleanup automático iniciado")

# Registrar cleanup al salir
atexit.register(cleanup_old_sessions)

# ===============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ===============================================================================

if __name__ == '__main__':
    # ✅ CONFIGURACIÓN CRÍTICA CORREGIDA
    app.debug = True
    app._start_time = time.time()  # Para tracking de uptime
    
    # Iniciar thread de cleanup
    start_cleanup_thread()
    
    # ✅ CORRECCIÓN PRINCIPAL: Configuración de host y puerto optimizada
    # Puerto por defecto 5001 para evitar conflicto con AirPlay Receiver en macOS
    port = int(os.environ.get('PORT', 5001))  # ← CAMBIO: de 5000 a 5001 (evita AirPlay)
    host = os.environ.get('HOST', '0.0.0.0')  # ← CAMBIO CRÍTICO: de '127.0.0.1' a '0.0.0.0'
    
    # Log de inicio mejorado
    logger.info("=" * 60)
    logger.info("🚀 AURORA V7 BACKEND - INICIANDO")
    logger.info("=" * 60)
    logger.info(f"🌐 Servidor ejecutándose en http://{host}:{port}")
    logger.info(f"🌐 También accesible desde:")
    logger.info(f"   • http://localhost:{port}")
    logger.info(f"   • http://127.0.0.1:{port}")
    if host == '0.0.0.0':
        logger.info(f"   • http://[tu-ip-local]:{port}")
    logger.info(f"🔧 Modo debug: {'Activado' if app.debug else 'Desactivado'}")
    logger.info(f"📁 Carpeta de audio: {audio_folder}")
    logger.info(f"🎵 Aurora Bridge: {'Disponible' if AURORA_BRIDGE_AVAILABLE else 'Modo Fallback'}")
    logger.info(f"💡 Nota: Puerto {port} elegido para evitar conflictos con AirPlay Receiver")
    logger.info("=" * 60)
    
    # ✅ CONFIGURACIÓN FLASK OPTIMIZADA
    try:
        app.run(
            host=host,
            port=port,
            debug=True,
            threaded=True,
            use_reloader=False  # Evita problemas de doble inicialización
        )
    except Exception as e:
        logger.error(f"❌ Error iniciando servidor Flask: {e}")
        print(f"\n💡 SOLUCIONES PARA PROBLEMAS DE PUERTO:")
        print(f"   1. Puerto {port} ocupado? Prueba: PORT=5002 python aurora_flask_app.py")
        print(f"   2. En macOS: Desactiva AirPlay Receiver en Configuraciones > Compartir")
        print(f"   3. Ver qué usa el puerto: lsof -i :{port}")
        print(f"   4. Ejecuta como administrador si es necesario")
        print(f"   5. Verifica firewall/antivirus")
        print(f"   6. Prueba otros puertos: 8000, 3000, 8080")
        print(f"\n🔧 COMANDOS ÚTILES:")
        print(f"   • PORT=8000 python aurora_flask_app.py")
        print(f"   • lsof -i :{port}")
        print(f"   • kill -9 [PID]")
        print(f"\n💻 URLs de acceso una vez funcionando:")
        print(f"   • http://localhost:{port}/test")
        print(f"   • http://127.0.0.1:{port}/api/health")
