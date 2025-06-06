#!/usr/bin/env python3
"""
Aurora V7 - Backend Flask Principal
Sistema neuroacústico modular avanzado
"""

import os
import uuid
import time
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
import logging

from aurora_bridge import AuroraBridge, GenerationStatus

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aurora.Flask.Backend")

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS para desarrollo

# Configuración
app.config.update({
    'SECRET_KEY': 'aurora-v7-development-key',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max
    'AUDIO_FOLDER': 'static/audio',
    'GENERATION_TIMEOUT': 300,  # 5 minutos timeout
})

# Crear carpetas necesarias
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Inicializar Aurora Bridge
aurora_bridge = AuroraBridge()

# Storage para sesiones de generación
generation_sessions = {}
generation_lock = threading.Lock()

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
        
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'status': self.status.value,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'audio_filename': self.audio_filename,
            'metadata': self.metadata,
            'estimated_duration': self.estimated_duration
        }

def cleanup_old_sessions():
    """Limpiar sesiones antiguas y archivos huérfanos"""
    current_time = datetime.now()
    with generation_lock:
        expired_sessions = []
        for session_id, session in generation_sessions.items():
            time_diff = (current_time - session.created_at).total_seconds()
            if time_diff > app.config['GENERATION_TIMEOUT']:
                expired_sessions.append(session_id)
                
                # Eliminar archivo de audio si existe
                if session.audio_filename:
                    audio_path = os.path.join(app.config['AUDIO_FOLDER'], session.audio_filename)
                    if os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                            logger.info(f"Archivo limpiado: {audio_path}")
                        except Exception as e:
                            logger.warning(f"Error limpiando archivo {audio_path}: {e}")
        
        for session_id in expired_sessions:
            del generation_sessions[session_id]
            logger.info(f"Sesión expirada limpiada: {session_id}")

# =======================
# RUTAS PRINCIPALES
# =======================

@app.route('/')
def index():
    """Página principal - Interfaz GUI de Aurora V7"""
    return render_template('index.html')

@app.route('/api/info')
def api_info():
    """Info básica del sistema (movida desde la ruta raíz)"""
    system_status = aurora_bridge.get_complete_system_status()
    return jsonify({
        'message': 'Aurora V7 Backend Activo',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'system_status': system_status['summary'],
        'endpoints': {
            'generate_experience': '/api/generate/experience',
            'system_status': '/api/system/status',
            'generation_status': '/api/generation/{session_id}/status',
            'download_audio': '/api/generation/{session_id}/download'
        }
    })

@app.route('/api/system/status')
def system_status():
    """Estado completo del sistema Aurora"""
    try:
        status = aurora_bridge.get_complete_system_status()
        
        # Agregar info de sesiones activas
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
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        return jsonify({
            'error': 'Error interno del sistema',
            'message': str(e)
        }), 500

@app.route('/api/generate/experience', methods=['POST'])
def generate_experience():
    """Generar experiencia neuroacústica"""
    try:
        # Validar datos de entrada
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400
        
        # Parámetros requeridos
        objetivo = data.get('objetivo', '').strip()
        if not objetivo:
            return jsonify({'error': 'El objetivo es requerido'}), 400
            
        duracion_min = data.get('duracion_min', 20)
        try:
            duracion_min = int(duracion_min)
            if duracion_min < 1 or duracion_min > 120:
                return jsonify({'error': 'Duración debe estar entre 1 y 120 minutos'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Duración debe ser un número válido'}), 400
        
        # Parámetros opcionales con valores por defecto
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
            'validacion_automatica': data.get('validacion_automatica', True)
        }
        
        # Validar parámetros
        validation_result = aurora_bridge.validate_generation_params(params)
        if not validation_result['valid']:
            return jsonify({
                'error': 'Parámetros inválidos',
                'details': validation_result['errors']
            }), 400
        
        # Crear sesión de generación
        session_id = str(uuid.uuid4())
        session = GenerationSession(session_id, params)
        session.estimated_duration = aurora_bridge.estimate_generation_time(params)
        
        with generation_lock:
            generation_sessions[session_id] = session
        
        # Iniciar generación en hilo separado
        generation_thread = threading.Thread(
            target=_generate_audio_async,
            args=(session_id,),
            daemon=True
        )
        generation_thread.start()
        
        logger.info(f"Generación iniciada: {session_id} - {objetivo} ({duracion_min}min)")
        
        return jsonify({
            'session_id': session_id,
            'status': 'pending',
            'estimated_duration_seconds': session.estimated_duration,
            'polling_url': f'/api/generation/{session_id}/status',
            'download_url': f'/api/generation/{session_id}/download'
        }), 202
        
    except Exception as e:
        logger.error(f"Error en generate_experience: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/generation/<session_id>/status')
def generation_status(session_id):
    """Estado de una generación específica"""
    try:
        with generation_lock:
            session = generation_sessions.get(session_id)
        
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
            
        return jsonify(session.to_dict())
        
    except Exception as e:
        logger.error(f"Error obteniendo estado de sesión {session_id}: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/generation/<session_id>/download')
def download_audio(session_id):
    """Descargar audio generado"""
    try:
        with generation_lock:
            session = generation_sessions.get(session_id)
        
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
            
        if session.status != GenerationStatus.COMPLETED:
            return jsonify({
                'error': 'Audio no disponible',
                'status': session.status.value
            }), 400
            
        if not session.audio_filename:
            return jsonify({'error': 'Archivo de audio no encontrado'}), 404
        
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], session.audio_filename)
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Archivo de audio no existe en disco'}), 404
        
        # Nombre descriptivo para descarga
        download_name = f"aurora_v7_{session.params['objetivo']}_{session.params['duracion_min']}min.wav"
        
        return send_file(
            audio_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='audio/wav'
        )
        
    except Exception as e:
        logger.error(f"Error descargando audio {session_id}: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/generation/<session_id>/metadata')
def get_metadata(session_id):
    """Obtener metadatos de la generación"""
    try:
        with generation_lock:
            session = generation_sessions.get(session_id)
        
        if not session:
            return jsonify({'error': 'Sesión no encontrada'}), 404
            
        return jsonify({
            'session_info': session.to_dict(),
            'generation_params': session.params,
            'metadata': session.metadata
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo metadata {session_id}: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

# =======================
# FUNCIONES AUXILIARES
# =======================

def _generate_audio_async(session_id: str):
    """Generar audio de forma asíncrona"""
    try:
        with generation_lock:
            session = generation_sessions.get(session_id)
        
        if not session:
            logger.error(f"Sesión no encontrada para generación: {session_id}")
            return
        
        # Actualizar estado a procesando
        session.status = GenerationStatus.PROCESSING
        session.progress = 10
        
        # Callback para actualizar progreso
        def progress_callback(progress: int, status: str = None):
            session.progress = min(95, max(10, progress))
            if status:
                logger.info(f"Sesión {session_id}: {status} ({progress}%)")
        
        # Generar audio usando Aurora Bridge
        result = aurora_bridge.crear_experiencia_completa(
            session.params,
            progress_callback=progress_callback
        )
        
        if result['success']:
            session.status = GenerationStatus.COMPLETED
            session.progress = 100
            session.completed_at = datetime.now()
            session.audio_filename = result['audio_filename']
            session.metadata = result['metadata']
            
            logger.info(f"Generación completada: {session_id}")
            
        else:
            session.status = GenerationStatus.ERROR
            session.error_message = result['error']
            
            logger.error(f"Error en generación {session_id}: {result['error']}")
            
    except Exception as e:
        logger.error(f"Error crítico en generación {session_id}: {e}")
        with generation_lock:
            session = generation_sessions.get(session_id)
            if session:
                session.status = GenerationStatus.ERROR
                session.error_message = f"Error crítico: {str(e)}"

# =======================
# MANEJO DE ERRORES
# =======================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Método no permitido'}), 405

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'Archivo demasiado grande'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {error}")
    return jsonify({'error': 'Error interno del servidor'}), 500

# =======================
# CLEANUP Y MANTENIMIENTO
# =======================

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

# Función de inicialización reemplazada
def _old_init():
    """Inicialización del sistema"""
    logger.info("🌟 Aurora V7 Backend iniciando...")
    
    # Verificar sistema Aurora
    status = aurora_bridge.get_complete_system_status()
    logger.info(f"Estado Aurora: {status['summary']['status']}")
    logger.info(f"Motores detectados: {status['summary']['motores_activos']}")
    
    # Limpiar archivos de audio antiguos
    cleanup_old_sessions()
    
    logger.info("✅ Aurora V7 Backend listo")

# Cleanup periódico cada 30 minutos
import atexit
cleanup_thread = None

def start_cleanup_thread():
    global cleanup_thread
    if cleanup_thread is None:
        def cleanup_loop():
            while True:
                time.sleep(1800)  # 30 minutos
                cleanup_old_sessions()
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()

atexit.register(cleanup_old_sessions)

# =======================
# DESARROLLO Y DEBUG
# =======================

@app.route('/debug/sessions')
def debug_sessions():
    """Debug: ver todas las sesiones (solo desarrollo)"""
    if not app.debug:
        return jsonify({'error': 'Solo disponible en modo debug'}), 403
    
    with generation_lock:
        sessions_info = {sid: session.to_dict() for sid, session in generation_sessions.items()}
    
    return jsonify({
        'total_sessions': len(sessions_info),
        'sessions': sessions_info
    })

@app.route('/debug/cleanup')
def debug_cleanup():
    """Debug: forzar cleanup (solo desarrollo)"""
    if not app.debug:
        return jsonify({'error': 'Solo disponible en modo debug'}), 403
    
    cleanup_old_sessions()
    return jsonify({'message': 'Cleanup ejecutado'})

# =======================
# MAIN
# =======================

if __name__ == '__main__':
    # Configurar para desarrollo
    app.debug = True
    
    # Iniciar thread de cleanup
    start_cleanup_thread()
    
    # Configuración de desarrollo
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    
    logger.info(f"🚀 Aurora V7 Backend ejecutándose en http://{host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=True,
        threaded=True
    )