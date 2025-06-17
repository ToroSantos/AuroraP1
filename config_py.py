#!/usr/bin/env python3
"""
Aurora V7 Backend - Configuración del Sistema
"""

import os
from pathlib import Path

class Config:
    """Configuración base"""
    
    # Configuración de la aplicación
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'aurora-v7-development-key-change-in-production'
    
    # Configuración de Flask
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    TESTING = False
    
    # Configuración de archivos
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    AUDIO_FOLDER = os.environ.get('AUDIO_FOLDER') or 'static/audio'
    TEMP_FOLDER = os.environ.get('TEMP_FOLDER') or 'temp'
    
    # Configuración de audio
    SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', '44100'))
    AUDIO_FORMAT = 'wav'
    AUDIO_CHANNELS = 2
    AUDIO_BIT_DEPTH = 16
    
    # Configuración de generación
    GENERATION_TIMEOUT = int(os.environ.get('GENERATION_TIMEOUT', '300'))  # 5 minutos
    MAX_DURATION_MINUTES = int(os.environ.get('MAX_DURATION_MINUTES', '120'))  # 2 horas
    MIN_DURATION_MINUTES = int(os.environ.get('MIN_DURATION_MINUTES', '1'))
    
    # Configuración de sesiones
    SESSION_CLEANUP_INTERVAL = int(os.environ.get('SESSION_CLEANUP_INTERVAL', '1800'))  # 30 min
    MAX_CONCURRENT_GENERATIONS = int(os.environ.get('MAX_CONCURRENT_GENERATIONS', '3'))
    
    # Configuración Aurora
    AURORA_SYSTEM_PATH = os.environ.get('AURORA_SYSTEM_PATH') or 'aurora_system'
    AURORA_FALLBACK_ENABLED = os.environ.get('AURORA_FALLBACK_ENABLED', 'True').lower() == 'true'
    AURORA_LOG_LEVEL = os.environ.get('AURORA_LOG_LEVEL', 'INFO')
    
    # Configuración de red
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', '5000'))
    
    @staticmethod
    def init_app(app):
        """Inicializar configuración de la aplicación"""
        # Crear directorios necesarios
        os.makedirs(Config.AUDIO_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
        
        # Configurar logging según el nivel
        import logging
        level = getattr(logging, Config.AURORA_LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(level=level)

class DevelopmentConfig(Config):
    """Configuración de desarrollo"""
    DEBUG = True
    AURORA_LOG_LEVEL = 'DEBUG'
    
    # Configuraciones más permisivas para desarrollo
    GENERATION_TIMEOUT = 600  # 10 minutos
    SESSION_CLEANUP_INTERVAL = 900  # 15 minutos
    
class ProductionConfig(Config):
    """Configuración de producción"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Configuraciones más estrictas para producción
    MAX_CONCURRENT_GENERATIONS = 5
    GENERATION_TIMEOUT = 180  # 3 minutos
    SESSION_CLEANUP_INTERVAL = 600  # 10 minutos
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Configuración adicional para producción
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug:
            file_handler = RotatingFileHandler(
                'logs/aurora_backend.log',
                maxBytes=10240000,  # 10MB
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            
            app.logger.setLevel(logging.INFO)
            app.logger.info('Aurora V7 Backend startup')

class TestingConfig(Config):
    """Configuración para testing"""
    TESTING = True
    DEBUG = True
    AURORA_FALLBACK_ENABLED = True
    
    # Configuraciones para tests
    AUDIO_FOLDER = 'tests/audio'
    TEMP_FOLDER = 'tests/temp'
    GENERATION_TIMEOUT = 60  # 1 minuto para tests
    MAX_DURATION_MINUTES = 5  # Máximo 5 minutos en tests

# Mapeo de configuraciones
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Obtener configuración basada en variable de entorno"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])

# Configuración específica de Aurora V7
class AuroraConfig:
    """Configuración específica del sistema Aurora"""
    
    # Motores disponibles
    AVAILABLE_ENGINES = [
        'neuromix_v27',
        'hypermod_v32', 
        'harmonic_essence_v34'
    ]
    
    # Objetivos predefinidos
    PREDEFINED_OBJECTIVES = [
        'concentracion', 'claridad_mental', 'enfoque', 'relajacion', 'meditacion',
        'creatividad', 'gratitud', 'energia', 'calma', 'sanacion', 'sueño',
        'ansiedad', 'estres', 'expansion_consciencia', 'conexion_espiritual',
        'flujo_creativo', 'inspiracion', 'alegria', 'amor', 'compasion',
        'equilibrio_emocional', 'autoestima', 'confianza', 'determinacion'
    ]
    
    # Estilos disponibles
    AVAILABLE_STYLES = [
        'sereno', 'crystalline', 'organico', 'etereo', 'tribal',
        'mistico', 'cuantico', 'neural', 'neutro'
    ]
    
    # Intensidades
    INTENSITY_LEVELS = ['suave', 'media', 'intenso']
    
    # Calidades
    QUALITY_LEVELS = ['basica', 'media', 'alta', 'maxima']
    
    # Neurotransmisores
    NEUROTRANSMITTERS = [
        'dopamina', 'serotonina', 'gaba', 'acetilcolina', 'oxitocina',
        'anandamida', 'endorfina', 'bdnf', 'adrenalina', 'norepinefrina', 'melatonina'
    ]
    
    # Sample rates soportados
    SUPPORTED_SAMPLE_RATES = [22050, 44100, 48000]
    
    # Configuración de fallback
    FALLBACK_CONFIG = {
        'enabled': True,
        'frequency_map': {
            'concentracion': 14.0,
            'claridad_mental': 12.0,
            'enfoque': 15.0,
            'relajacion': 6.0,
            'meditacion': 7.83,
            'creatividad': 10.0,
            'gratitud': 8.0,
            'energia': 16.0,
            'calma': 5.0,
            'sanacion': 528.0
        },
        'default_frequency': 10.0,
        'fade_duration_seconds': 3.0,
        'harmonic_levels': [0.3, 0.1]  # Amplitudes de armónicos
    }

# Validaciones
class ValidationConfig:
    """Configuración de validaciones"""
    
    # Límites de parámetros
    OBJECTIVE_MAX_LENGTH = 200
    MIN_DURATION_SECONDS = 10
    MAX_DURATION_SECONDS = 7200  # 2 horas
    
    # Patrones de validación
    SAFE_FILENAME_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    
    # Mensajes de error
    ERROR_MESSAGES = {
        'objective_required': 'El objetivo es requerido',
        'objective_too_long': f'El objetivo es demasiado largo (máximo {OBJECTIVE_MAX_LENGTH} caracteres)',
        'duration_invalid': 'La duración debe ser un número válido',
        'duration_too_short': f'La duración mínima es {MIN_DURATION_SECONDS} segundos',
        'duration_too_long': f'La duración máxima es {MAX_DURATION_SECONDS} segundos',
        'intensity_invalid': f'La intensidad debe ser una de: {", ".join(AuroraConfig.INTENSITY_LEVELS)}',
        'style_invalid': f'El estilo debe ser uno de: {", ".join(AuroraConfig.AVAILABLE_STYLES)}',
        'quality_invalid': f'La calidad debe ser una de: {", ".join(AuroraConfig.QUALITY_LEVELS)}',
        'sample_rate_invalid': f'Sample rate debe ser uno de: {", ".join(map(str, AuroraConfig.SUPPORTED_SAMPLE_RATES))}'
    }