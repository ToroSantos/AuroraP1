#!/usr/bin/env python3
"""
🎯 AURORA INTERFACES V7 - Contratos del Sistema
===============================================

Este módulo define todas las interfaces abstractas y protocolos que permiten
la comunicación entre componentes del Sistema Aurora V7 sin dependencias circulares.

Es el "lenguaje común" que hablan todos los módulos del sistema.

Versión: 1.0
Propósito: Eliminar dependencias circulares mediante contratos claros
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol, Union, Callable
from enum import Enum
import numpy as np

# ===============================================================================
# ENUMS Y TIPOS BASE
# ===============================================================================

class TipoComponenteAurora(Enum):
    """Tipos de componentes en el sistema Aurora"""
    MOTOR = "motor"
    GESTOR = "gestor" 
    ANALIZADOR = "analizador"
    EXTENSION = "extension"
    PARCHE = "parche"

class EstadoComponente(Enum):
    """Estados posibles de un componente"""
    INICIALIZADO = "inicializado"
    CONFIGURADO = "configurado"
    ACTIVO = "activo"
    ERROR = "error"
    DESHABILITADO = "deshabilitado"

class NivelCalidad(Enum):
    """Niveles de calidad del sistema"""
    BASICO = "basico"
    ESTANDAR = "estandar"
    ALTO = "alto"
    PROFESIONAL = "profesional"
    MAXIMA = "maxima"

# ===============================================================================
# INTERFACES PRINCIPALES
# ===============================================================================

class MotorAuroraInterface(Protocol):
    """
    Interface estándar para todos los motores de audio Aurora V7
    
    Todos los motores (NeuroMix, HyperMod, HarmonicEssence, etc.) 
    deben implementar estos métodos para ser compatibles con AuroraDirector.
    """
    
    # Propiedades requeridas
    nombre: str
    version: str
    
    @abstractmethod
    def generar_audio(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera audio según la configuración proporcionada
        
        Args:
            config: Configuración de generación
            
        Returns:
            Dict con audio_data, metadatos, etc.
        """
        pass
    
    @abstractmethod
    def validar_configuracion(self, config: Dict[str, Any]) -> bool:
        """
        Valida si la configuración es correcta para este motor
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        pass
    
    @abstractmethod
    def obtener_capacidades(self) -> Dict[str, Any]:
        """
        Retorna las capacidades y características del motor
        
        Returns:
            Dict con capacidades, formatos soportados, etc.
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """
        Información básica del motor
        
        Returns:
            Dict con nombre, versión, descripción, etc.
        """
        pass
    
    def obtener_estado(self) -> EstadoComponente:
        """Estado actual del motor"""
        return EstadoComponente.ACTIVO
    
    def inicializar(self) -> bool:
        """Inicializa el motor"""
        return True

class GestorPerfilesInterface(Protocol):
    """
    Interface para gestores de perfiles (emocionales, estilos, campos)
    
    Usado por emotion_style_profiles, field_profiles, etc.
    """
    
    @abstractmethod
    def obtener_perfil(self, nombre: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un perfil por nombre
        
        Args:
            nombre: Nombre del perfil a obtener
            
        Returns:
            Dict con configuración del perfil o None si no existe
        """
        pass
    
    @abstractmethod
    def listar_perfiles(self) -> List[str]:
        """
        Lista todos los perfiles disponibles
        
        Returns:
            Lista de nombres de perfiles
        """
        pass
    
    @abstractmethod
    def validar_perfil(self, perfil: Dict[str, Any]) -> bool:
        """
        Valida si un perfil es correcto
        
        Args:
            perfil: Configuración del perfil
            
        Returns:
            True si el perfil es válido
        """
        pass
    
    def obtener_categoria(self) -> str:
        """Categoría del gestor de perfiles"""
        return "general"

class AnalizadorInterface(Protocol):
    """
    Interface para analizadores de audio (Carmine, Quality Pipeline, etc.)
    """
    
    @abstractmethod
    def analizar(self, audio_data: Union[np.ndarray, bytes]) -> Dict[str, Any]:
        """
        Analiza audio y retorna métricas
        
        Args:
            audio_data: Datos de audio a analizar
            
        Returns:
            Dict con métricas y análisis
        """
        pass
    
    @abstractmethod
    def obtener_metricas_disponibles(self) -> List[str]:
        """
        Lista las métricas que puede calcular este analizador
        
        Returns:
            Lista de nombres de métricas
        """
        pass
    
    def configurar_analisis(self, config: Dict[str, Any]) -> None:
        """Configura parámetros del análisis"""
        pass

class ExtensionAuroraInterface(Protocol):
    """
    Interface para extensiones del sistema (MultiMotor, Sync, etc.)
    """
    
    @abstractmethod
    def aplicar_extension(self, contexto: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica la extensión al contexto dado
        
        Args:
            contexto: Contexto de aplicación
            
        Returns:
            Contexto modificado por la extensión
        """
        pass
    
    @abstractmethod
    def es_compatible(self, version_aurora: str) -> bool:
        """
        Verifica compatibilidad con versión de Aurora
        
        Args:
            version_aurora: Versión del sistema Aurora
            
        Returns:
            True si es compatible
        """
        pass

class GestorObjetivosInterface(Protocol):
    """
    Interface para gestores de objetivos terapéuticos
    """
    
    @abstractmethod
    def obtener_template(self, objetivo: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene template para un objetivo específico
        
        Args:
            objetivo: Nombre del objetivo (ej: "relajacion", "enfoque")
            
        Returns:
            Dict con template o None si no existe
        """
        pass
    
    @abstractmethod
    def listar_objetivos(self) -> List[str]:
        """
        Lista objetivos disponibles
        
        Returns:
            Lista de nombres de objetivos
        """
        pass
    
    @abstractmethod
    def rutear_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determina la mejor configuración para un objetivo
        
        Args:
            objetivo: Objetivo deseado
            contexto: Contexto del usuario
            
        Returns:
            Configuración optimizada
        """
        pass

# ===============================================================================
# INTERFACES DE COMUNICACIÓN
# ===============================================================================

class EventoAuroraInterface(Protocol):
    """Interface para eventos del sistema"""
    
    tipo: str
    timestamp: float
    datos: Dict[str, Any]
    
    def serializar(self) -> Dict[str, Any]:
        """Convierte el evento a diccionario"""
        pass

class CallbackInterface(Protocol):
    """Interface para callbacks de progreso"""
    
    def __call__(self, progreso: float, mensaje: str, datos: Optional[Dict[str, Any]] = None) -> None:
        """
        Callback de progreso
        
        Args:
            progreso: Porcentaje completado (0.0 a 1.0)
            mensaje: Mensaje descriptivo
            datos: Datos adicionales opcionales
        """
        pass

# ===============================================================================
# INTERFACES DE CONFIGURACIÓN
# ===============================================================================

class ConfiguracionBaseInterface(Protocol):
    """Interface base para configuraciones"""
    
    @abstractmethod
    def validar(self) -> bool:
        """Valida la configuración"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Carga desde diccionario"""
        pass

class ResultadoBaseInterface(Protocol):
    """Interface base para resultados"""
    
    exito: bool
    mensaje: str
    datos: Dict[str, Any]
    
    @abstractmethod
    def serializar(self) -> Dict[str, Any]:
        """Serializa el resultado"""
        pass

# ===============================================================================
# INTERFACES ESPECÍFICAS AURORA
# ===============================================================================

class DirectorAuroraInterface(Protocol):
    """
    Interface para el director principal del sistema Aurora
    """
    
    @abstractmethod
    def registrar_componente(self, nombre: str, componente: Any, tipo: TipoComponenteAurora) -> bool:
        """
        Registra un componente en el director
        
        Args:
            nombre: Nombre del componente
            componente: Instancia del componente
            tipo: Tipo de componente
            
        Returns:
            True si se registró correctamente
        """
        pass
    
    @abstractmethod
    def crear_experiencia(self, config: ConfiguracionBaseInterface) -> ResultadoBaseInterface:
        """
        Crea una experiencia de audio completa
        
        Args:
            config: Configuración de la experiencia
            
        Returns:
            Resultado con audio generado y metadatos
        """
        pass
    
    @abstractmethod
    def obtener_estado_sistema(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo del sistema
        
        Returns:
            Dict con estado de todos los componentes
        """
        pass

# ===============================================================================
# MIXINS Y CLASES BASE
# ===============================================================================

class ComponenteAuroraBase:
    """
    Clase base con funcionalidad común para componentes Aurora
    """
    
    def __init__(self, nombre: str, version: str = "1.0"):
        self.nombre = nombre
        self.version = version
        self.estado = EstadoComponente.INICIALIZADO
        self._configuracion = {}
        self._logs = []
    
    def log(self, mensaje: str, nivel: str = "INFO"):
        """Registra un log del componente"""
        import time
        self._logs.append({
            "timestamp": time.time(),
            "nivel": nivel,
            "mensaje": f"[{self.nombre}] {mensaje}"
        })
    
    def obtener_logs(self) -> List[Dict[str, Any]]:
        """Obtiene los logs del componente"""
        return self._logs.copy()
    
    def cambiar_estado(self, nuevo_estado: EstadoComponente):
        """Cambia el estado del componente"""
        estado_anterior = self.estado
        self.estado = nuevo_estado
        self.log(f"Estado cambió de {estado_anterior.value} a {nuevo_estado.value}")

class CompatibilidadV6Mixin:
    """
    Mixin para mantener compatibilidad con versiones anteriores
    """
    
    def obtener_info_v6(self) -> Dict[str, str]:
        """Información en formato V6 para compatibilidad"""
        return {
            "name": getattr(self, 'nombre', 'componente'),
            "version": getattr(self, 'version', '1.0'),
            "type": "aurora_component"
        }

# ===============================================================================
# VALIDADORES Y UTILIDADES
# ===============================================================================

def validar_motor_aurora(componente: Any) -> bool:
    """
    Valida si un componente implementa la interface MotorAuroraInterface
    
    Args:
        componente: Componente a validar
        
    Returns:
        True si implementa la interface correctamente
    """
    metodos_requeridos = ['generar_audio', 'validar_configuracion', 'obtener_capacidades', 'get_info']
    
    for metodo in metodos_requeridos:
        if not hasattr(componente, metodo):
            return False
        if not callable(getattr(componente, metodo)):
            return False
    
    # Verificar propiedades
    if not hasattr(componente, 'nombre') or not hasattr(componente, 'version'):
        return False
    
    return True

def validar_gestor_perfiles(componente: Any) -> bool:
    """
    Valida si un componente implementa la interface GestorPerfilesInterface
    
    Args:
        componente: Componente a validar
        
    Returns:
        True si implementa la interface correctamente
    """
    metodos_requeridos = ['obtener_perfil', 'listar_perfiles', 'validar_perfil']
    
    for metodo in metodos_requeridos:
        if not hasattr(componente, metodo):
            return False
        if not callable(getattr(componente, metodo)):
            return False
    
    return True

def obtener_interface_para_tipo(tipo_componente: TipoComponenteAurora) -> Optional[type]:
    """
    Obtiene la interface apropiada para un tipo de componente
    
    Args:
        tipo_componente: Tipo del componente
        
    Returns:
        Clase de interface correspondiente
    """
    mapping = {
        TipoComponenteAurora.MOTOR: MotorAuroraInterface,
        TipoComponenteAurora.GESTOR: GestorPerfilesInterface, 
        TipoComponenteAurora.ANALIZADOR: AnalizadorInterface,
        TipoComponenteAurora.EXTENSION: ExtensionAuroraInterface
    }
    
    return mapping.get(tipo_componente)

# ===============================================================================
# EXCEPCIONES ESPECÍFICAS
# ===============================================================================

class AuroraInterfaceError(Exception):
    """Error relacionado con interfaces Aurora"""
    pass

class ComponenteNoCompatibleError(AuroraInterfaceError):
    """Error cuando un componente no es compatible con la interface"""
    pass

class MetodoNoImplementadoError(AuroraInterfaceError):
    """Error cuando un método requerido no está implementado"""
    pass

# ===============================================================================
# INFORMACIÓN DEL MÓDULO
# ===============================================================================

__version__ = "1.0.0"
__author__ = "Sistema Aurora V7"
__description__ = "Interfaces y protocolos para eliminar dependencias circulares"

# Exportar interfaces principales
__all__ = [
    # Enums
    'TipoComponenteAurora', 'EstadoComponente', 'NivelCalidad',
    
    # Interfaces principales
    'MotorAuroraInterface', 'GestorPerfilesInterface', 'AnalizadorInterface',
    'ExtensionAuroraInterface', 'GestorObjetivosInterface', 'DirectorAuroraInterface',
    
    # Interfaces de comunicación
    'EventoAuroraInterface', 'CallbackInterface',
    
    # Interfaces de configuración
    'ConfiguracionBaseInterface', 'ResultadoBaseInterface',
    
    # Clases base
    'ComponenteAuroraBase', 'CompatibilidadV6Mixin',
    
    # Validadores
    'validar_motor_aurora', 'validar_gestor_perfiles', 'obtener_interface_para_tipo',
    
    # Excepciones
    'AuroraInterfaceError', 'ComponenteNoCompatibleError', 'MetodoNoImplementadoError'
]

if __name__ == "__main__":
    print("🎯 Aurora Interfaces V7 - Sistema de Contratos Cargado")
    print(f"Versión: {__version__}")
    print(f"Interfaces disponibles: {len(__all__)}")
    print("✅ Listo para eliminar dependencias circulares")
