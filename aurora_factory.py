#!/usr/bin/env python3
"""
🏭 AURORA FACTORY V7 - Sistema de Creación Dinámica
==================================================

Factory Pattern avanzado para crear componentes Aurora sin dependencias circulares.
Utiliza lazy loading, registro diferido y inyección de dependencias.

Es la "fábrica inteligente" que sabe cómo crear y conectar todos los componentes
del sistema de manera ordenada y sin ciclos.

Versión: 1.0
Propósito: Creación dinámica de componentes sin dependencias circulares
"""

import importlib
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Importar interfaces (sin crear ciclos)
try:
    from aurora_interfaces import (
        MotorAuroraInterface, GestorPerfilesInterface, AnalizadorInterface,
        ExtensionAuroraInterface, TipoComponenteAurora, EstadoComponente,
        validar_motor_aurora, validar_gestor_perfiles, ComponenteAuroraBase
    )
    INTERFACES_DISPONIBLES = True
except ImportError as e:
    logging.warning(f"Interfaces Aurora no disponibles: {e}")
    INTERFACES_DISPONIBLES = False
    # Definir clases dummy para evitar errores
    class MotorAuroraInterface: pass
    class GestorPerfilesInterface: pass
    class AnalizadorInterface: pass
    class ExtensionAuroraInterface: pass
    class TipoComponenteAurora: pass
    class EstadoComponente: pass

# Configurar logging
logger = logging.getLogger("Aurora.Factory")

# ===============================================================================
# SISTEMA DE LAZY IMPORTS
# ===============================================================================

class LazyImporter:
    """
    Sistema avanzado de importación tardía para evitar dependencias circulares
    
    Funciona como un "bibliotecario inteligente" que solo busca los libros
    cuando realmente los necesitas, evitando cargar todo al inicio.
    """
    
    def __init__(self):
        self._modules_cache = {}
        self._failed_imports = set()
        self._import_attempts = {}
        self._lock = threading.Lock()
    
    def import_module(self, module_name: str, package: Optional[str] = None, 
                     max_retries: int = 3) -> Optional[Any]:
        """
        Importa un módulo de forma tardía con reintentos
        
        Args:
            module_name: Nombre del módulo a importar
            package: Paquete padre (opcional)
            max_retries: Máximo número de reintentos
            
        Returns:
            Módulo importado o None si falla
        """
        cache_key = f"{package}.{module_name}" if package else module_name
        
        with self._lock:
            # Si ya está en caché, retornarlo
            if cache_key in self._modules_cache:
                return self._modules_cache[cache_key]
            
            # Si falló antes y no debe reintentar, retornar None
            if cache_key in self._failed_imports:
                return None
            
            # Intentar importar
            attempts = self._import_attempts.get(cache_key, 0)
            if attempts >= max_retries:
                self._failed_imports.add(cache_key)
                return None
            
            try:
                logger.debug(f"Importando {cache_key} (intento {attempts + 1})")
                module = importlib.import_module(module_name, package)
                self._modules_cache[cache_key] = module
                logger.debug(f"✅ {cache_key} importado exitosamente")
                return module
                
            except ImportError as e:
                self._import_attempts[cache_key] = attempts + 1
                logger.debug(f"❌ Error importando {cache_key}: {e}")
                
                if attempts + 1 >= max_retries:
                    self._failed_imports.add(cache_key)
                    logger.warning(f"❌ {cache_key} marcado como fallido después de {max_retries} intentos")
                
                return None
    
    def get_class(self, module_name: str, class_name: str, package: Optional[str] = None) -> Optional[Type]:
        """
        Obtiene una clase de un módulo de forma tardía
        
        Args:
            module_name: Nombre del módulo
            class_name: Nombre de la clase
            package: Paquete padre (opcional)
            
        Returns:
            Clase encontrada o None si no existe
        """
        module = self.import_module(module_name, package)
        if module and hasattr(module, class_name):
            return getattr(module, class_name)
        return None
    
    def get_function(self, module_name: str, function_name: str, package: Optional[str] = None) -> Optional[Callable]:
        """
        Obtiene una función de un módulo de forma tardía
        
        Args:
            module_name: Nombre del módulo
            function_name: Nombre de la función
            package: Paquete padre (opcional)
            
        Returns:
            Función encontrada o None si no existe
        """
        module = self.import_module(module_name, package)
        if module and hasattr(module, function_name):
            func = getattr(module, function_name)
            if callable(func):
                return func
        return None
    
    def clear_cache(self):
        """Limpia el caché de módulos importados"""
        with self._lock:
            self._modules_cache.clear()
            self._failed_imports.clear()
            self._import_attempts.clear()
            logger.info("🧹 Caché de LazyImporter limpiado")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del importador"""
        with self._lock:
            return {
                "modules_cached": len(self._modules_cache),
                "failed_imports": len(self._failed_imports),
                "import_attempts": dict(self._import_attempts),
                "cached_modules": list(self._modules_cache.keys()),
                "failed_modules": list(self._failed_imports)
            }

# Instancia global del lazy importer
lazy_importer = LazyImporter()

# ===============================================================================
# REGISTRO DE COMPONENTES
# ===============================================================================

@dataclass
class ComponenteRegistrado:
    """Información de un componente registrado en el factory"""
    
    nombre: str
    tipo: TipoComponenteAurora
    clase: Optional[Type] = None
    modulo_nombre: Optional[str] = None
    clase_nombre: Optional[str] = None
    instancia: Optional[Any] = None
    configuracion: Dict[str, Any] = field(default_factory=dict)
    dependencias: List[str] = field(default_factory=list)
    timestamp_registro: float = field(default_factory=time.time)
    activo: bool = True

class RegistroComponentes:
    """
    Registro centralizado de componentes Aurora
    
    Mantiene un catálogo de todos los componentes disponibles sin instanciarlos
    hasta que sean necesarios (lazy instantiation).
    """
    
    def __init__(self):
        self._componentes: Dict[str, ComponenteRegistrado] = {}
        self._tipos_disponibles: Dict[TipoComponenteAurora, List[str]] = {}
        self._lock = threading.Lock()
    
    def registrar(self, nombre: str, tipo: TipoComponenteAurora, 
                 clase: Optional[Type] = None, modulo_nombre: Optional[str] = None,
                 clase_nombre: Optional[str] = None, configuracion: Optional[Dict[str, Any]] = None,
                 dependencias: Optional[List[str]] = None) -> bool:
        """
        Registra un componente en el catálogo
        
        Args:
            nombre: Nombre único del componente
            tipo: Tipo de componente Aurora
            clase: Clase del componente (si ya está cargada)
            modulo_nombre: Nombre del módulo (para lazy loading)
            clase_nombre: Nombre de la clase en el módulo
            configuracion: Configuración inicial
            dependencias: Lista de dependencias
            
        Returns:
            True si se registró correctamente
        """
        with self._lock:
            if nombre in self._componentes:
                logger.warning(f"Componente {nombre} ya está registrado")
                return False
            
            # Validar que se proporcione información suficiente
            if not clase and not (modulo_nombre and clase_nombre):
                logger.error(f"Información insuficiente para registrar {nombre}")
                return False
            
            componente = ComponenteRegistrado(
                nombre=nombre,
                tipo=tipo,
                clase=clase,
                modulo_nombre=modulo_nombre,
                clase_nombre=clase_nombre,
                configuracion=configuracion or {},
                dependencias=dependencias or []
            )
            
            self._componentes[nombre] = componente
            
            # Actualizar índice por tipos
            if tipo not in self._tipos_disponibles:
                self._tipos_disponibles[tipo] = []
            self._tipos_disponibles[tipo].append(nombre)
            
            logger.debug(f"✅ Componente {nombre} ({tipo.value}) registrado")
            return True
    
    def obtener_componente(self, nombre: str) -> Optional[ComponenteRegistrado]:
        """Obtiene información de un componente registrado"""
        with self._lock:
            return self._componentes.get(nombre)
    
    def listar_por_tipo(self, tipo: TipoComponenteAurora) -> List[str]:
        """Lista componentes por tipo"""
        with self._lock:
            return self._tipos_disponibles.get(tipo, []).copy()
    
    def esta_registrado(self, nombre: str) -> bool:
        """Verifica si un componente está registrado"""
        with self._lock:
            return nombre in self._componentes
    
    def desregistrar(self, nombre: str) -> bool:
        """Desregistra un componente"""
        with self._lock:
            if nombre not in self._componentes:
                return False
            
            componente = self._componentes[nombre]
            del self._componentes[nombre]
            
            # Actualizar índice por tipos
            if componente.tipo in self._tipos_disponibles:
                if nombre in self._tipos_disponibles[componente.tipo]:
                    self._tipos_disponibles[componente.tipo].remove(nombre)
            
            logger.debug(f"🗑️ Componente {nombre} desregistrado")
            return True

# Instancia global del registro
registro_componentes = RegistroComponentes()

# ===============================================================================
# FACTORY PRINCIPAL
# ===============================================================================

class AuroraComponentFactory:
    """
    Factory principal para crear componentes Aurora dinámicamente
    
    Es la "fábrica inteligente" que sabe cómo crear, configurar y conectar
    todos los componentes del sistema Aurora sin crear dependencias circulares.
    """
    
    def __init__(self):
        self._instancias_cache: Dict[str, Any] = {}
        self._configuraciones_cache: Dict[str, Dict[str, Any]] = {}
        self._estadisticas = {
            "componentes_creados": 0,
            "errores_creacion": 0,
            "cache_hits": 0,
            "tiempo_total_creacion": 0.0
        }
        self._lock = threading.Lock()
        
        # Auto-registrar componentes conocidos del sistema Aurora
        self._auto_registrar_componentes_aurora()
    
    def _auto_registrar_componentes_aurora(self):
        """Auto-registra los componentes principales del sistema Aurora"""
        
        # Motores principales
        motores = [
            ("neuromix_aurora_v27", "AuroraNeuroAcousticEngineV27", "neuromix_aurora_v27"),
            ("neuromix_definitivo_v7", "NeuroMixSuperUnificado", "neuromix_definitivo_v7"),
            ("harmonicEssence_v34", "HarmonicEssenceV34Refactored", "harmonicEssence_v34"),
            ("hypermod_v32", "HyperModEngineV32AuroraConnected", "hypermod_v32")
        ]
        
        for nombre, clase_nombre, modulo_nombre in motores:
            self.registrar_motor(nombre, modulo_nombre, clase_nombre)
        
        # Gestores de perfiles
        gestores = [
            ("emotion_style_profiles", "GestorPresetsEmocionales", "emotion_style_profiles"),
            ("field_profiles", "GestorPerfilesCampo", "field_profiles"),
            ("objective_manager", "ObjectiveManagerUnificado", "objective_manager")
        ]
        
        for nombre, clase_nombre, modulo_nombre in gestores:
            self.registrar_gestor(nombre, modulo_nombre, clase_nombre)
        
        # Analizadores
        analizadores = [
            ("carmine_analyzer", "CarmineAuroraAnalyzer", "Carmine_Analyzer"),
            ("aurora_quality_pipeline", "AuroraQualityPipeline", "aurora_quality_pipeline")
        ]
        
        for nombre, clase_nombre, modulo_nombre in analizadores:
            self.registrar_analizador(nombre, modulo_nombre, clase_nombre)
        
        # Extensiones
        extensiones = [
            ("multimotor_extension", "ColaboracionMultiMotorV7Optimizada", "aurora_multimotor_extension"),
            ("sync_scheduler", "SyncAndSchedulerV7", "sync_and_scheduler")
        ]
        
        for nombre, clase_nombre, modulo_nombre in extensiones:
            self.registrar_extension(nombre, modulo_nombre, clase_nombre)
        
        logger.info("🏭 Componentes Aurora auto-registrados en el factory")
    
    # Métodos de registro especializados
    
    def registrar_motor(self, nombre: str, modulo_nombre: str, clase_nombre: str, 
                       dependencias: Optional[List[str]] = None) -> bool:
        """Registra un motor de audio Aurora"""
        return registro_componentes.registrar(
            nombre=nombre,
            tipo=TipoComponenteAurora.MOTOR,
            modulo_nombre=modulo_nombre,
            clase_nombre=clase_nombre,
            dependencias=dependencias or []
        )
    
    def registrar_gestor(self, nombre: str, modulo_nombre: str, clase_nombre: str,
                        dependencias: Optional[List[str]] = None) -> bool:
        """Registra un gestor de perfiles"""
        return registro_componentes.registrar(
            nombre=nombre,
            tipo=TipoComponenteAurora.GESTOR,
            modulo_nombre=modulo_nombre,
            clase_nombre=clase_nombre,
            dependencias=dependencias or []
        )
    
    def registrar_analizador(self, nombre: str, modulo_nombre: str, clase_nombre: str,
                           dependencias: Optional[List[str]] = None) -> bool:
        """Registra un analizador de audio"""
        return registro_componentes.registrar(
            nombre=nombre,
            tipo=TipoComponenteAurora.ANALIZADOR,
            modulo_nombre=modulo_nombre,
            clase_nombre=clase_nombre,
            dependencias=dependencias or []
        )
    
    def registrar_extension(self, nombre: str, modulo_nombre: str, clase_nombre: str,
                          dependencias: Optional[List[str]] = None) -> bool:
        """Registra una extensión del sistema"""
        return registro_componentes.registrar(
            nombre=nombre,
            tipo=TipoComponenteAurora.EXTENSION,
            modulo_nombre=modulo_nombre,
            clase_nombre=clase_nombre,
            dependencias=dependencias or []
        )
    
    # Métodos de creación
    
    def crear_motor(self, nombre: str, configuracion: Optional[Dict[str, Any]] = None) -> Optional[MotorAuroraInterface]:
        """
        Crea un motor de audio por nombre
        
        Args:
            nombre: Nombre del motor registrado
            configuracion: Configuración inicial
            
        Returns:
            Instancia del motor o None si falla
        """
        return self._crear_componente(nombre, TipoComponenteAurora.MOTOR, configuracion)
    
    def crear_gestor(self, nombre: str, configuracion: Optional[Dict[str, Any]] = None) -> Optional[GestorPerfilesInterface]:
        """
        Crea un gestor de perfiles por nombre
        
        Args:
            nombre: Nombre del gestor registrado
            configuracion: Configuración inicial
            
        Returns:
            Instancia del gestor o None si falla
        """
        return self._crear_componente(nombre, TipoComponenteAurora.GESTOR, configuracion)
    
    def crear_analizador(self, nombre: str, configuracion: Optional[Dict[str, Any]] = None) -> Optional[AnalizadorInterface]:
        """
        Crea un analizador por nombre
        
        Args:
            nombre: Nombre del analizador registrado
            configuracion: Configuración inicial
            
        Returns:
            Instancia del analizador o None si falla
        """
        return self._crear_componente(nombre, TipoComponenteAurora.ANALIZADOR, configuracion)
    
    def crear_extension(self, nombre: str, configuracion: Optional[Dict[str, Any]] = None) -> Optional[ExtensionAuroraInterface]:
        """
        Crea una extensión por nombre
        
        Args:
            nombre: Nombre de la extensión registrada
            configuracion: Configuración inicial
            
        Returns:
            Instancia de la extensión o None si falla
        """
        return self._crear_componente(nombre, TipoComponenteAurora.EXTENSION, configuracion)
    
    def _crear_componente(self, nombre: str, tipo_esperado: TipoComponenteAurora,
                         configuracion: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Método interno para crear cualquier tipo de componente
        
        Args:
            nombre: Nombre del componente
            tipo_esperado: Tipo esperado del componente
            configuracion: Configuración inicial
            
        Returns:
            Instancia del componente o None si falla
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Verificar si ya está en caché
                cache_key = f"{nombre}_{hash(str(configuracion or {}))}"
                if cache_key in self._instancias_cache:
                    self._estadisticas["cache_hits"] += 1
                    return self._instancias_cache[cache_key]
                
                # Obtener información del componente
                componente_info = registro_componentes.obtener_componente(nombre)
                if not componente_info:
                    logger.error(f"Componente {nombre} no registrado")
                    return None
                
                # Verificar tipo
                if componente_info.tipo != tipo_esperado:
                    logger.error(f"Componente {nombre} es {componente_info.tipo.value}, se esperaba {tipo_esperado.value}")
                    return None
                
                # Obtener clase del componente
                clase_componente = self._obtener_clase_componente(componente_info)
                if not clase_componente:
                    logger.error(f"No se pudo obtener la clase para {nombre}")
                    return None
                
                # Preparar configuración
                config_final = componente_info.configuracion.copy()
                if configuracion:
                    config_final.update(configuracion)
                
                # Crear instancia
                logger.debug(f"Creando instancia de {nombre}...")
                
                try:
                    # Intentar crear con configuración
                    if config_final:
                        instancia = clase_componente(**config_final)
                    else:
                        instancia = clase_componente()
                except TypeError:
                    # Si falla, crear sin parámetros
                    instancia = clase_componente()
                
                # Validar que implementa la interface correcta
                if not self._validar_interface_componente(instancia, tipo_esperado):
                    logger.error(f"Componente {nombre} no implementa la interface requerida")
                    return None
                
                # Cachear y estadísticas
                self._instancias_cache[cache_key] = instancia
                self._estadisticas["componentes_creados"] += 1
                
                end_time = time.time()
                self._estadisticas["tiempo_total_creacion"] += (end_time - start_time)
                
                logger.debug(f"✅ Componente {nombre} creado exitosamente")
                return instancia
                
        except Exception as e:
            self._estadisticas["errores_creacion"] += 1
            logger.error(f"❌ Error creando componente {nombre}: {e}")
            return None
    
    def _obtener_clase_componente(self, componente_info: ComponenteRegistrado) -> Optional[Type]:
        """Obtiene la clase de un componente usando lazy loading"""
        
        # Si ya tiene la clase cargada
        if componente_info.clase:
            return componente_info.clase
        
        # Cargar usando lazy importer
        if componente_info.modulo_nombre and componente_info.clase_nombre:
            return lazy_importer.get_class(
                componente_info.modulo_nombre,
                componente_info.clase_nombre
            )
        
        return None
    
    def _validar_interface_componente(self, instancia: Any, tipo: TipoComponenteAurora) -> bool:
        """Valida que un componente implemente la interface correcta"""
    
        if not INTERFACES_DISPONIBLES:
            return True  # Skip validation si no hay interfaces
    
        # FIX ESPECÍFICO PARA HYPERMOD V32: Resolver problema de timing
        if hasattr(instancia, 'nombre') and 'HyperMod' in str(instancia.nombre):
            logger.info("🔧 Aplicando fix de timing para HyperMod V32...")
        
            # Dar tiempo a HyperMod para completar auto-integración
            import time
            time.sleep(0.1)  # 100ms para completar auto-integración
        
            # Verificar que los métodos estén realmente disponibles
            metodos_requeridos = ['generar_audio', 'validar_configuracion', 'obtener_capacidades', 'get_info']
            for metodo in metodos_requeridos:
                if not hasattr(instancia, metodo):
                    logger.warning(f"⚠️ Método {metodo} no encontrado en {instancia.nombre}, reintentando...")
                    time.sleep(0.05)  # 50ms adicionales
                    if not hasattr(instancia, metodo):
                        logger.error(f"❌ Método {metodo} definitivamente no disponible")
                        return False
        
            logger.info("✅ HyperMod V32 timing fix aplicado correctamente")
    
        if tipo == TipoComponenteAurora.MOTOR:
            return validar_motor_aurora(instancia)
        elif tipo == TipoComponenteAurora.GESTOR:
            return validar_gestor_perfiles(instancia)
        # Agregar más validaciones según necesidad
    
        return True  # Default: asumir válido
    
    # Métodos de utilidad
    
    def listar_motores_disponibles(self) -> List[str]:
        """Lista todos los motores registrados"""
        return registro_componentes.listar_por_tipo(TipoComponenteAurora.MOTOR)
    
    def listar_gestores_disponibles(self) -> List[str]:
        """Lista todos los gestores registrados"""
        return registro_componentes.listar_por_tipo(TipoComponenteAurora.GESTOR)
    
    def listar_analizadores_disponibles(self) -> List[str]:
        """Lista todos los analizadores registrados"""
        return registro_componentes.listar_por_tipo(TipoComponenteAurora.ANALIZADOR)
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas del factory"""
        with self._lock:
            return {
                "factory_stats": self._estadisticas.copy(),
                "lazy_importer_stats": lazy_importer.get_stats(),
                "componentes_en_cache": len(self._instancias_cache),
                "total_registrados": len(registro_componentes._componentes)
            }
    
    def limpiar_cache(self):
        """Limpia todos los caches"""
        with self._lock:
            self._instancias_cache.clear()
            self._configuraciones_cache.clear()
            lazy_importer.clear_cache()
            logger.info("🧹 Cache del factory limpiado")

# ===============================================================================
# BUILDER PATTERN PARA CONFIGURACIÓN COMPLEJA
# ===============================================================================

class AuroraSystemBuilder:
    """
    Builder para configurar el sistema Aurora completo
    
    Permite crear configuraciones complejas del sistema paso a paso
    sin preocuparse por las dependencias circulares.
    """
    
    def __init__(self):
        self.factory = AuroraComponentFactory()
        self._configuracion = {}
        self._motores_deseados = []
        self._gestores_deseados = []
        self._analizadores_deseados = []
        self._extensiones_deseadas = []
    
    def con_motor(self, nombre: str, configuracion: Optional[Dict[str, Any]] = None):
        """Añade un motor al sistema"""
        self._motores_deseados.append((nombre, configuracion))
        return self
    
    def con_gestor(self, nombre: str, configuracion: Optional[Dict[str, Any]] = None):
        """Añade un gestor al sistema"""
        self._gestores_deseados.append((nombre, configuracion))
        return self
    
    def con_analizador(self, nombre: str, configuracion: Optional[Dict[str, Any]] = None):
        """Añade un analizador al sistema"""
        self._analizadores_deseados.append((nombre, configuracion))
        return self
    
    def con_extension(self, nombre: str, configuracion: Optional[Dict[str, Any]] = None):
        """Añade una extensión al sistema"""
        self._extensiones_deseadas.append((nombre, configuracion))
        return self
    
    def construir(self) -> Dict[str, Any]:
        """
        Construye el sistema Aurora completo
        
        Returns:
            Diccionario con todos los componentes creados
        """
        sistema = {
            "motores": {},
            "gestores": {},
            "analizadores": {},
            "extensiones": {},
            "errores": []
        }
        
        # Crear motores
        for nombre, config in self._motores_deseados:
            motor = self.factory.crear_motor(nombre, config)
            if motor:
                sistema["motores"][nombre] = motor
            else:
                sistema["errores"].append(f"No se pudo crear motor {nombre}")
        
        # Crear gestores
        for nombre, config in self._gestores_deseados:
            gestor = self.factory.crear_gestor(nombre, config)
            if gestor:
                sistema["gestores"][nombre] = gestor
            else:
                sistema["errores"].append(f"No se pudo crear gestor {nombre}")
        
        # Crear analizadores
        for nombre, config in self._analizadores_deseados:
            analizador = self.factory.crear_analizador(nombre, config)
            if analizador:
                sistema["analizadores"][nombre] = analizador
            else:
                sistema["errores"].append(f"No se pudo crear analizador {nombre}")
        
        # Crear extensiones
        for nombre, config in self._extensiones_deseadas:
            extension = self.factory.crear_extension(nombre, config)
            if extension:
                sistema["extensiones"][nombre] = extension
            else:
                sistema["errores"].append(f"No se pudo crear extensión {nombre}")
        
        return sistema

# ===============================================================================
# INSTANCIA GLOBAL DEL FACTORY
# ===============================================================================

# Factory global para uso en todo el sistema
aurora_factory = AuroraComponentFactory()

# ===============================================================================
# FUNCIONES DE CONVENIENCIA
# ===============================================================================

def crear_motor_aurora(nombre: str, configuracion: Optional[Dict[str, Any]] = None) -> Optional[MotorAuroraInterface]:
    """Función de conveniencia para crear motores"""
    return aurora_factory.crear_motor(nombre, configuracion)

def crear_gestor_perfiles(nombre: str, configuracion: Optional[Dict[str, Any]] = None) -> Optional[GestorPerfilesInterface]:
    """Función de conveniencia para crear gestores"""
    return aurora_factory.crear_gestor(nombre, configuracion)

def crear_sistema_aurora_basico() -> Dict[str, Any]:
    """Crea un sistema Aurora básico con componentes esenciales"""
    builder = AuroraSystemBuilder()
    return (builder
            .con_motor("neuromix_aurora_v27")
            .con_motor("harmonicEssence_v34")
            .con_gestor("emotion_style_profiles")
            .con_analizador("carmine_analyzer")
            .construir())

def crear_sistema_aurora_completo() -> Dict[str, Any]:
    """Crea un sistema Aurora completo con todos los componentes"""
    builder = AuroraSystemBuilder()
    return (builder
            .con_motor("neuromix_aurora_v27")
            .con_motor("neuromix_definitivo_v7")
            .con_motor("harmonicEssence_v34")
            .con_motor("hypermod_v32")
            .con_gestor("emotion_style_profiles")
            .con_gestor("field_profiles")
            .con_gestor("objective_manager")
            .con_analizador("carmine_analyzer")
            .con_analizador("aurora_quality_pipeline")
            .con_extension("multimotor_extension")
            .con_extension("sync_scheduler")
            .construir())

# ===============================================================================
# INFORMACIÓN DEL MÓDULO
# ===============================================================================

__version__ = "1.0.0"
__author__ = "Sistema Aurora V7"
__description__ = "Factory Pattern para creación dinámica sin dependencias circulares"

if __name__ == "__main__":
    print("🏭 Aurora Factory V7 - Sistema de Creación Dinámica")
    print(f"Versión: {__version__}")
    
    # Mostrar estadísticas
    stats = aurora_factory.obtener_estadisticas()
    print(f"Componentes registrados: {stats['total_registrados']}")
    print(f"Motores disponibles: {len(aurora_factory.listar_motores_disponibles())}")
    print(f"Gestores disponibles: {len(aurora_factory.listar_gestores_disponibles())}")
    
    print("✅ Factory listo para crear componentes sin dependencias circulares")

# 🔁 Motor híbrido: Selección inteligente entre definitivo y base
def crear_motor_neuromix_hibrido() -> Optional[object]:
    """
    Intenta crear el motor definitivo; si falla, usa el motor base.
    """
    try:
        from neuromix_definitivo_v7 import NeuroMixSuperUnificado
        print("✅ Motor definitivo NeuroMixSuperUnificado activado")
        return NeuroMixSuperUnificado()
    except Exception as e:
        print(f"⚠️ Falla al cargar definitivo: {e}")
        try:
            from neuromix_aurora_v27 import AuroraNeuroAcousticEngineV27
            print("✅ Fallback: Motor base NeuroMix V27 activado")
            return AuroraNeuroAcousticEngineV27()
        except Exception as e2:
            print(f"❌ Error total en carga de NeuroMix: {e2}")
            return None