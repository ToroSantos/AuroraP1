# ================================================================
# PROTECCIÓN ANTI-RECURSIÓN SIMPLIFICADA PARA AURORA FACTORY
# ================================================================

import threading
import logging
from typing import Dict, Set

# Configurar logger
logger = logging.getLogger("AntiRecursion")

class AntiRecursionGuard:
    """Guardia anti-recursión simple para prevenir bucles infinitos"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._creation_stack: Dict[int, Set[str]] = {}
        self._blocked_components: Set[str] = set()
    
    def is_creating(self, component_name: str) -> bool:
        """Verifica si un componente está siendo creado actualmente"""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._creation_stack:
                return False
            return component_name in self._creation_stack[thread_id]
    
    def enter_creation(self, component_name: str) -> bool:
        """Intenta entrar en creación de componente"""
        thread_id = threading.get_ident()
        with self._lock:
            if component_name in self._blocked_components:
                return False
            
            if thread_id in self._creation_stack:
                if component_name in self._creation_stack[thread_id]:
                    return False
            
            if thread_id not in self._creation_stack:
                self._creation_stack[thread_id] = set()
            
            self._creation_stack[thread_id].add(component_name)
            return True
    
    def exit_creation(self, component_name: str):
        """Sale de la creación de componente"""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id in self._creation_stack:
                self._creation_stack[thread_id].discard(component_name)
                if not self._creation_stack[thread_id]:
                    del self._creation_stack[thread_id]

# Instancia global
_anti_recursion_guard = AntiRecursionGuard()

def patch_aurora_factory_anti_recursion():
    """Aplica parche anti-recursión al aurora_factory"""
    
    try:
        import aurora_factory
        
        # Detectar método disponible
        if hasattr(aurora_factory, 'crear_motor_aurora'):
            metodo_name = 'crear_motor_aurora'
            metodo_original = aurora_factory.crear_motor_aurora
        elif hasattr(aurora_factory, 'crear_motor'):
            metodo_name = 'crear_motor'
            metodo_original = aurora_factory.crear_motor
        else:
            logger.warning("⚠️ No se encontró método de creación en aurora_factory")
            return False
        
        # Guardar original
        backup_name = f'_{metodo_name}_original'
        if not hasattr(aurora_factory, backup_name):
            setattr(aurora_factory, backup_name, metodo_original)
        
        def crear_motor_protected(nombre_motor: str, *args, **kwargs):
            """Versión protegida contra recursión"""
            
            if nombre_motor == "neuromix_aurora_v27":
                if _anti_recursion_guard.is_creating("neuromix_aurora_v27"):
                    logger.warning("🚫 Recursión detectada en neuromix_aurora_v27")
                    return None
                
                if not _anti_recursion_guard.enter_creation("neuromix_aurora_v27"):
                    logger.warning("🚫 Creación de neuromix_aurora_v27 bloqueada")
                    return None
                
                try:
                    # Usar método original guardado
                    metodo_backup = getattr(aurora_factory, backup_name)
                    resultado = metodo_backup(nombre_motor, *args, **kwargs)
                    
                    # Verificar clase problemática
                    if resultado and hasattr(resultado, '__class__'):
                        nombre_clase = resultado.__class__.__name__
                        problemas = ["Connected", "Super", "Unificado", "Definitivo"]
                        if any(p in nombre_clase for p in problemas):
                            logger.warning(f"🚫 Clase problemática: {nombre_clase}")
                            return None
                    
                    logger.info(f"✅ neuromix_aurora_v27 creado: {type(resultado).__name__}")
                    return resultado
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error creando neuromix_aurora_v27: {e}")
                    return None
                    
                finally:
                    _anti_recursion_guard.exit_creation("neuromix_aurora_v27")
            
            else:
                # Para otros motores, usar método original
                try:
                    metodo_backup = getattr(aurora_factory, backup_name)
                    return metodo_backup(nombre_motor, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ Error creando {nombre_motor}: {e}")
                    return None
        
        # Reemplazar método
        setattr(aurora_factory, metodo_name, crear_motor_protected)
        logger.info(f"✅ Parche anti-recursión aplicado a {metodo_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error aplicando parche: {e}")
        return False

# Auto-aplicar al importar
try:
    if __name__ != "__main__":
        patch_aurora_factory_anti_recursion()
        print("🛡️ Protección anti-recursión activada")
except:
    print("⚠️ Error activando protección anti-recursión")

if __name__ == "__main__":
    print("🧪 Testeando protección anti-recursión...")
    patch_aurora_factory_anti_recursion()
    print("✅ Test completado")
