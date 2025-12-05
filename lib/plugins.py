"""
🔌 Système de Plugins Kibalone
==============================

Gère le chargement et l'utilisation des plugins Kibalone.
Isol est le premier plugin officiel.
"""

import os
import importlib
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class PluginManager:
    """Gestionnaire de plugins pour Kibalone"""
    
    def __init__(self, plugins_dir: Optional[str] = None):
        """
        Args:
            plugins_dir: Répertoire contenant les plugins (auto-détecté si None)
        """
        if plugins_dir is None:
            # Trouver le répertoire plugins relativement à ce fichier
            base_dir = Path(__file__).parent
            plugins_dir = base_dir
        
        self.plugins_dir = Path(plugins_dir)
        self.loaded_plugins = {}
        
        # Ajouter le répertoire au PYTHONPATH
        if str(self.plugins_dir) not in sys.path:
            sys.path.insert(0, str(self.plugins_dir))
    
    def load_plugin(self, plugin_name: str) -> Any:
        """
        Charge un plugin par son nom
        
        Args:
            plugin_name: Nom du plugin (ex: "isol")
            
        Returns:
            Module du plugin chargé
        """
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name]
        
        try:
            # Importer le module du plugin
            plugin_module = importlib.import_module(plugin_name)
            self.loaded_plugins[plugin_name] = plugin_module
            
            print(f"✅ Plugin '{plugin_name}' chargé")
            return plugin_module
            
        except ImportError as e:
            raise ImportError(f"Impossible de charger le plugin '{plugin_name}': {e}")
    
    def list_plugins(self) -> list:
        """Liste tous les plugins disponibles"""
        plugins = []
        
        for item in self.plugins_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                # Vérifier si c'est un plugin (contient __init__.py)
                if (item / '__init__.py').exists():
                    plugins.append(item.name)
        
        return plugins
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Récupère les infos d'un plugin"""
        plugin = self.load_plugin(plugin_name)
        
        return {
            'name': plugin_name,
            'version': getattr(plugin, '__version__', 'unknown'),
            'author': getattr(plugin, '__author__', 'unknown'),
            'description': getattr(plugin, '__doc__', 'No description')
        }


# Instance globale
_plugin_manager = None


def get_plugin_manager() -> PluginManager:
    """Retourne l'instance globale du gestionnaire de plugins"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def load_plugin(name: str) -> Any:
    """Raccourci pour charger un plugin"""
    return get_plugin_manager().load_plugin(name)


# =================== Interface Isol pour Kibalone ===================

class IsolPlugin:
    """Interface simplifiée pour utiliser Isol dans Kibalone"""
    
    def __init__(self):
        self.isol = load_plugin('isol')
        self._services = {}
    
    def load_service(self, service_name: str):
        """
        Charge un service Isol
        
        Args:
            service_name: Nom du service (ex: "animation_service")
            
        Returns:
            IsolService wrapper
        """
        if service_name in self._services:
            return self._services[service_name]
        
        service = IsolService(service_name)
        self._services[service_name] = service
        return service
    
    @property
    def animation(self):
        """Raccourci vers animation_service"""
        return self.load_service('animation_keyframes')
    
    @property
    def llm(self):
        """Raccourci vers llm_service"""
        return self.load_service('llm_service')
    
    @property
    def rag(self):
        """Raccourci vers rag_service"""
        return self.load_service('rag_service')
    
    @property
    def vision(self):
        """Raccourci vers vision_service"""
        return self.load_service('vision_service')


class IsolService:
    """Wrapper pour appeler un service Isol facilement"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.service_path = self._find_service_path()
    
    def _find_service_path(self) -> Path:
        """Trouve le chemin du service"""
        plugins_dir = get_plugin_manager().plugins_dir
        service_path = plugins_dir / 'isol' / f'{self.service_name}.py'
        
        if not service_path.exists():
            raise FileNotFoundError(f"Service non trouvé: {service_path}")
        
        return service_path
    
    def call(self, params: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Appelle le service avec les paramètres donnés
        
        Args:
            params: Paramètres JSON pour le service
            timeout: Timeout en secondes
            
        Returns:
            Résultat JSON du service
        """
        import subprocess
        import json
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.service_path)],
                input=json.dumps(params),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Service échoué (code {result.returncode})",
                    'stderr': result.stderr
                }
            
            # Parser la dernière ligne (JSON)
            output = json.loads(result.stdout.strip().split('\n')[-1])
            return output
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Timeout après {timeout}s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# =================== Exemple d'utilisation ===================

if __name__ == '__main__':
    print("🔌 Test du système de plugins Kibalone\n")
    
    # Lister les plugins
    manager = get_plugin_manager()
    plugins = manager.list_plugins()
    print(f"📦 Plugins disponibles: {', '.join(plugins)}\n")
    
    # Charger Isol
    isol_interface = IsolPlugin()
    print("✅ Plugin Isol chargé\n")
    
    # Tester un service
    print("🧪 Test du service animation...")
    result = isol_interface.animation.call({
        'prompt': 'test character',
        'num_keyframes': 2,
        'width': 128,
        'height': 128,
        'num_inference_steps': 10,
        'seed': 42
    })
    
    if result['success']:
        print(f"✅ Succès! {result['num_keyframes']} keyframes générées")
    else:
        print(f"❌ Erreur: {result.get('error', 'Inconnue')}")
