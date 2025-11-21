# ⚙️ Configuration - Paramètres Globaux du Système KIBALI

"""Module de configuration centralisée pour l'écosystème KIBALI

Ce module définit tous les paramètres de configuration utilisés par :
- Les cellules autonomes
- L'écosystème
- L'agent intelligent
- Les modèles d'IA
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime

@dataclass
class ConfigCellule:
    """Configuration spécifique aux cellules"""
    vieillissement_base: float = 0.5
    regeneration_base: float = 0.3
    consommation_energie_base: float = 2.0
    distance_interaction_max: float = 5.0
    taille_memoire_max: int = 100
    taille_historique_max: int = 1000

    # Limites des états internes
    limites_etats: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=lambda: {
        "sante": {"min": 0, "max": 100},
        "stress": {"min": 0, "max": 100},
        "energie": {"min": 0, "max": 100},
        "age": {"min": 0, "max": 1000}
    })

@dataclass
class ConfigEcosysteme:
    """Configuration de l'écosystème"""
    taille_max_cellules: int = 1000
    cycles_evolution_par_jour: int = 24
    taux_mutation: float = 0.01
    taux_extinction: float = 0.001
    facteur_stress_global: float = 1.0

    # Conditions environnementales par défaut
    conditions_base: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 20.0,
        "humidite": 60.0,
        "luminosite": 1000.0,
        "qualite_air": 80.0,
        "nutriments_sol": 50.0
    })

@dataclass
class ConfigIA:
    """Configuration des modèles d'IA"""
    # Code Llama
    codellama_model_path: str = "models/codellama-7b-instruct"
    codellama_quantization: str = "4bit"
    codellama_max_tokens: int = 512
    codellama_temperature: float = 0.7

    # Phi-1.5
    phi_model_path: str = "models/phi-1.5"
    phi_max_tokens: int = 256
    phi_temperature: float = 0.8

    # RAG
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class ConfigAgent:
    """Configuration de l'agent intelligent"""
    cycle_orchestration_interval: int = 60  # secondes
    seuil_stress_critique: float = 70.0
    seuil_sante_critique: float = 30.0
    facteur_adaptation_max: float = 2.0

    # Stratégies d'intervention
    strategies_intervention: List[str] = field(default_factory=lambda: [
        "adaptation_cellulaire",
        "modification_environnementale",
        "introduction_nouvelles_cellules",
        "optimisation_ressources"
    ])

@dataclass
class ConfigLogging:
    """Configuration du système de logging"""
    niveau: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    fichier_log: str = "logs/kibali.log"
    taille_max_fichier: int = 10 * 1024 * 1024  # 10MB
    nombre_fichiers_backup: int = 5

@dataclass
class ConfigPerformance:
    """Configuration des performances"""
    multithreading_active: bool = True
    nombre_threads_max: int = 4
    cache_active: bool = True
    taille_cache_max: int = 1000
    optimisation_gpu: bool = True

class Config:
    """Classe principale de configuration

    Centralise tous les paramètres de configuration du système KIBALI.
    Peut être chargée depuis un fichier JSON ou configurée programmatiquement.
    """

    def __init__(self,
                 fichier_config: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None):
        """Initialise la configuration

        Args:
            fichier_config: Chemin vers un fichier de configuration JSON
            config_dict: Dictionnaire de configuration directe
        """
        # Configurations par défaut
        self.cellule = ConfigCellule()
        self.ecosysteme = ConfigEcosysteme()
        self.ia = ConfigIA()
        self.agent = ConfigAgent()
        self.logging = ConfigLogging()
        self.performance = ConfigPerformance()

        # Métadonnées
        self.version = "1.0.0"
        self.date_creation = datetime.now()
        self.environnement = "development"

        # Chargement depuis fichier ou dict
        if fichier_config:
            self.charger_depuis_fichier(fichier_config)
        elif config_dict:
            self.charger_depuis_dict(config_dict)

        # Validation
        self._valider_configuration()

        # Logger temporaire pour l'initialisation
        logging.basicConfig(level=getattr(logging, self.logging.niveau))
        self.logger = logging.getLogger("Config")

        self.logger.info("Configuration KIBALI initialisée")

    def charger_depuis_fichier(self, chemin: str) -> None:
        """Charge la configuration depuis un fichier JSON

        Args:
            chemin: Chemin vers le fichier de configuration
        """
        try:
            with open(chemin, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.charger_depuis_dict(data)
            self.logger.info(f"Configuration chargée depuis {chemin}")
        except FileNotFoundError:
            self.logger.warning(f"Fichier de configuration {chemin} non trouvé, utilisation des valeurs par défaut")
        except json.JSONDecodeError as e:
            self.logger.error(f"Erreur de parsing JSON dans {chemin}: {e}")
            raise

    def charger_depuis_dict(self, data: Dict[str, Any]) -> None:
        """Charge la configuration depuis un dictionnaire

        Args:
            data: Dictionnaire de configuration
        """
        # Mise à jour des sous-configurations
        if "cellule" in data:
            for key, value in data["cellule"].items():
                if hasattr(self.cellule, key):
                    setattr(self.cellule, key, value)

        if "ecosysteme" in data:
            for key, value in data["ecosysteme"].items():
                if hasattr(self.ecosysteme, key):
                    setattr(self.ecosysteme, key, value)

        if "ia" in data:
            for key, value in data["ia"].items():
                if hasattr(self.ia, key):
                    setattr(self.ia, key, value)

        if "agent" in data:
            for key, value in data["agent"].items():
                if hasattr(self.agent, key):
                    setattr(self.agent, key, value)

        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)

        if "performance" in data:
            for key, value in data["performance"].items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)

        # Métadonnées
        if "version" in data:
            self.version = data["version"]
        if "environnement" in data:
            self.environnement = data["environnement"]

    def sauvegarder_vers_fichier(self, chemin: str) -> None:
        """Sauvegarde la configuration vers un fichier JSON

        Args:
            chemin: Chemin où sauvegarder la configuration
        """
        # Création du répertoire si nécessaire
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)

        # Conversion en dictionnaire
        data = {
            "version": self.version,
            "environnement": self.environnement,
            "date_creation": self.date_creation.isoformat(),
            "cellule": {
                "vieillissement_base": self.cellule.vieillissement_base,
                "regeneration_base": self.cellule.regeneration_base,
                "consommation_energie_base": self.cellule.consommation_energie_base,
                "distance_interaction_max": self.cellule.distance_interaction_max,
                "taille_memoire_max": self.cellule.taille_memoire_max,
                "taille_historique_max": self.cellule.taille_historique_max,
                "limites_etats": self.cellule.limites_etats
            },
            "ecosysteme": {
                "taille_max_cellules": self.ecosysteme.taille_max_cellules,
                "cycles_evolution_par_jour": self.ecosysteme.cycles_evolution_par_jour,
                "taux_mutation": self.ecosysteme.taux_mutation,
                "taux_extinction": self.ecosysteme.taux_extinction,
                "facteur_stress_global": self.ecosysteme.facteur_stress_global,
                "conditions_base": self.ecosysteme.conditions_base
            },
            "ia": {
                "codellama_model_path": self.ia.codellama_model_path,
                "codellama_quantization": self.ia.codellama_quantization,
                "codellama_max_tokens": self.ia.codellama_max_tokens,
                "codellama_temperature": self.ia.codellama_temperature,
                "phi_model_path": self.ia.phi_model_path,
                "phi_max_tokens": self.ia.phi_max_tokens,
                "phi_temperature": self.ia.phi_temperature,
                "rag_chunk_size": self.ia.rag_chunk_size,
                "rag_chunk_overlap": self.ia.rag_chunk_overlap,
                "rag_top_k": self.ia.rag_top_k,
                "rag_embedding_model": self.ia.rag_embedding_model
            },
            "agent": {
                "cycle_orchestration_interval": self.agent.cycle_orchestration_interval,
                "seuil_stress_critique": self.agent.seuil_stress_critique,
                "seuil_sante_critique": self.agent.seuil_sante_critique,
                "facteur_adaptation_max": self.agent.facteur_adaptation_max,
                "strategies_intervention": self.agent.strategies_intervention
            },
            "logging": {
                "niveau": self.logging.niveau,
                "format": self.logging.format,
                "fichier_log": self.logging.fichier_log,
                "taille_max_fichier": self.logging.taille_max_fichier,
                "nombre_fichiers_backup": self.logging.nombre_fichiers_backup
            },
            "performance": {
                "multithreading_active": self.performance.multithreading_active,
                "nombre_threads_max": self.performance.nombre_threads_max,
                "cache_active": self.performance.cache_active,
                "taille_cache_max": self.performance.taille_cache_max,
                "optimisation_gpu": self.performance.optimisation_gpu
            }
        }

        # Sauvegarde
        with open(chemin, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Configuration sauvegardée vers {chemin}")

    def _valider_configuration(self) -> None:
        """Valide la cohérence de la configuration"""
        erreurs = []

        # Validation des limites
        for etat, limites in self.cellule.limites_etats.items():
            if limites["min"] >= limites["max"]:
                erreurs.append(f"Limites invalides pour {etat}: min >= max")

        # Validation des chemins d'IA
        if not self.ia.codellama_model_path:
            erreurs.append("Chemin Code Llama requis")
        if not self.ia.phi_model_path:
            erreurs.append("Chemin Phi requis")

        # Validation des seuils
        if not (0 <= self.agent.seuil_stress_critique <= 100):
            erreurs.append("Seuil stress critique invalide")
        if not (0 <= self.agent.seuil_sante_critique <= 100):
            erreurs.append("Seuil santé critique invalide")

        if erreurs:
            raise ValueError(f"Erreurs de configuration: {', '.join(erreurs)}")

    def obtenir_config_par_defaut(self) -> Dict[str, Any]:
        """Retourne la configuration par défaut

        Returns:
            Dict: Configuration par défaut
        """
        return {
            "cellule": {
                "vieillissement_base": 0.5,
                "regeneration_base": 0.3,
                "consommation_energie_base": 2.0,
                "distance_interaction_max": 5.0,
                "taille_memoire_max": 100,
                "taille_historique_max": 1000
            },
            "ecosysteme": {
                "taille_max_cellules": 1000,
                "cycles_evolution_par_jour": 24,
                "taux_mutation": 0.01,
                "taux_extinction": 0.001,
                "facteur_stress_global": 1.0
            },
            "ia": {
                "codellama_model_path": "models/codellama-7b-instruct",
                "codellama_quantization": "4bit",
                "codellama_max_tokens": 512,
                "codellama_temperature": 0.7,
                "phi_model_path": "models/phi-1.5",
                "phi_max_tokens": 256,
                "phi_temperature": 0.8
            },
            "agent": {
                "cycle_orchestration_interval": 60,
                "seuil_stress_critique": 70.0,
                "seuil_sante_critique": 30.0,
                "facteur_adaptation_max": 2.0
            },
            "logging": {
                "niveau": "INFO",
                "fichier_log": "logs/kibali.log"
            },
            "performance": {
                "multithreading_active": True,
                "cache_active": True,
                "optimisation_gpu": True
            }
        }

    def __repr__(self) -> str:
        return f"Config(version='{self.version}', env='{self.environnement}')"