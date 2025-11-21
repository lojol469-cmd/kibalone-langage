# 🌱 KIBALI - Écosystème d'Intelligence Autonome

"""Framework KIBALI - Intelligence Artificielle pour Écosystèmes Autonomes

Ce package fournit les composants principaux de l'écosystème KIBALI,
un système d'intelligence artificielle conçu pour orchestrer des écosystèmes
biologiques autonomes utilisant des modèles de langage avancés.
"""

__version__ = "0.2.0"
__author__ = "Écosystème KIBALI"
__description__ = "Intelligence Artificielle pour Écosystèmes Autonomes"

# Imports principaux
from .src.core.agent import AgentKibali
from .src.core.ecosystem import Ecosysteme
from .src.core.cellule import Cellule
from .src.core.environment import Environnement

# Exports publics
__all__ = [
    'AgentKibali',
    'Ecosysteme',
    'Cellule',
    'Environnement'
]

def get_version():
    """Retourne la version du framework"""
    return __version__

def create_ecosystem():
    """Crée un nouvel écosystème KIBALI"""
    return Ecosysteme()

def create_agent(strategie="optimisation_adaptative"):
    """Crée un nouvel agent KIBALI"""
    return AgentKibali(strategie=strategie)