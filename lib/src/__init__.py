# 🌱 KIBALI Core - Noyau du Système

"""Module principal du framework KIBALI

Ce module contient tous les composants de base nécessaires au
fonctionnement de l'écosystème KIBALI.
"""

# Exports du noyau
from .core import AgentKibali, Ecosysteme, Cellule, Environnement
from .ai import ModelManager, IntelligentAnalyzer, SystemeRAG, RAGTrainer
from .cells import charger_cellule, lister_cellules
from .shared import Config, get_logger

__version__ = "1.0.0"
__author__ = "KIBALI Ecosystem Team"

__all__ = [
    # Core
    'AgentKibali',
    'Ecosysteme',
    'Cellule',
    'Environnement',

    # AI
    'ModelManager',
    'IntelligentAnalyzer',
    'SystemeRAG',
    'RAGTrainer',

    # Cells
    'charger_cellule',
    'lister_cellules',

    # Shared
    'Config',
    'get_logger'
]