# 🧠 Core - Noyau du Système KIBALI

"""Composants de base du framework KIBALI

Ce module contient les classes fondamentales :
- AgentKibali : Orchestrateur intelligent
- Ecosysteme : Gestionnaire d'écosystème
- Cellule : Unité biologique autonome
- Environnement : Conditions environnementales
"""

from .agent import AgentKibali
from .ecosystem import Ecosysteme
from .cellule import Cellule
from .environment import Environnement

__all__ = [
    'AgentKibali',
    'Ecosysteme',
    'Cellule',
    'Environnement'
]