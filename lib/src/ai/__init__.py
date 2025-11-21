# 🧠 AI Module - Intelligence Artificielle KIBALI

"""Module d'intelligence artificielle pour KIBALI

Ce module contient tous les composants d'IA :
- Gestion des modèles de langage (CodeLlama, Phi, etc.)
- Système RAG pour la recherche intelligente
- Analyseurs intelligents pour l'orchestration
- Entraînement et optimisation des modèles
"""

from .models import ModelManager, BaseModel, CodeLlamaModel, PhiModel
from .rag import SystemeRAG, DocumentConnaissance, ContexteRAG
from .analyzer import IntelligentAnalyzer
from .trainer import RAGTrainer, ConfigurationEntrainement

__all__ = [
    # Gestion des modèles
    "ModelManager",
    "BaseModel", 
    "CodeLlamaModel",
    "PhiModel",
    
    # Système RAG
    "SystemeRAG",
    "DocumentConnaissance",
    "ContexteRAG",
    
    # Analyse intelligente
    "IntelligentAnalyzer",
    
    # Entraînement RAG
    "RAGTrainer",
    "ConfigurationEntrainement"
]