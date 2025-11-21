# 📝 Logger - Système de Logging Unifié

"""Module de logging unifié pour l'écosystème KIBALI

Fournit un système de logging configurable et centralisé pour tous les composants :
- Configuration flexible des niveaux de log
- Rotation automatique des fichiers
- Formatage cohérent
- Intégration avec la configuration globale
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import sys

from .config import Config

class KIBALILogger:
    """Classe principale de gestion du logging KIBALI

    Configure et gère tous les loggers de l'écosystème avec :
    - Niveaux configurables
    - Rotation des fichiers
    - Formatage cohérent
    - Filtres personnalisés
    """

    _instance = None
    _loggers: Dict[str, logging.Logger] = {}

    def __new__(cls, config: Optional[Config] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Config] = None):
        if self._initialized:
            return

        self.config = config or Config()
        self._setup_logging()
        self._initialized = True

    def _setup_logging(self) -> None:
        """Configure le système de logging"""
        # Création du répertoire de logs
        log_dir = Path(self.config.logging.fichier_log).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configuration du formatter
        formatter = logging.Formatter(
            self.config.logging.format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler pour fichier avec rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.config.logging.fichier_log,
            maxBytes=self.config.logging.taille_max_fichier,
            backupCount=self.config.logging.nombre_fichiers_backup,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        # Handler pour console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Configuration du logger racine
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.logging.niveau))

        # Suppression des handlers existants
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Ajout des nouveaux handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Logger spécifique pour KIBALI
        kibali_logger = logging.getLogger('kibali')
        kibali_logger.setLevel(getattr(logging, self.config.logging.niveau))

        # Filtre personnalisé pour les messages d'urgence
        class UrgenceFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'urgence') or record.levelno >= logging.ERROR

        # Handler spécial pour les urgences (vers stderr)
        urgence_handler = logging.StreamHandler(sys.stderr)
        urgence_handler.setFormatter(formatter)
        urgence_handler.addFilter(UrgenceFilter())
        urgence_handler.setLevel(logging.WARNING)

        root_logger.addHandler(urgence_handler)

    def get_logger(self, nom: str) -> logging.Logger:
        """Obtient ou crée un logger pour un module

        Args:
            nom: Nom du module/logger

        Returns:
            logging.Logger: Logger configuré
        """
        if nom not in self._loggers:
            logger = logging.getLogger(nom)

            # Configuration spécifique selon le type de module
            if nom.startswith('Cellule'):
                logger.setLevel(logging.DEBUG if self.config.logging.niveau == 'DEBUG' else logging.INFO)
            elif nom.startswith('Agent'):
                logger.setLevel(logging.DEBUG)
            elif nom.startswith('IA'):
                logger.setLevel(logging.INFO)
            else:
                logger.setLevel(getattr(logging, self.config.logging.niveau))

            self._loggers[nom] = logger

        return self._loggers[nom]

    def log_evenement(self, niveau: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log un événement avec des informations supplémentaires

        Args:
            niveau: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Message à logger
            extra: Informations supplémentaires
        """
        logger = self.get_logger('evenements')
        extra = extra or {}

        if niveau.upper() == 'DEBUG':
            logger.debug(message, extra=extra)
        elif niveau.upper() == 'INFO':
            logger.info(message, extra=extra)
        elif niveau.upper() == 'WARNING':
            logger.warning(message, extra=extra)
        elif niveau.upper() == 'ERROR':
            logger.error(message, extra=extra)
        elif niveau.upper() == 'CRITICAL':
            logger.critical(message, extra=extra)

    def log_adaptation(self, cellule_nom: str, type_adaptation: str, succes: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Log une adaptation cellulaire

        Args:
            cellule_nom: Nom de la cellule
            type_adaptation: Type d'adaptation
            succes: Succès de l'adaptation
            details: Détails supplémentaires
        """
        status = "SUCCES" if succes else "ECHEC"
        message = f"Adaptation {type_adaptation} pour {cellule_nom}: {status}"

        extra = {
            'cellule': cellule_nom,
            'adaptation': type_adaptation,
            'succes': succes,
            'details': details or {}
        }

        self.log_evenement('INFO', message, extra)

    def log_urgence(self, type_urgence: str, cellule_nom: str, description: str) -> None:
        """Log un événement d'urgence

        Args:
            type_urgence: Type d'urgence
            cellule_nom: Nom de la cellule concernée
            description: Description de l'urgence
        """
        message = f"URGENCE {type_urgence.upper()} - {cellule_nom}: {description}"

        extra = {
            'urgence': True,
            'type': type_urgence,
            'cellule': cellule_nom,
            'description': description
        }

        self.log_evenement('CRITICAL', message, extra)

    def log_performance(self, operation: str, duree: float, succes: bool) -> None:
        """Log les performances d'une opération

        Args:
            operation: Nom de l'opération
            duree: Durée en secondes
            succes: Succès de l'opération
        """
        status = "OK" if succes else "FAIL"
        message = f"PERF {operation}: {duree:.3f}s ({status})"

        extra = {
            'operation': operation,
            'duree': duree,
            'succes': succes
        }

        niveau = 'INFO' if succes else 'WARNING'
        self.log_evenement(niveau, message, extra)

# Instance globale
_logger_instance = None

def get_logger(nom: str) -> logging.Logger:
    """Fonction utilitaire pour obtenir un logger

    Args:
        nom: Nom du logger

    Returns:
        logging.Logger: Logger configuré
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = KIBALILogger()
    return _logger_instance.get_logger(nom)

def log_evenement(niveau: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Fonction utilitaire pour logger un événement"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = KIBALILogger()
    _logger_instance.log_evenement(niveau, message, extra)

def log_adaptation(cellule_nom: str, type_adaptation: str, succes: bool, details: Optional[Dict[str, Any]] = None) -> None:
    """Fonction utilitaire pour logger une adaptation"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = KIBALILogger()
    _logger_instance.log_adaptation(cellule_nom, type_adaptation, succes, details)

def log_urgence(type_urgence: str, cellule_nom: str, description: str) -> None:
    """Fonction utilitaire pour logger une urgence"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = KIBALILogger()
    _logger_instance.log_urgence(type_urgence, cellule_nom, description)

def log_performance(operation: str, duree: float, succes: bool) -> None:
    """Fonction utilitaire pour logger les performances"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = KIBALILogger()
    _logger_instance.log_performance(operation, duree, succes)