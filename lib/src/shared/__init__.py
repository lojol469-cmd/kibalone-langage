# 🔧 Shared Utilities - Utilitaires Partagés

"""Module d'utilitaires partagés pour l'écosystème KIBALI

Contient les classes et fonctions utilitaires utilisées par tous les modules :
- Configuration centralisée
- Logging unifié
- Utilitaires divers
"""

from .config import Config
from .logger import get_logger, log_evenement, log_adaptation, log_urgence, log_performance
from .utils import (
    timer, memoize, clamp, lerp, smooth_step, distance_euclidienne,
    moyenne_mobile, deep_merge, flatten_dict, safe_get,
    load_json_file, save_json_file, load_pickle_file, save_pickle_file,
    hash_string, generate_id, format_duration, parse_duration,
    validate_email, validate_url, generate_random_data,
    calculer_statistiques, detecter_outliers
)

__all__ = [
    "Config",
    "get_logger", "log_evenement", "log_adaptation", "log_urgence", "log_performance",
    "timer", "memoize", "clamp", "lerp", "smooth_step", "distance_euclidienne",
    "moyenne_mobile", "deep_merge", "flatten_dict", "safe_get",
    "load_json_file", "save_json_file", "load_pickle_file", "save_pickle_file",
    "hash_string", "generate_id", "format_duration", "parse_duration",
    "validate_email", "validate_url", "generate_random_data",
    "calculer_statistiques", "detecter_outliers"
]