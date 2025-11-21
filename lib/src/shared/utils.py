# 🛠️ Utils - Utilitaires Divers

"""Module d'utilitaires divers pour l'écosystème KIBALI

Fournit des fonctions utilitaires générales :
- Manipulation de données
- Calculs mathématiques
- Gestion des fichiers
- Fonctions d'aide diverses
"""

import json
import pickle
import hashlib
import random
from typing import Any, Dict, List, Optional, Union, TypeVar, Callable
from pathlib import Path
from datetime import datetime, timedelta
import time
import math
import statistics
from functools import wraps

T = TypeVar('T')

# Décorateurs utilitaires
def timer(func: Callable) -> Callable:
    """Décorateur pour mesurer le temps d'exécution d'une fonction

    Args:
        func: Fonction à décorer

    Returns:
        Callable: Fonction décorée
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        debut = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            fin = time.time()
            duree = fin - debut
            print(f"{func.__name__} exécuté en {duree:.3f} secondes")
    return wrapper

def memoize(maxsize: int = 128):
    """Décorateur de memoization avec limite de taille

    Args:
        maxsize: Taille maximale du cache

    Returns:
        Callable: Décorateur de memoization
    """
    def decorator(func: Callable) -> Callable:
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Création d'une clé de cache
            key = (args, tuple(sorted(kwargs.items())))

            if key in cache:
                return cache[key]

            result = func(*args, **kwargs)

            if len(cache) >= maxsize:
                # Suppression d'une entrée aléatoire
                cache.pop(random.choice(list(cache.keys())))

            cache[key] = result
            return result

        return wrapper
    return decorator

# Fonctions mathématiques
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Contraint une valeur entre min et max

    Args:
        value: Valeur à contraindre
        min_val: Valeur minimale
        max_val: Valeur maximale

    Returns:
        float: Valeur contrainte
    """
    return max(min_val, min(max_val, value))

def lerp(a: float, b: float, t: float) -> float:
    """Interpolation linéaire entre deux valeurs

    Args:
        a: Valeur de départ
        b: Valeur d'arrivée
        t: Facteur d'interpolation (0-1)

    Returns:
        float: Valeur interpolée
    """
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t

def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """Fonction de lissage smoothstep

    Args:
        edge0: Bord gauche
        edge1: Bord droit
        x: Valeur à lisser

    Returns:
        float: Valeur lissée (0-1)
    """
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)

def distance_euclidienne(p1: tuple, p2: tuple) -> float:
    """Calcule la distance euclidienne entre deux points

    Args:
        p1: Premier point (x, y)
        p2: Deuxième point (x, y)

    Returns:
        float: Distance euclidienne
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def moyenne_mobile(data: List[float], window: int) -> List[float]:
    """Calcule la moyenne mobile d'une série de données

    Args:
        data: Série de données
        window: Taille de la fenêtre

    Returns:
        List[float]: Moyennes mobiles
    """
    if len(data) < window:
        return []

    moyennes = []
    for i in range(len(data) - window + 1):
        fenetre = data[i:i + window]
        moyennes.append(statistics.mean(fenetre))

    return moyennes

# Fonctions de manipulation de données
def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Fusion profonde de deux dictionnaires

    Args:
        dict1: Premier dictionnaire
        dict2: Deuxième dictionnaire

    Returns:
        Dict[str, Any]: Dictionnaire fusionné
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result

def flatten_dict(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Aplatit un dictionnaire imbriqué

    Args:
        d: Dictionnaire à aplatir
        prefix: Préfixe pour les clés

    Returns:
        Dict[str, Any]: Dictionnaire aplati
    """
    items = []
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def safe_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Accès sécurisé à une valeur imbriquée dans un dictionnaire

    Args:
        d: Dictionnaire
        keys: Liste des clés
        default: Valeur par défaut

    Returns:
        Any: Valeur trouvée ou défaut
    """
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

# Fonctions de gestion de fichiers
def load_json_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Charge un fichier JSON de manière sécurisée

    Args:
        filepath: Chemin vers le fichier

    Returns:
        Dict[str, Any]: Contenu du fichier

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        json.JSONDecodeError: Si le JSON est invalide
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """Sauvegarde des données dans un fichier JSON

    Args:
        data: Données à sauvegarder
        filepath: Chemin vers le fichier
        indent: Indentation du JSON
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_pickle_file(filepath: Union[str, Path]) -> Any:
    """Charge un fichier pickle

    Args:
        filepath: Chemin vers le fichier

    Returns:
        Any: Objet désérialisé
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")

    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle_file(data: Any, filepath: Union[str, Path]) -> None:
    """Sauvegarde des données dans un fichier pickle

    Args:
        data: Données à sauvegarder
        filepath: Chemin vers le fichier
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Fonctions de hachage et sécurité
def hash_string(text: str, algorithm: str = 'sha256') -> str:
    """Calcule le hash d'une chaîne de caractères

    Args:
        text: Texte à hasher
        algorithm: Algorithme de hash

    Returns:
        str: Hash hexadécimal
    """
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'sha512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Algorithme non supporté: {algorithm}")

    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()

def generate_id(prefix: str = "", length: int = 8) -> str:
    """Génère un identifiant unique

    Args:
        prefix: Préfixe optionnel
        length: Longueur de l'identifiant

    Returns:
        str: Identifiant généré
    """
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    random_part = ''.join(random.choice(chars) for _ in range(length))
    timestamp = str(int(time.time() * 1000))[-6:]  # 6 derniers chiffres du timestamp

    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    else:
        return f"{timestamp}_{random_part}"

# Fonctions temporelles
def format_duration(seconds: float) -> str:
    """Formate une durée en secondes de manière lisible

    Args:
        seconds: Durée en secondes

    Returns:
        str: Durée formatée
    """
    if seconds < 1:
        return ".1f"
    elif seconds < 60:
        return ".1f"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return ".1f"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"
    else:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}j{hours}h"

def parse_duration(duration_str: str) -> float:
    """Parse une durée formatée en secondes

    Args:
        duration_str: Durée formatée (ex: "1h30m", "45s", "2j")

    Returns:
        float: Durée en secondes
    """
    import re

    # Patterns pour différents formats
    patterns = [
        (r'(\d+)j', 86400),    # jours
        (r'(\d+)h', 3600),     # heures
        (r'(\d+)m', 60),       # minutes
        (r'(\d+)s', 1),        # secondes
        (r'(\d+(?:\.\d+)?)', 1)  # nombre seul = secondes
    ]

    total_seconds = 0.0

    for pattern, multiplier in patterns:
        matches = re.findall(pattern, duration_str)
        for match in matches:
            total_seconds += float(match) * multiplier

    return total_seconds

# Fonctions de validation
def validate_email(email: str) -> bool:
    """Valide une adresse email

    Args:
        email: Adresse email à valider

    Returns:
        bool: True si valide
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """Valide une URL

    Args:
        url: URL à valider

    Returns:
        bool: True si valide
    """
    import re
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))*)?$'
    return bool(re.match(pattern, url))

# Fonctions de génération de données
def generate_random_data(size: int, data_type: str = 'float', **kwargs) -> List[Any]:
    """Génère des données aléatoires

    Args:
        size: Nombre d'éléments à générer
        data_type: Type de données ('int', 'float', 'str', 'bool')
        **kwargs: Paramètres supplémentaires

    Returns:
        List[Any]: Liste de données générées
    """
    if data_type == 'int':
        min_val = kwargs.get('min', 0)
        max_val = kwargs.get('max', 100)
        return [random.randint(min_val, max_val) for _ in range(size)]

    elif data_type == 'float':
        min_val = kwargs.get('min', 0.0)
        max_val = kwargs.get('max', 1.0)
        return [random.uniform(min_val, max_val) for _ in range(size)]

    elif data_type == 'str':
        length = kwargs.get('length', 10)
        chars = kwargs.get('chars', 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return [''.join(random.choice(chars) for _ in range(length)) for _ in range(size)]

    elif data_type == 'bool':
        return [random.choice([True, False]) for _ in range(size)]

    else:
        raise ValueError(f"Type de données non supporté: {data_type}")

# Fonctions statistiques
def calculer_statistiques(data: List[float]) -> Dict[str, float]:
    """Calcule les statistiques de base d'une série de données

    Args:
        data: Série de données numériques

    Returns:
        Dict[str, float]: Statistiques calculées
    """
    if not data:
        return {}

    return {
        'count': len(data),
        'mean': statistics.mean(data),
        'median': statistics.median(data),
        'std': statistics.stdev(data) if len(data) > 1 else 0,
        'min': min(data),
        'max': max(data),
        'q25': statistics.quantiles(data, n=4)[0] if len(data) >= 4 else min(data),
        'q75': statistics.quantiles(data, n=4)[2] if len(data) >= 4 else max(data)
    }

def detecter_outliers(data: List[float], seuil: float = 1.5) -> List[int]:
    """Détecte les outliers dans une série de données (méthode IQR)

    Args:
        data: Série de données
        seuil: Seuil de détection (multiplicateur de l'IQR)

    Returns:
        List[int]: Indices des outliers
    """
    if len(data) < 4:
        return []

    q1, q3 = statistics.quantiles(data, n=4)[0], statistics.quantiles(data, n=4)[2]
    iqr = q3 - q1

    limite_inf = q1 - seuil * iqr
    limite_sup = q3 + seuil * iqr

    outliers = []
    for i, valeur in enumerate(data):
        if valeur < limite_inf or valeur > limite_sup:
            outliers.append(i)

    return outliers