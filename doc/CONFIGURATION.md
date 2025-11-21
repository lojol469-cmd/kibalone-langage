# ⚙️ Configuration - Écosystème KIBALI

## Fichier de Configuration Principal

Créer un fichier `config.yaml` ou `config.json` dans le répertoire racine du projet.

### Configuration YAML (Recommandé)

```yaml
# Configuration principale de l'écosystème KIBALI
kibali:
  version: "0.2.0"
  environment: "development"  # development | staging | production

# Configuration de l'agent intelligent
agent:
  name: "AgentKibali"
  strategie: "optimisation_adaptative"  # optimisation_adaptative | survie_maximale | innovation

  # Seuils de décision
  seuils:
    adaptation: 0.7      # Seuil pour déclencher une adaptation (0-1)
    urgence: 0.8         # Seuil pour mode urgence (0-1)
    evolution: 0.9       # Seuil pour évolution majeure (0-1)

  # Stratégies disponibles
  strategies:
    optimisation_adaptative:
      priorite_environnement: 0.6
      priorite_cellules: 0.4
      tolerance_risque: 0.3

    survie_maximale:
      priorite_environnement: 0.8
      priorite_cellules: 0.2
      tolerance_risque: 0.1

    innovation:
      priorite_environnement: 0.4
      priorite_cellules: 0.6
      tolerance_risque: 0.7

# Configuration des modèles IA
ai:
  models:
    codellama:
      model: "codellama/CodeLlama-7b-hf"
      quantization: "4bit"          # 4bit | 8bit | none
      device: "cuda"                # cuda | cpu
      max_tokens: 512
      temperature: 0.7
      cache_dir: "~/.cache/kibali/models"

    phi:
      model: "microsoft/phi-1_5"
      device: "cpu"                 # cpu recommandé pour Phi
      max_tokens: 256
      temperature: 0.3
      cache_dir: "~/.cache/kibali/models"

  # Configuration générale IA
  general:
    fallback_timeout: 30           # secondes
    max_retries: 3
    cache_enabled: true
    cache_ttl: 3600               # secondes

# Configuration de l'écosystème
ecosystem:
  max_cellules: 100
  cycles_max: 1000
  sauvegarde_auto: true
  interval_sauvegarde: 100        # cycles

  # Environnement initial
  environnement_initial:
    temperature: 22.0             # °C
    humidite: 65.0               # %
    luminosite: 70.0             # %
    vent: 5.0                    # km/h
    co2: 400.0                   # ppm

  # Paramètres dynamiques
  dynamique:
    variation_temperature: 5.0   # °C de variation max
    variation_humidite: 20.0     # % de variation max
    cycle_jour_nuit: true
    saisons: true

# Configuration des cellules
cellules:
  types_actifs:
    - "arbre"
    - "climat"
    - "ecureuil"
    - "fleur"

  # Paramètres par défaut
  defaults:
    age_initial: 0
    sante_initiale: 100
    stress_initial: 20

  # Limites biologiques
  limites:
    age_max: 1000                # cycles
    sante_min: 0
    sante_max: 100
    stress_max: 100

# Configuration du système RAG
rag:
  enabled: true
  base_connaissances:
    - "arbres_biology.json"
    - "climat_science.json"
    - "ecureuil_behavior.json"
    - "fleur_biology.json"

  recherche:
    top_k: 5                     # Nombre de résultats max
    seuil_similarite: 0.7        # Seuil de similarité (0-1)
    cache_enabled: true

  embeddings:
    model: "all-MiniLM-L6-v2"    # Modèle d'embeddings
    device: "cpu"
    cache_dir: "~/.cache/kibali/embeddings"

# Configuration de performance
performance:
  cache_analyses: true
  parallel_processing: true
  batch_size: 4
  memory_limit: "8GB"            # Limite mémoire GPU/CPU

  # Optimisations GPU
  gpu:
    enabled: true
    device: 0                    # ID du GPU
    memory_fraction: 0.9         # Fraction mémoire utilisable

  # Monitoring
  monitoring:
    enabled: true
    interval: 10                 # secondes
    metrics:
      - "cpu_usage"
      - "memory_usage"
      - "gpu_usage"
      - "cycle_time"

# Configuration de logging
logging:
  level: "INFO"                  # DEBUG | INFO | WARNING | ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  handlers:
    console:
      enabled: true
      level: "INFO"

    file:
      enabled: true
      level: "DEBUG"
      filename: "logs/kibali.log"
      max_bytes: 10485760         # 10MB
      backup_count: 5

    rotating:
      enabled: false
      filename: "logs/kibali_detailed.log"
      max_bytes: 52428800         # 50MB
      backup_count: 10

# Configuration réseau (pour versions distribuées)
network:
  enabled: false
  host: "localhost"
  port: 5000
  api_key: null                 # Clé API pour sécurisation

  # Configuration multi-agents (futur)
  multi_agent:
    enabled: false
    discovery: "multicast"      # multicast | redis | etcd
    heartbeat_interval: 30      # secondes

# Configuration de développement
development:
  debug_mode: false
  profile_enabled: false
  test_mode: false

  # Outils de développement
  tools:
    profiler: false
    memory_tracker: false
    performance_monitor: true
```

### Configuration JSON (Alternative)

```json
{
  "kibali": {
    "version": "0.2.0",
    "environment": "development"
  },
  "agent": {
    "name": "AgentKibali",
    "strategie": "optimisation_adaptative",
    "seuils": {
      "adaptation": 0.7,
      "urgence": 0.8,
      "evolution": 0.9
    }
  },
  "ai": {
    "models": {
      "codellama": {
        "model": "codellama/CodeLlama-7b-hf",
        "quantization": "4bit",
        "device": "cuda"
      },
      "phi": {
        "model": "microsoft/phi-1_5",
        "device": "cpu"
      }
    }
  }
}
```

## Configurations Spécialisées

### Configuration Haute Performance

```yaml
performance:
  cache_analyses: true
  parallel_processing: true
  batch_size: 8
  memory_limit: "16GB"

gpu:
  enabled: true
  device: 0
  memory_fraction: 0.95

ai:
  models:
    codellama:
      quantization: "4bit"
      device: "cuda"
      max_tokens: 1024
```

### Configuration Recherche Scientifique

```yaml
logging:
  level: "DEBUG"

development:
  debug_mode: true
  profile_enabled: true

monitoring:
  enabled: true
  interval: 5
  metrics:
    - "cpu_usage"
    - "memory_usage"
    - "gpu_usage"
    - "cycle_time"
    - "adaptation_success_rate"
    - "cell_health_distribution"
```

### Configuration Production

```yaml
environment: "production"

logging:
  level: "WARNING"

performance:
  cache_analyses: true
  parallel_processing: true

ecosystem:
  sauvegarde_auto: true
  interval_sauvegarde: 50
```

## Variables d'Environnement

Vous pouvez également utiliser des variables d'environnement :

```bash
# Configuration IA
export KIBALI_AI_CODELLAMA_MODEL="codellama/CodeLlama-7b-hf"
export KIBALI_AI_PHI_MODEL="microsoft/phi-1_5"
export KIBALI_AI_DEVICE="cuda"

# Performance
export KIBALI_GPU_ENABLED="true"
export KIBALI_MEMORY_LIMIT="8GB"

# Logging
export KIBALI_LOG_LEVEL="INFO"
export KIBALI_LOG_FILE="logs/kibali.log"

# Écosystème
export KIBALI_MAX_CELLS="100"
export KIBALI_AUTO_SAVE="true"
```

## Validation de Configuration

Le système valide automatiquement la configuration au démarrage :

```python
from kibali.config import ConfigValidator

validator = ConfigValidator()
is_valid, errors = validator.validate_config(config)

if not is_valid:
    for error in errors:
        print(f"Erreur de configuration: {error}")
```

## Exemples de Configurations

### Configuration Minimale

```yaml
kibali:
  version: "0.2.0"

agent:
  strategie: "optimisation_adaptative"

ai:
  models:
    codellama:
      model: "codellama/CodeLlama-7b-hf"
      device: "cpu"  # Pour machines sans GPU
```

### Configuration Avancée

```yaml
kibali:
  version: "0.2.0"
  environment: "production"

agent:
  strategie: "innovation"
  seuils:
    adaptation: 0.8
    urgence: 0.9

ai:
  models:
    codellama:
      quantization: "4bit"
      device: "cuda"
      temperature: 0.5
    phi:
      device: "cuda"  # Utilisation GPU si disponible

performance:
  parallel_processing: true
  batch_size: 8

rag:
  enabled: true
  recherche:
    top_k: 10
```

---

*Cette configuration flexible permet d'adapter KIBALI à différents environnements et cas d'usage.*