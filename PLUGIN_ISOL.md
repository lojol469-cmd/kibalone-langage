# 🔌 Plugin Isol - Gestion des Dépendances IA

## 🎯 Qu'est-ce qu'Isol ?

**Isol** (Isolation Service Orchestration Layer) est le **plugin officiel de Kibalone** pour gérer les dépendances IA sans conflits. C'est le moteur qui permet à Kibalone d'orchestrer plusieurs modèles d'IA (Stable Diffusion, LLM, YOLO, etc.) **sans les horribles erreurs de dépendances** !

## 🧬 Architecture

```
Kibalone Framework (langage organique)
    └── Plugin Isol (gestion isolation)
        ├── base.py (classe ServiceBase)
        ├── animation_keyframes.py (Stable Diffusion isolé)
        ├── yolo_service.py (Détection objets isolée)
        ├── llm_service.py (LLM isolé - à venir)
        └── rag_service.py (RAG 3D isolé - à venir)
```

## 🚀 Pourquoi Isol ?

### Le Problème
```python
# ❌ Sans Isol : CONFLIT !
from diffusers import StableDiffusionPipeline  # transformers 4.45
from peft import LoraConfig                    # transformers 4.51
# => ImportError: cannot import name 'MODELS_TO_PIPELINE'
```

### La Solution
```python
# ✅ Avec Isol : 0 CONFLIT !
from isol import AnimationService

service = AnimationService()
frames = service.generate("character walking")
# => Fonctionne ! Service isolé dans son propre processus
```

## 📖 Utilisation avec Kibalone

### 1. Dans une cellule Kibalone

```kibali
cellule GenerateurImage {
    plugin: "isol.animation_keyframes"
    
    action generer(prompt: Texte) -> Image {
        // Isol gère automatiquement l'isolation
        resultat = isol.appeler("generate_keyframes", {
            prompt: prompt,
            num_keyframes: 1
        })
        
        retourner resultat.image
    }
}

// Utilisation
image = GenerateurImage.generer("beautiful sunset")
afficher(image)
```

### 2. Directement en Python

```python
from isol.client import IsolClient

client = IsolClient()

# Générer une animation
result = client.call_service(
    'animation_keyframes',
    action='generate',
    prompt='character dancing',
    num_keyframes=5
)

if result['success']:
    frames = result['frames']
    print(f"✅ {len(frames)} frames générées")
```

## 🛠️ Services Disponibles

| Service | Description | Status |
|---------|-------------|--------|
| `animation_keyframes` | Stable Diffusion 1.5 pour animations | ✅ |
| `yolo_service` | Détection d'objets YOLO11 | ✅ |
| `llm_service` | LLM (Phi-3, Mistral) | 🚧 |
| `rag_service` | RAG 3D avec embeddings | 🚧 |
| `whisper_service` | Speech-to-text | 📝 |
| `tts_service` | Text-to-speech | 📝 |

## 📝 Créer un Nouveau Service

1. Hériter de `ServiceBase` :

```python
# isol/mon_service.py
from isol.base import ServiceBase

class MonService(ServiceBase):
    def process(self, params: dict) -> dict:
        # Votre logique ici
        result = ma_fonction_ia(params['input'])
        
        return {
            'success': True,
            'output': result
        }

if __name__ == '__main__':
    service = MonService()
    service.run()
```

2. Utiliser depuis Kibalone :

```kibali
cellule MonIA {
    plugin: "isol.mon_service"
    
    action traiter(data: Texte) -> Texte {
        resultat = isol.appeler("process", {
            input: data
        })
        retourner resultat.output
    }
}
```

## 🎓 Exemples Complets

Voir les exemples dans `isol/examples/` :
- `example_animation.py` - Générer des animations
- `example_yolo.py` - Détecter des objets
- `example_pipeline.py` - Combiner plusieurs services

## 🔧 Configuration

### Variables d'environnement

```bash
# Timeout par défaut (secondes)
export ISOL_TIMEOUT=300

# Mode debug
export ISOL_DEBUG=1

# Chemin des services
export ISOL_SERVICES_PATH=/path/to/services
```

### Dans Kibalone

```kibali
configuration Isol {
    timeout: 300
    debug: vrai
    cache: vrai
    services_path: "./isol"
}
```

## 🐛 Dépannage

### Service ne répond pas
```bash
# Tester le service directement
echo '{"action":"test"}' | python isol/animation_keyframes.py
```

### Timeout
```python
# Augmenter le timeout
client = IsolClient(timeout=600)
```

### Erreur de dépendances
Les services Isol sont isolés, donc les conflits de dépendances **ne peuvent PAS se produire** ! Si vous avez une erreur, c'est probablement un autre problème.

## 📚 Documentation Complète

- [Guide Isol](../isolated_services/README.md)
- [API Reference](../isolated_services/GUIDE.md)
- [Roadmap](../isolated_services/ROADMAP.md)

## 🤝 Contribuer

1. Créer un nouveau service dans `isol/`
2. Hériter de `ServiceBase`
3. Implémenter `process(params) -> dict`
4. Ajouter des tests
5. Pull Request !

## 📄 Licence

MIT - Créé par lojol469-cmd

---

**Isol rend Kibalone indestructible face aux conflits de dépendances ! 🛡️**
