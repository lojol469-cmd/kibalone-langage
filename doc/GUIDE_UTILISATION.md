# 🚀 Guide d'Utilisation - Écosystème KIBALI

## Démarrage Rapide

### Prérequis

```bash
# Installation des dépendances Python
pip install -r requirements.txt

# Téléchargement des modèles IA
python download_llm.py

# Installation de KIBALI
./install.sh  # Linux/Mac
# ou
install_windows.bat  # Windows
```

### Lancement de l'Écosystème

```bash
# Mode démonstration complet
python demo_kibali.sh

# Mode agent intelligent seul
python kibali_cmd.py --mode agent

# Mode simulation autonome
python run.py --ecosystem --cycles 100
```

## Exemples d'Utilisation

### 1. Création d'une Nouvelle Cellule

```python
from kibali import Cellule, EtatInterne, ObjetPhysique

# Création d'une cellule personnalisée
nouvelle_cellule = Cellule(
    nom="MonArbre",
    type="vegetal",
    etats_internes={
        "sante": EtatInterne(valeur=100, unite="%"),
        "age": EtatInterne(valeur=5, unite="ans"),
        "stress": EtatInterne(valeur=20, unite="%")
    },
    objets_physiques={
        "racines": ObjetPhysique(type="structure", etat="fonctionnel"),
        "feuilles": ObjetPhysique(type="organe", etat="sain")
    }
)

# Ajout à l'écosystème
ecosysteme.ajouter_cellule(nouvelle_cellule)
```

### 2. Configuration de l'Agent KIBALI

```python
from kibali_agent import AgentKibali

# Configuration de l'agent
agent = AgentKibali(
    modeles_ia={
        "codellama": "codellama/CodeLlama-7b-hf",
        "phi": "microsoft/phi-1_5"
    },
    strategie="optimisation_adaptative",
    seuil_urgence=0.8
)

# Démarrage de l'orchestration
agent.demarrer_orchestration()
```

### 3. Simulation d'un Scénario de Stress

```python
from environnement import Environnement

# Création d'un environnement stressant
environnement = Environnement(
    temperature=40,  # Canicule
    humidite=20,     # Sécheresse
    luminosite=90,   # Forte exposition solaire
    vent=15          # Vent fort
)

# Simulation sur plusieurs cycles
for cycle in range(50):
    # L'agent analyse et adapte automatiquement
    rapport = agent.cycle_orchestration()

    # Affichage des adaptations
    print(f"Cycle {cycle}: {len(rapport.adaptations)} adaptations appliquées")

    # Vérification de la stabilité
    if rapport.stress_global > 0.9:
        print("⚠️  Seuil d'urgence atteint!")
        break
```

### 4. Utilisation du Système RAG

```python
from rag_system import RAGSystem

# Initialisation du système RAG
rag = RAGSystem(
    base_connaissances=[
        "arbres_biology.json",
        "climat_science.json",
        "ecureuil_behavior.json"
    ]
)

# Recherche d'informations pertinentes
connaissances = rag.rechercher(
    query="adaptation des arbres à la sécheresse",
    top_k=5
)

# Utilisation dans l'analyse IA
analyse = agent.analyser_avec_contexte(
    situation="arbre en stress hydrique",
    connaissances=connaissances
)
```

## Commandes Avancées

### Mode Interactif

```bash
# Lancement du mode interactif
python kibali_cmd.py --interactive

# Commandes disponibles:
# help - Affiche l'aide
# status - État de l'écosystème
# add_cell <type> <nom> - Ajouter une cellule
# set_env <param> <valeur> - Modifier l'environnement
# run_cycle - Exécuter un cycle d'orchestration
# save_state - Sauvegarder l'état
# load_state <fichier> - Charger un état
# exit - Quitter
```

### Configuration Personnalisée

```python
# Configuration avancée
config = {
    "agent": {
        "modeles_ia": {
            "codellama": {
                "model": "codellama/CodeLlama-7b-hf",
                "quantization": "4bit",
                "device": "cuda"
            },
            "phi": {
                "model": "microsoft/phi-1_5",
                "device": "cpu"
            }
        },
        "strategies": {
            "optimisation": {
                "seuil_adaptation": 0.7,
                "priorite_environnement": 0.6
            }
        }
    },
    "ecosysteme": {
        "max_cellules": 100,
        "cycles_max": 1000,
        "sauvegarde_auto": True
    },
    "performance": {
        "cache_analyses": True,
        "parallel_processing": True,
        "memory_limit": "8GB"
    }
}

# Application de la configuration
agent.appliquer_configuration(config)
```

## Scénarios d'Exemple

### Scénario 1: Forêt en Santé

```python
# Configuration initiale saine
ecosysteme = Ecosysteme()
ecosysteme.configurer_environnement(
    temperature=22,
    humidite=65,
    luminosite=70
)

# Ajout de cellules équilibrées
ecosysteme.ajouter_cellule(Arbre("Chêne", age=50))
ecosysteme.ajouter_cellule(Arbre("Sapin", age=30))
ecosysteme.ajouter_cellule(Ecureuil("Noisette"))
ecosysteme.ajouter_cellule(Fleur("Rose", saison="printemps"))

# Simulation équilibrée
agent.orchestrer_cycles(100)
```

### Scénario 2: Gestion d'une Crise Climatique

```python
# Simulation de canicule
environnement.modifier(
    temperature=38,
    humidite=25,
    vent=20
)

# L'agent détecte automatiquement le stress
while agent.detecter_urgence():
    # Analyse de la situation
    situation = agent.analyser_situation()

    # Application d'adaptations d'urgence
    adaptations = agent.appliquer_strategie_urgence(situation)

    # Suivi des effets
    effets = agent.mesurer_effets_adaptations(adaptations)

    print(f"Adaptations appliquées: {len(adaptations)}")
    print(f"Réduction du stress: {effets.reduction_stress}%")
```

### Scénario 3: Évolution d'une Nouvelle Espèce

```python
# Création d'une cellule expérimentale
cellule_experimentale = CelluleExperimentale(
    nom="ArbreHybride",
    caracteristiques={
        "resistance_thermique": 0.9,
        "efficacite_photosynthetique": 1.3,
        "adaptation_climatique": "variable"
    }
)

# Intégration dans l'écosystème
ecosysteme.ajouter_cellule(cellule_experimentale)

# Observation de l'évolution
for generation in range(10):
    # Cycle d'évolution
    evolution = agent.cycle_evolution()

    # Analyse des mutations bénéfiques
    mutations_utiles = evolution.filtrer_mutations_benefiques()

    # Application des améliorations
    agent.appliquer_mutations(mutations_utiles)

    print(f"Génération {generation}: {len(mutations_utiles)} améliorations")
```

## Dépannage

### Problèmes Courants

#### Erreur de Mémoire GPU

```python
# Solution: Réduire la quantification
config["agent"]["modeles_ia"]["codellama"]["quantization"] = "8bit"

# Ou utiliser le CPU pour Phi
config["agent"]["modeles_ia"]["phi"]["device"] = "cpu"
```

#### Analyses Trop Lentes

```python
# Optimisations de performance
config["performance"] = {
    "cache_analyses": True,
    "batch_size": 4,
    "parallel_processing": True
}
```

#### Cellules Non Réactives

```python
# Vérification de la configuration IA
agent.verifier_cerveaux_cellules()

# Redémarrage des cellules problématiques
for cellule in cellules_inactives:
    cellule.redemarrer_cerveau()
```

### Logs et Debugging

```python
# Activation des logs détaillés
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sauvegarde des états pour analyse
agent.sauvegarder_etat_debug("debug_state.json")
```

## Intégration Avancée

### API REST

```python
from flask import Flask
from kibali_api import KibaliAPI

app = Flask(__name__)
api = KibaliAPI(agent, ecosysteme)

@app.route('/status')
def get_status():
    return api.obtenir_statut()

@app.route('/cycle', methods=['POST'])
def run_cycle():
    return api.executer_cycle()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Interface Web

```python
# Lancement de l'interface web
from kibali_web import WebInterface

web = WebInterface(agent, ecosysteme)
web.demarrer(port=8080)
```

---

*Ce guide fournit tous les exemples nécessaires pour maîtriser l'utilisation de l'écosystème KIBALI.*