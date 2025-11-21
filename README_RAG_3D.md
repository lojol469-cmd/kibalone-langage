# 🌱 Kibali - Écosystème de Nano-IA Vivantes

**Langage organique pour nano-IA autonomes avec cerveau LLM et base de connaissances RAG**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/lojol469-cmd/kibalone-langage)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-orange.svg)](https://www.python.org/)

## 📖 Table des Matières

- [🌟 Vue d'ensemble](#-vue-densemble)
- [🚀 Installation](#-installation)
- [🎯 Utilisation](#-utilisation)
- [🧠 Entraînement des Nano-IA](#-entraînement-des-nano-ia)
- [💻 Codage des Cellules](#-codage-des-cellules)
- [🔤 Langage Kibali](#-langage-kibali)
- [🎮 Interface 3D](#-interface-3d)
- [🔧 Architecture](#-architecture)
- [📚 API Reference](#-api-reference)
- [🤝 Contribution](#-contribution)

---

## 🌟 Vue d'ensemble

**Kibali** est un écosystème révolutionnaire pour créer des **nano-IA vivantes** qui évoluent de manière autonome. Contrairement aux IA traditionnelles, les programmes Kibali sont des **cellules organiques** qui :

- 🤖 **Pensent par elles-mêmes** grâce à un cerveau LLM (Phi-1.5)
- 📚 **Apprennent continuellement** via un système RAG intégré
- 🧬 **Évoluent automatiquement** basé sur leurs expériences
- 🌐 **Communiquent entre elles** dans un écosystème vivant
- 🎮 **S'expriment en 3D** pour une interaction immersive

### 🏗️ Architecture Unique

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cellules      │    │   Cerveau       │    │   Connaissances │
│   .kib          │◄──►│   Phi-1.5       │◄──►│   Base RAG      │
│                 │    │   Autonome      │    │   FAISS         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                         Évolution Continue
```

---

## 🚀 Installation

### Prérequis

```bash
# Python 3.8+
python --version

# Dépendances système
sudo apt-get update
sudo apt-get install python3-pip git

# Installation des dépendances Python
pip install torch transformers sentence-transformers faiss-cpu pymupdf
```

### Installation des Modèles LLM

**⚠️ Important :** Les modèles LLM ne sont pas inclus dans le repository GitHub en raison de leur taille. Vous devez les télécharger séparément.

#### Option 1 : Utiliser Phi-1.5 (Recommandé)

```bash
# Créer le dossier des modèles
mkdir -p models/phi-1_5

# Télécharger Phi-1.5 depuis HuggingFace
# Le système le fait automatiquement au premier lancement
# ou vous pouvez le pré-télécharger :
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/phi-1_5', cache_dir='models/phi-1_5'); AutoModelForCausalLM.from_pretrained('microsoft/phi-1_5', cache_dir='models/phi-1_5')"
```

#### Option 2 : Utiliser votre propre modèle

```bash
# Créer la structure de dossiers
mkdir -p models/votre_modele

# Placer vos fichiers de modèle
# models/votre_modele/
# ├── config.json
# ├── tokenizer.json
# ├── model.safetensors (ou pytorch_model.bin)
# └── ...

# Modifier la configuration si nécessaire
# Le système détecte automatiquement les modèles dans models/
```

#### Option 3 : Utiliser Mistral (Fallback)

```bash
# Le système utilise automatiquement Mistral si disponible
# Placer dans ia/mistral-7b/ ou models/mistral-7b/
```

### Vérification de l'Installation

```bash
# Tester le chargement du cerveau
python -c "from kibali import KibaliRuntime; r = KibaliRuntime(); r.load_brain(); print('✅ Cerveau chargé')"

# Tester une cellule
kibali run cells/arbre.kib
```

---

## 🎯 Utilisation

### Commandes de Base

```bash
# Afficher l'aide
kibali --help

# Exécuter un programme .kib
kibali run cells/arbre.kib

# Lancer un projet complet
kibali launch .

# Entraîner le système RAG
kibali run train.kib
```

### Premier Programme

```bash
# Créer votre première cellule
cat > hello.kib << 'EOF'
cellule HelloWorld {
    // Votre première nano-IA
    message: "Bonjour le monde !"

    action saluer()
    action evoluer()
}
EOF

# L'exécuter
kibali run hello.kib
```

### Interface 3D Immersive

```bash
# Lancer le serveur 3D
kibali launch .

# Ouvrir http://localhost:8080 dans votre navigateur
# Explorer les connaissances en 3D !
```

---

## 🧠 Entraînement des Nano-IA

### 1. Préparation des Données

```bash
# Créer le dossier des données
mkdir -p data/pdfs

# Placer vos documents PDF
cp votre_document.pdf data/pdfs/

# Créer le programme d'entraînement
cat > train.kib << 'EOF'
cellule RAGTrainer {
    // Entraînement du système RAG
    pdf_path: "data/pdfs/votre_document.pdf"
    output_index: "rag/indexes/document.index"
    output_metadata: "rag/metadata/document.json"

    action construire_index()
    action tester_index()
}
EOF
```

### 2. Configuration RAG

```json
// rag/config.json
{
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "similarity_threshold": 0.7,
    "max_results": 5
}
```

### 3. Lancement de l'Entraînement

```bash
# Entraîner le modèle
kibali run train.kib

# Vérifier les résultats
ls -la rag/indexes/
ls -la rag/metadata/
```

### 4. Test des Connaissances

```bash
# Lancer l'interface interactive
kibali launch .

# Dans l'interface, tester des requêtes :
# "Quelles sont les principales caractéristiques ?"
# "Comment fonctionne le système ?"
# "Quels sont les avantages ?"
```

### 5. Évolution Continue

Les nano-IA apprennent automatiquement :

```bash
# Observer l'évolution
tail -f logs/evolution.log

# Voir les décisions autonomes
kibali run cells/arbre.kib | jq '.decisions_autonomes'
```

---

## 💻 Codage des Cellules

### Structure d'une Cellule

```kibali
cellule NomCellule {
    // Commentaires avec //

    // Propriétés statiques
    propriete: "valeur"
    nombre: 42
    actif: true

    // Mémoire persistante
    memoire: "nom_memoire"

    // Actions définies
    action nom_action()
    action autre_action(parametre)

    // Évolution (optionnel)
    evolution: auto
}
```

### Exemple Complet : Cellule Arbre

```kibali
cellule Arbre {
    // Nano-IA pour la gestion des arbres

    // Propriétés biologiques
    couleur: "vert"
    age: 3
    hauteur: 2.5
    temperature: 25

    // État dynamique
    mouvement: "croissance"
    sante: "excellente"

    // Mémoire spécialisée
    memoire: "biologie_arbres"

    // Actions comportementales
    action pousser()
    action adapter_temperature()
    action surveiller_sante()
    action photosynthese()

    // Évolution autonome
    evolution: auto
}
```

### Types de Propriétés

```kibali
cellule ExempleTypes {
    // Types de base
    texte: "chaîne de caractères"
    nombre: 42
    decimal: 3.14
    booleen: true

    // Tableaux
    liste_nombres: [1, 2, 3, 4, 5]
    liste_textes: ["a", "b", "c"]

    // Objets complexes
    configuration: {
        actif: true,
        seuil: 0.8,
        parametres: [1, 2, 3]
    }

    // Références
    memoire: "nom_memoire"
    parent: "cellule_parent"
}
```

### Actions et Comportements

```kibali
cellule IAComplexe {
    // Actions simples
    action demarrer()
    action arreter()

    // Actions avec paramètres
    action calculer(valeur, coefficient)
    action communiquer(message, destinataire)

    // Actions conditionnelles
    action adapter_environnement() {
        si temperature > 30 {
            action: "refroidir"
        } sinon si temperature < 10 {
            action: "chauffer"
        } sinon {
            action: "maintenir"
        }
    }

    // Actions évolutives
    action apprendre(experience) {
        memoire.ajouter(experience)
        cerveau.analyser(experience)
    }
}
```

---

## 🔤 Langage Kibali

### Syntaxe de Base

```ebnf
programme ::= cellule*

cellule ::= "cellule" identifiant "{" propriete* action* "}"

propriete ::= identifiant ":" valeur

action ::= "action" identifiant "(" parametres? ")" corps?

valeur ::= chaine | nombre | booleen | tableau | objet

parametres ::= identifiant ("," identifiant)*
```

### Mots-clés Réservés

- `cellule` : Définit une nouvelle cellule
- `action` : Définit un comportement
- `memoire` : Référence une mémoire persistante
- `evolution` : Active l'évolution autonome
- `importe` : Importe des dépendances
- `si` : Condition if
- `sinon` : Condition else
- `pour` : Boucle for
- `dans` : Opérateur in

### Types de Données

| Type | Exemple | Description |
|------|---------|-------------|
| `chaine` | `"hello"` | Texte |
| `nombre` | `42` | Entier |
| `decimal` | `3.14` | Flottant |
| `booleen` | `true` | Booléen |
| `tableau` | `[1, 2, 3]` | Liste |
| `objet` | `{cle: valeur}` | Dictionnaire |

### Opérateurs

```kibali
// Arithmétiques
+ - * / %

// Comparaisons
== != < > <= >=

// Logiques
et ou non

// Assignation
=
```

### Exemple Avancé

```kibali
cellule IAIntelligente {
    // Propriétés complexes
    nom: "Alice"
    age: 1
    competences: ["apprentissage", "communication", "adaptation"]
    reseau_social: {
        amis: ["Bob", "Charlie"],
        influence: 0.85
    }

    // Mémoire contextuelle
    memoire: "experiences_alice"

    // Actions comportementales
    action apprendre(sujet) {
        connaissances = cerveau.rechercher(sujet)
        memoire.stocker(connaissances)
        competence = competence + 0.1
    }

    action communiquer(message, destinataire) {
        si reseau_social.amis.contient(destinataire) {
            envoyer(message, destinataire)
            influence = influence + 0.05
        }
    }

    action adapter(difficulte) {
        si difficulte > seuil_adaptation {
            evolution.activer()
            strategie = cerveau.proposer_strategie()
        }
    }

    // Évolution basée sur l'expérience
    evolution: auto
}
```

---

## 🎮 Interface 3D

### Lancement

```bash
# Démarrer le serveur 3D
kibali launch .

# Accéder à l'interface
# http://localhost:8080
```

### Navigation

- **🖱️ Clic + Glisser** : Tourner autour de la forêt
- **🔍 Molette** : Zoom avant/arrière
- **⌨️ Flèches** : Déplacement latéral
- **🎯 Clic sur arbre** : Focus sur une connaissance

### Recherche Interactive

```javascript
// Dans la console du navigateur
ws.send(JSON.stringify({
    type: 'query_rag',
    query: 'Comment poussent les arbres ?'
}));
```

### Personnalisation 3D

```kibali
cellule ServeurRAG3D {
    // Configuration 3D
    theme: "forest"
    arbres_par_chunk: 1
    couleur_base: "#228B22"
    eclairage: "natural"

    action generer_interface_3d() {
        // Code HTML/Three.js personnalisé
        return html_template
    }
}
```

---

## 🔧 Architecture

### Composants Principaux

```
kibali_project/
├── kibali.py              # Runtime principal
├── kibali_cmd.py          # Interface commande
├── launch.py              # Lanceur projets
├── cells/                 # Cellules .kib
├── memories/              # Mémoires persistantes
├── models/                # Modèles LLM
│   └── phi-1_5/          # Phi-1.5 local
├── rag/                   # Système RAG
│   ├── indexes/          # Index FAISS
│   ├── embeddings/       # Vectors
│   ├── metadata/         # Métadonnées
│   └── config.json       # Configuration
├── data/pdfs/            # Documents source
├── logs/                  # Logs système
└── ia/                    # Outils IA
```

### Flux de Données

1. **Chargement** : Cellules .kib → Runtime Kibali
2. **Exécution** : Runtime → Cerveau Phi-1.5
3. **Connaissances** : Cerveau ↔ Base RAG FAISS
4. **Évolution** : Expériences → Mémoires → Adaptation
5. **Interface** : Runtime → WebSocket → Three.js

### Sécurité

- **Sandboxing** : Exécution isolée des cellules
- **Validation** : Syntaxe et types vérifiés
- **Limites** : Ressources contrôlées
- **Audit** : Logs complets des actions

---

## 📚 API Reference

### Runtime Kibali

```python
from kibali import KibaliRuntime

runtime = KibaliRuntime()

# Charger et exécuter une cellule
result = runtime.run_program("cells/arbre.kib")

# Interroger le cerveau
response = runtime.query_brain_with_knowledge("Comment poussent les arbres ?")

# Accéder au RAG
results = runtime.query_rag("structure des arbres")
```

### Classes Principales

#### KibaliRuntime
- `load_brain()` : Charge le modèle Phi-1.5
- `run_program(file)` : Exécute un programme .kib
- `simulate_cell(name)` : Simule une cellule
- `query_brain_with_knowledge(query)` : Interroge le cerveau enrichi

#### KibaliRAGSystem
- `build_index(embeddings)` : Construit l'index FAISS
- `search(query)` : Recherche sémantique
- `chunk_text(text)` : Découpe en chunks
- `encode_chunks(chunks)` : Génère embeddings

### WebSocket API

```javascript
// Connexion
const ws = new WebSocket('ws://localhost:8080/ws');

// Envoi de requête
ws.send(JSON.stringify({
    type: 'query_rag',
    query: 'Comment fonctionnent les arbres ?'
}));

// Réception des résultats
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'rag_results') {
        afficherResultats(data.results);
    }
};
```

---

## 🤝 Contribution

### Développement

```bash
# Fork le repository
git clone https://github.com/lojol469-cmd/kibalone-langage.git
cd kibalone-langage

# Créer une branche
git checkout -b feature/nouvelle-fonctionnalite

# Installer en mode développement
pip install -e .

# Tests
python -m pytest tests/

# Commit et push
git add .
git commit -m "Ajout de nouvelle fonctionnalité"
git push origin feature/nouvelle-fonctionnalite
```

### Création de Cellules

1. **Concevoir** : Définir le comportement souhaité
2. **Coder** : Écrire la cellule en syntaxe Kibali
3. **Tester** : Exécuter avec `kibali run`
4. **Évoluer** : Observer l'apprentissage autonome

### Amélioration du Cerveau

```python
# Extension des capacités du cerveau
class ExtendedKibaliRuntime(KibaliRuntime):
    def query_brain_with_knowledge(self, query, context="", cell_state=None):
        # Logique personnalisée
        enhanced_context = self.enhance_context(query, context)
        return super().query_brain_with_knowledge(query, enhanced_context, cell_state)
```

### Tests

```bash
# Tests unitaires
python -m pytest tests/test_cells.py

# Tests d'intégration
python -m pytest tests/test_rag.py

# Tests du cerveau
python -m pytest tests/test_brain.py
```

---

## 📄 Licence

MIT License - voir [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- **Lojol469** - *Développement principal* - [lojol469@gmail.com](mailto:lojol469@gmail.com)
- **Communauté Kibali** - *Contributions et retours*

## 🙏 Remerciements

- **Microsoft** pour Phi-1.5
- **Meta** pour les modèles de base
- **Facebook AI** pour FAISS
- **Three.js** pour le rendu 3D

---

## 🚀 Roadmap

### v1.1.0
- [ ] Interface web complète
- [ ] Multi-modèle (images, audio)
- [ ] Communication inter-cellules
- [ ] Évolution génétique

### v1.2.0
- [ ] API REST complète
- [ ] Plugins extensibles
- [ ] Interface mobile
- [ ] Collaboration temps réel

### v2.0.0
- [ ] Système multi-agents
- [ ] Apprentissage par renforcement
- [ ] Interfaces neuronales
- [ ] Écosystème distribué

---

**🌱 Avec Kibali, créez des IA qui vivent, apprennent et évoluent comme des organismes biologiques !**