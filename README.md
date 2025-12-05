# 🧬 Kibali Framework - Langage Organique Multi-Plateforme

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Kibali** est un framework révolutionnaire pour créer des applications organiques vivantes qui s'adaptent et évoluent comme des êtres vivants. Inspiré par Flutter, Kibali permet de compiler des cellules IA vers Android, iOS, Web et Desktop avec une seule commande.

## ✨ Fonctionnalités

- 🚀 **Comme Flutter** : `kibali run` détecte automatiquement la plateforme
- 📱 **Multi-plateforme** : Android, iOS, Web, Desktop
- 🧠 **IA Intégrée** : Runtime avec LLM (Phi-1.5) et RAG
- 🌱 **Évolution Organique** : Les cellules apprennent et s'adaptent
- 🎯 **Syntaxe Naturelle** : Langage déclaratif inspiré de la biologie
- 🔌 **Plugin Isol** : Gestion des dépendances IA sans conflits (Stable Diffusion, YOLO, etc.)

## 📦 Installation Rapide

### Linux/macOS
```bash
curl -fsSL https://raw.githubusercontent.com/lojol469-cmd/kibalone-langage/main/install.sh | bash
```

### Windows
Téléchargez et exécutez `install_windows.bat` depuis le repository.

### Manuellement
```bash
git clone https://github.com/lojol469-cmd/kibalone-langage.git
cd kibalone-langage
pip install -r requirements.txt
```

## 🚀 Utilisation

### Créer une cellule
```kibali
cellule Arbre {
    // Nano-IA pour la gestion des arbres
    couleur: "vert"
    age: 3
    temperature: 25

    action pousser()
    action adapter_temperature()
    action surveiller_sante()

    evolution: auto
}
```

### Compiler comme Flutter
```bash
# Détection automatique de plateforme
kibali run cells/arbre.kib

# Compilation explicite
kibali compile cells/arbre.kib android
kibali compile cells/arbre.kib ios
kibali compile cells/arbre.kib web
kibali compile cells/arbre.kib desktop
```

## 📱 Plateformes Supportées

| Plateforme | Framework | Commande de Build |
|------------|-----------|-------------------|
| **Android** | Kivy + Buildozer | `buildozer android debug` |
| **iOS** | Toga + Briefcase | `briefcase build ios` |
| **Web** | Transcrypt | `transcrypt -b -m -n main.py` |
| **Desktop** | PyInstaller | `pyinstaller app.spec` |

## 🧬 Architecture

### Cellules (.kib)
Les programmes Kibali sont des **cellules** qui définissent :
- **Champs** : État statique (propriétés)
- **Actions** : Comportements dynamiques
- **Imports** : Dépendances IA
- **Évolution** : Capacité d'apprentissage

### Runtime IA
Chaque application compilée inclut :
- **Cerveau LLM** : Phi-1.5 pour décisions intelligentes
- **Mémoire RAG** : Base de connaissances vectorielle
- **Évolution** : Apprentissage continu
- **Autonomie** : Décisions indépendantes

## 📚 Exemples

### Cellule Simple
```kibali
cellule TemperatureIA {
    seuil: 25
    unite: "celsius"

    action mesurer()
    action alerter()

    evolution: auto
}
```

### Cellule Avancée
```kibali
cellule ArbreIntelligent {
    importe IA:vision.feuilles
    importe IA:climat.temperature

    couleur: "vert"
    age: 5
    sante: 95

    memoire: "arbres_biology"

    action pousser()
    action photosynthese()
    action adapter_climat()
    action communiquer()

    evolution: auto
}
```

## 🛠️ Développement

### Prérequis
- Python 3.8+
- Git
- Dépendances système (build tools)

### Installation Développeur
```bash
git clone https://github.com/lojol469-cmd/kibalone-langage.git
cd kibalone-langage
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Tests
```bash
# Test de compilation
python kibali.py compile cells/arbre.kib android

# Test d'exécution
python kibali.py cells/arbre.kib

# Script de démonstration
./demo_kibali.sh
```

## 📖 Documentation

- [Guide de Démarrage](docs/getting_started.md)
- [Référence API](docs/api_reference.md)
- [Exemples](examples/)
- [Architecture](docs/architecture.md)

## 🤝 Contribution

1. Fork le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez (`git commit -m 'Add some AmazingFeature'`)
4. Pushez (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

## 🙏 Remerciements

- **Flutter** pour l'inspiration multi-plateforme
- **Microsoft** pour le modèle Phi-1.5
- **Hugging Face** pour les transformers
- **Meta** pour PyTorch

## 🌟 Communauté

- 📧 **Email** : team@kibali.org
- 💬 **Discord** : [Rejoignez-nous](https://discord.gg/kibali)
- 🐛 **Issues** : [GitHub Issues](https://github.com/lojol469-cmd/kibalone-langage/issues)

---

**Kibali** - Où le code devient vivant 🧬✨</content>
<parameter name="filePath">/home/belikan/kibali_project/README.md
=======
# kibalone-langage
langage de programmation des IA nano 
>>>>>>> f7505d9d0e9a7dc90d87701cca12b0df14b255b3
