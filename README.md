# 🌱 Kibali - Écosystème de Nano-IA Vivantes

**Langage organique pour nano-IA autonomes avec cerveau LLM et base de connaissances RAG**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/lojol469-cmd/kibalone-langage)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-orange.svg)](https://www.python.org/)

## 🚀 Démarrage Rapide

```bash
# Installation
git clone https://github.com/lojol469-cmd/kibalone-langage.git
cd kibalone-langage
./install_kibali.sh
source ~/.bashrc

# 📥 Télécharger un modèle LLM (obligatoire)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/phi-1_5', cache_dir='models/phi-1_5'); AutoModelForCausalLM.from_pretrained('microsoft/phi-1_5', cache_dir='models/phi-1_5')"

# Premier programme
kibali run cells/arbre.kib

# Interface 3D
kibali launch .
# Ouvrir http://localhost:8080
```

## 📖 Documentation Complète

Voir [README_RAG_3D.md](README_RAG_3D.md) pour la documentation complète incluant :
- 🧠 Guide d'entraînement des nano-IA
- 💻 Tutoriel de codage des cellules
- 🔤 Référence complète du langage Kibali
- 🎮 Manuel de l'interface 3D
- 🔧 Guide de développement

## 🌟 Fonctionnalités Clés

- 🤖 **Cerveau LLM autonome** (Phi-1.5) pour décisions intelligentes
- 📚 **Base de connaissances RAG** avec FAISS pour recherche sémantique
- 🧬 **Évolution automatique** des cellules basée sur l'expérience
- 🎮 **Interface 3D immersive** pour exploration visuelle
- 🌐 **Écosystème vivant** de nano-IA communicantes

## 🏗️ Architecture

```
Cellules .kib → Runtime Kibali → Cerveau Phi-1.5 ↔ Base RAG FAISS
      ↓              ↓              ↓              ↓
   Organiques   Autonome      Intelligent    Vectorielle
```

## 📚 Exemple d'Utilisation

```kibali
// Créer une cellule intelligente
cellule ArbreIntelligent {
    couleur: "vert"
    age: 3
    memoire: "biologie_arbres"

    action pousser()
    action adapter_temperature()
    action surveiller_sante()

    evolution: auto  // Évolution autonome !
}
```

```bash
# Exécuter
kibali run cells/arbre.kib

# Observer l'évolution
tail -f logs/evolution.log
```

---

**🌱 Avec Kibali, créez des IA qui vivent, apprennent et évoluent comme des organismes biologiques !**

[📖 Documentation Complète](README_RAG_3D.md) • [🐛 Signaler un Bug](https://github.com/lojol469-cmd/kibalone-langage/issues) • [💡 Proposer une Feature](https://github.com/lojol469-cmd/kibalone-langage/issues)</content>
<parameter name="filePath">/home/belikan/kibali_project/README.md