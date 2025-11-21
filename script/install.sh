#!/bin/bash
# Installateur Kibali Framework - Comme Flutter
# Usage: curl -fsSL https://raw.githubusercontent.com/lojol469-cmd/kibalone-langage/main/install.sh | bash

set -e

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'affichage
print_step() {
    echo -e "${BLUE}[KIBALI]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Détection du système d'exploitation
detect_os() {
    case "$(uname -s)" in
        Linux*)     OS="linux";;
        Darwin*)    OS="macos";;
        CYGWIN*|MINGW*|MSYS*) OS="windows";;
        *)          OS="unknown";;
    esac
    echo "$OS"
}

# Vérification des prérequis
check_prerequisites() {
    print_step "Vérification des prérequis..."

    # Vérifier Python 3.8+
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 est requis. Installez Python 3.8+ depuis https://python.org"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python $PYTHON_VERSION détecté"
    else
        print_error "Python 3.8+ requis. Version actuelle: $PYTHON_VERSION"
        exit 1
    fi

    # Vérifier pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 est requis"
        exit 1
    fi
    print_success "pip3 détecté"

    # Vérifier git
    if ! command -v git &> /dev/null; then
        print_error "git est requis"
        exit 1
    fi
    print_success "git détecté"
}

# Installation des dépendances système
install_system_dependencies() {
    print_step "Installation des dépendances système..."

    OS=$(detect_os)

    case $OS in
        linux)
            if command -v apt-get &> /dev/null; then
                print_step "Installation via apt-get..."
                sudo apt-get update
                sudo apt-get install -y python3-dev build-essential git curl
            elif command -v yum &> /dev/null; then
                print_step "Installation via yum..."
                sudo yum install -y python3-devel gcc git curl
            elif command -v pacman &> /dev/null; then
                print_step "Installation via pacman..."
                sudo pacman -S python python-pip base-devel git curl
            else
                print_warning "Gestionnaire de paquets non détecté. Installez manuellement: python3-dev, build-essential, git"
            fi
            ;;
        macos)
            if command -v brew &> /dev/null; then
                print_step "Installation via Homebrew..."
                brew install python3 git curl
            else
                print_warning "Homebrew non détecté. Installez manuellement: python3, git, curl"
            fi
            ;;
        windows)
            print_warning "Sur Windows, assurez-vous que Python, git et curl sont installés"
            ;;
    esac
}

# Création de l'environnement virtuel
setup_venv() {
    print_step "Configuration de l'environnement virtuel..."

    # Créer répertoire d'installation
    KIBALI_HOME="$HOME/.kibali"
    mkdir -p "$KIBALI_HOME"

    # Créer environnement virtuel
    python3 -m venv "$KIBALI_HOME/venv"
    source "$KIBALI_HOME/venv/bin/activate"

    print_success "Environnement virtuel créé dans $KIBALI_HOME"
}

# Installation du framework Kibali
install_kibali() {
    print_step "Installation du framework Kibali..."

    source "$KIBALI_HOME/venv/bin/activate"

    # Cloner ou mettre à jour le repository
    if [ -d "$KIBALI_HOME/framework" ]; then
        print_step "Mise à jour du framework..."
        cd "$KIBALI_HOME/framework"
        git pull
    else
        print_step "Téléchargement du framework..."
        git clone https://github.com/lojol469-cmd/kibalone-langage.git "$KIBALI_HOME/framework"
        cd "$KIBALI_HOME/framework"
    fi

    # Installation des dépendances Python
    print_step "Installation des dépendances Python..."
    pip install --upgrade pip

    # Dépendances de base
    pip install sentence-transformers faiss-cpu transformers torch numpy kivy buildozer briefcase transcrypt pyinstaller

    # Dépendances optionnelles selon la plateforme
    if [ "$(detect_os)" = "linux" ]; then
        pip install kivy  # Pour Android
    fi

    print_success "Framework Kibali installé"
}

# Configuration des modèles IA
setup_models() {
    print_step "Configuration des modèles IA..."

    cd "$KIBALI_HOME/framework"

    # Créer répertoire models
    mkdir -p models

    # Télécharger Phi-1.5 si pas présent
    if [ ! -d "models/phi-1_5" ]; then
        print_step "Téléchargement du modèle Phi-1.5..."
        python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.makedirs('models/phi-1_5', exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5')
model = AutoModelForCausalLM.from_pretrained('microsoft/phi-1_5')
tokenizer.save_pretrained('models/phi-1_5')
model.save_pretrained('models/phi-1_5')
print('Modèle Phi-1.5 téléchargé')
        "
    fi

    print_success "Modèles IA configurés"
}

# Installation de la commande globale
install_command() {
    print_step "Installation de la commande 'kibali'..."

    # Créer le script de lancement
    cat > "$KIBALI_HOME/kibali" << 'EOF'
#!/bin/bash
# Lanceur Kibali Framework

KIBALI_HOME="$HOME/.kibali"

# Activer l'environnement virtuel
source "$KIBALI_HOME/venv/bin/activate"

# Ajouter le framework au PYTHONPATH
export PYTHONPATH="$KIBALI_HOME/framework:$PYTHONPATH"

# Lancer la commande
python3 "$KIBALI_HOME/framework/kibali_cmd.py" "$@"
EOF

    chmod +x "$KIBALI_HOME/kibali"

    # Ajouter au PATH
    SHELL_RC=""
    if [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.profile" ]; then
        SHELL_RC="$HOME/.profile"
    fi

    if [ -n "$SHELL_RC" ]; then
        if ! grep -q "export PATH=\"\$HOME/.kibali:\$PATH\"" "$SHELL_RC"; then
            echo "export PATH=\"\$HOME/.kibali:\$PATH\"" >> "$SHELL_RC"
            print_success "PATH configuré dans $SHELL_RC"
        fi
    else
        print_warning "Ajoutez manuellement au PATH: export PATH=\"\$HOME/.kibali:\$PATH\""
    fi

    # Lien symbolique dans /usr/local/bin si possible
    if [ -w "/usr/local/bin" ] || [ -w "/usr/local" ]; then
        sudo ln -sf "$KIBALI_HOME/kibali" "/usr/local/bin/kibali" 2>/dev/null || true
        print_success "Commande installée dans /usr/local/bin"
    fi

    print_success "Commande 'kibali' installée"
}

# Test de l'installation
test_installation() {
    print_step "Test de l'installation..."

    # Recharger le PATH
    export PATH="$HOME/.kibali:$PATH"

    # Tester la commande
    if command -v kibali &> /dev/null; then
        print_success "Commande 'kibali' disponible"

        # Tester l'aide
        if kibali --help &> /dev/null; then
            print_success "Framework Kibali opérationnel"
        else
            print_warning "La commande fonctionne mais l'aide a échoué"
        fi
    else
        print_error "Commande 'kibali' non trouvée dans le PATH"
        print_warning "Redémarrez votre terminal ou exécutez: source ~/.bashrc"
    fi
}

# Fonction principale
main() {
    echo ""
    echo "🧬 KIBALI FRAMEWORK INSTALLER"
    echo "============================="
    echo ""
    print_step "Bienvenue dans l'installateur Kibali!"
    print_step "Ce programme va installer le framework Kibali comme Flutter."
    echo ""

    # Étapes d'installation
    check_prerequisites
    install_system_dependencies
    setup_venv
    install_kibali
    setup_models
    install_command
    test_installation

    echo ""
    echo "🎉 INSTALLATION TERMINÉE !"
    echo ""
    print_success "Kibali Framework installé avec succès!"
    echo ""
    echo "📚 Pour commencer:"
    echo "   1. Redémarrez votre terminal"
    echo "   2. Créez votre première cellule: kibali --help"
    echo "   3. Exemple: kibali run cells/arbre.kib"
    echo ""
    echo "📁 Fichiers installés dans: $HOME/.kibali"
    echo "🌐 Repository: https://github.com/lojol469-cmd/kibalone-langage"
    echo ""
}

# Gestion des erreurs
trap 'print_error "Installation interrompue"' INT TERM

# Lancer l'installation
main "$@"