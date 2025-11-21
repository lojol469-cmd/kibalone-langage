#!/bin/bash
# Script d'installation de la commande Kibali
# Usage: ./install_kibali.sh

echo "🌳 Installation de la commande Kibali..."

# Créer le répertoire bin local si nécessaire
mkdir -p ~/bin

# Créer le lien symbolique
ln -sf /home/belikan/kibali_project/kibali_cmd.py ~/bin/kibali

# Rendre exécutable
chmod +x ~/bin/kibali

# Ajouter au PATH si pas déjà présent
if ! grep -q "export PATH=\"\$HOME/bin:\$PATH\"" ~/.bashrc; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
    echo "✅ PATH mis à jour dans ~/.bashrc"
fi

echo "✅ Commande 'kibali' installée!"
echo ""
echo "Utilisation:"
echo "  kibali run <fichier.kib>     # Exécuter un programme Kibali"
echo "  kibali launch <dossier>      # Lancer un projet Kibali"
echo ""
echo "Exemple:"
echo "  kibali launch /home/belikan/kibali_project"
echo ""
echo "Rechargez votre terminal ou exécutez: source ~/.bashrc"