#!/bin/bash
# Démonstration du système Kibali comme Flutter

echo "🎯 Démonstration Kibali - Framework Multi-Plateforme comme Flutter"
echo "=================================================================="
echo ""

echo "📱 1. Exécution automatique (détecte la plateforme)"
echo "   kibali run cells/arbre.kib"
echo ""
kibali run cells/arbre.kib | head -20
echo "..."
echo ""

echo "🤖 2. Compilation explicite Android"
echo "   kibali compile cells/arbre.kib android"
echo ""
kibali compile cells/arbre.kib android
echo ""

echo "🌐 3. Compilation explicite Web"
echo "   kibali compile cells/climat.kib web"
echo ""
kibali compile cells/climat.kib web
echo ""

echo "💻 4. Compilation explicite Desktop"
echo "   kibali compile cells/climat.kib desktop"
echo ""
kibali compile cells/climat.kib desktop
echo ""

echo "✅ Démonstration terminée !"
echo ""
echo "📂 Fichiers générés dans build/:"
find build/ -name "*.py" -o -name "*.html" -o -name "*.spec" | head -10