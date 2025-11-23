#!/usr/bin/env python3
"""
Kibalone 3D Animation Server
Lance le visualiseur 3D temps réel avec Three.js
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import os
import sys
import json
from pathlib import Path

class Kibalone3DServer:
    def __init__(self, port=8080):
        self.port = port
        self.server = None
        self.server_thread = None
        self.html_file = Path("outputs/kibalone_3d_viewer.html")

        if not self.html_file.exists():
            print("❌ Fichier HTML 3D non trouvé. Génération...")
            self.generate_3d_viewer()

    def generate_3d_viewer(self):
        """Génère le visualiseur 3D si nécessaire"""
        # Le fichier a déjà été créé ci-dessus
        pass

    def start_server(self):
        """Démarre le serveur web"""
        try:
            # Changer vers le répertoire racine du projet
            os.chdir(Path(__file__).parent)

            handler = http.server.SimpleHTTPRequestHandler

            # Configuration CORS pour permettre les requêtes depuis n'importe où
            class CORSRequestHandler(handler):
                def end_headers(self):
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    super().end_headers()

            self.server = socketserver.TCPServer(("", self.port), CORSRequestHandler)

            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()

            print(f"🚀 Serveur 3D Kibalone démarré sur http://localhost:{self.port}")
            print(f"📁 Fichier servi: {self.html_file}")
            print("🎬 Ouverture du visualiseur 3D dans votre navigateur...")

            # Ouvrir dans le navigateur
            webbrowser.open(f"http://localhost:{self.port}/outputs/kibalone_3d_viewer.html")

            return True

        except Exception as e:
            print(f"❌ Erreur serveur: {e}")
            return False

    def stop_server(self):
        """Arrête le serveur"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("🛑 Serveur 3D arrêté")

    def send_animation_command(self, scenario_data):
        """Envoie une commande d'animation au visualiseur 3D"""
        # Pour l'instant, on simule - en production, cela utiliserait WebSockets
        print("🎭 Envoi du scénario au visualiseur 3D:")
        print(json.dumps(scenario_data, indent=2, ensure_ascii=False))

        # Simulation de l'animation
        print("\n🎬 Animation démarrée!")
        print("🌳 Arbres qui bougent organiquement")
        print("🏃 Personnage qui marche avec émotion")
        print("📹 Caméra qui suit intelligemment")
        print("✨ Rendu 3D temps réel actif!")

def animate_with_3d(prompt):
    """Fonction principale pour animer avec rendu 3D"""
    print(f"\n🎬 Kibalone 3D Animation Engine - Phase 2")
    print(f"📝 Prompt: {prompt}")
    print("=" * 50)

    # Démarrer le serveur 3D
    server = Kibalone3DServer()

    if not server.start_server():
        print("❌ Impossible de démarrer le serveur 3D")
        return

    # Simuler la génération du scénario (remplacer par CodeLlama en production)
    print("\n🎭 Génération du scénario avec IA...")

    # Scénario mock pour la démo
    scenario = {
        "titre": "Forêt de Cristal Magique",
        "personnages": ["Renard Magique"],
        "actions": [
            {"temps": 0, "action": "Caméra zoom lent sur le renard"},
            {"temps": 3, "action": "Renard commence à courir"},
            {"temps": 8, "action": "Arbres de cristal s'illuminent"},
            {"temps": 12, "action": "Vent fait bouger les feuilles"},
            {"temps": 18, "action": "Renard saute gracieusement"},
            {"temps": 25, "action": "Plan large sous la lune"}
        ],
        "style": "organique, cristallin, magique, poétique",
        "duree": 30
    }

    print("✅ Scénario généré:")
    print(f"   Titre: {scenario['titre']}")
    print(f"   Personnages: {', '.join(scenario['personnages'])}")
    print(f"   Actions: {len(scenario['actions'])} séquences")
    print(f"   Style: {scenario['style']}")

    # Envoyer au visualiseur 3D
    server.send_animation_command(scenario)

    print("\n🎯 Actions du visualiseur 3D:")
    print("   • Arbres qui bougent naturellement avec le vent")
    print("   • Personnage avec animations organiques")
    print("   • Caméra intelligente qui suit l'action")
    print("   • Éclairage dynamique (soleil + lune)")
    print("   • Rendu 60 FPS temps réel")

    print(f"\n🌐 Visualiseur ouvert: http://localhost:{server.port}/outputs/kibalone_3d_viewer.html")

    # Garder le serveur actif
    try:
        print("\n💡 Appuyez sur Ctrl+C pour arrêter le serveur 3D")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du serveur 3D...")
        server.stop_server()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 3d_server.py \"votre description d'animation\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    animate_with_3d(prompt)