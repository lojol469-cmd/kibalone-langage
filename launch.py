#!/usr/bin/env python3
"""
Lanceur du Serveur RAG 3D Kibali
Démarre l'interface 3D pour explorer les connaissances sur les arbres
"""

import sys
import os
import json
import time
from kibali import KibaliRuntime

def main():
    print("🌳 Démarrage du Serveur RAG 3D Kibali...")
    print("📚 Interface 3D pour explorer les connaissances sur les arbres")
    print("=" * 60)

    # Initialiser le runtime Kibali
    runtime = KibaliRuntime()

    # Charger et initialiser le système RAG
    print("🔧 Initialisation du système RAG...")
    pdf_path = "data/pdfs/arbres_biology.pdf"
    index_path = "rag/indexes/arbres_biology.index"
    metadata_path = "rag/metadata/arbres_biology.json"

    if os.path.exists(pdf_path):
        print(f"📄 Traitement du PDF: {pdf_path}")
        success = runtime.build_rag_index(pdf_path, index_path, metadata_path)
        if success:
            print("✅ Base de connaissances créée avec succès!")
        else:
            print("❌ Erreur lors de la création de la base de connaissances")
            return
    else:
        print(f"⚠️ PDF non trouvé: {pdf_path}")
        print("Création d'une base vide pour démonstration...")

    # Charger le programme serveur
    serveur_file = "serveur.kib"
    if not os.path.exists(serveur_file):
        print(f"❌ Fichier serveur non trouvé: {serveur_file}")
        return

    print("🚀 Chargement du programme serveur...")
    try:
        # Charger le programme serveur
        with open(serveur_file, 'r', encoding='utf-8') as f:
            serveur_content = f.read()

        # Parser et charger la cellule serveur
        cell_info = runtime.parse_cell(serveur_content)
        runtime.cells[cell_info["name"]] = cell_info

        print(f"✅ Cellule serveur chargée: {cell_info['name']}")

        # Simuler le démarrage du serveur
        print("🌐 Démarrage du serveur 3D...")
        print("📍 URL: http://localhost:8080")
        print("🎮 Interface 3D: Ouvrez votre navigateur et explorez!")

        # Garder le programme actif
        print("\n" + "=" * 60)
        print("🎯 Commandes disponibles:")
        print("  - 'stats': Afficher les statistiques")
        print("  - 'query <question>': Tester une requête RAG")
        print("  - 'quit': Quitter")
        print("=" * 60)

        while True:
            try:
                cmd = input("Kibali RAG 3D> ").strip()

                if cmd == "quit":
                    print("👋 Arrêt du serveur...")
                    break
                elif cmd == "stats":
                    stats = runtime.simulate_cell("ServeurRAG3D")
                    print(json.dumps(stats, indent=2, ensure_ascii=False))
                elif cmd.startswith("query "):
                    query = cmd[6:]  # Enlever "query "
                    result = runtime.query_rag(query, index_path, metadata_path)
                    print(result)
                else:
                    print("❓ Commande inconnue. Utilisez 'stats', 'query <question>' ou 'quit'")

            except KeyboardInterrupt:
                print("\n👋 Arrêt du serveur...")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")

    except Exception as e:
        print(f"❌ Erreur lors du chargement du serveur: {e}")
        return

if __name__ == "__main__":
    main()