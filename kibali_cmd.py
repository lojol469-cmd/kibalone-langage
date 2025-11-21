#!/usr/bin/env python3
"""
Commande Kibali - Lanceur unifié pour les programmes Kibali
Usage: kibali run <fichier.kib>
       kibali launch <dossier_projet>
"""

import sys
import os
import subprocess
import argparse

def run_kibali_program(file_path):
    """Exécuter un programme .kib avec le runtime Kibali"""
    if not os.path.exists(file_path):
        print(f"❌ Fichier non trouvé: {file_path}")
        return False

    # Importer et utiliser le runtime Kibali
    try:
        # Ajouter le répertoire du fichier au path pour les imports relatifs
        file_dir = os.path.dirname(os.path.abspath(file_path))
        sys.path.insert(0, file_dir)

        from kibali import KibaliRuntime

        runtime = KibaliRuntime()
        result = runtime.run_program(file_path)
        print("Résultat de l'exécution Kibali:")
        print(result)
        return True

    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Assurez-vous que kibali.py est dans le même répertoire que le fichier .kib")
        return False
    except Exception as e:
        print(f"❌ Erreur d'exécution: {e}")
        return False

def launch_project(project_dir):
    """Lancer un projet Kibali (cherche launch.py ou serveur.kib)"""
    if not os.path.exists(project_dir):
        print(f"❌ Répertoire non trouvé: {project_dir}")
        return False

    # Chercher launch.py d'abord
    launch_file = os.path.join(project_dir, "launch.py")
    if os.path.exists(launch_file):
        print(f"🚀 Lancement du projet via {launch_file}")
        os.chdir(project_dir)
        subprocess.run([sys.executable, "launch.py"])
        return True

    # Chercher serveur.kib
    server_file = os.path.join(project_dir, "serveur.kib")
    if os.path.exists(server_file):
        print(f"🚀 Lancement du serveur via {server_file}")
        return run_kibali_program(server_file)

    print("❌ Aucun fichier de lancement trouvé (launch.py ou serveur.kib)")
    return False

def main():
    parser = argparse.ArgumentParser(description="Commande Kibali - Runtime pour programmes organiques")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Commande run
    run_parser = subparsers.add_parser("run", help="Exécuter un programme .kib")
    run_parser.add_argument("file", help="Chemin vers le fichier .kib")

    # Commande launch
    launch_parser = subparsers.add_parser("launch", help="Lancer un projet Kibali")
    launch_parser.add_argument("directory", help="Répertoire du projet Kibali")

    # Parser les arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == "run":
        success = run_kibali_program(args.file)
        sys.exit(0 if success else 1)

    elif args.command == "launch":
        success = launch_project(args.directory)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()