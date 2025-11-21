#!/usr/bin/env python3
"""
Commande Kibali - Lanceur unifié pour les programmes Kibali
Usage: kibali run <fichier.kib>          # Exécute ou compile selon la plateforme
       kibali compile <fichier.kib> <platform>  # Compilation explicite
       kibali launch <dossier_projet>    # Lance un projet
"""

import sys
import os
import subprocess
import argparse
import platform

def detect_platform():
    """Détecte automatiquement la plateforme cible comme Flutter"""
    system = platform.system().lower()

    # Vérifier si on est sur un device Android/iOS connecté
    try:
        # Vérifier adb pour Android
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'device' in result.stdout:
            return 'android'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Vérifier si on est sur macOS pour iOS
    if system == 'darwin':
        try:
            result = subprocess.run(['xcodebuild', '-version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return 'ios'
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Par défaut selon le système
    if system == 'linux':
        return 'desktop'  # ou 'android' si on veut privilégier mobile
    elif system == 'darwin':
        return 'desktop'
    elif system == 'windows':
        return 'desktop'
    else:
        return 'web'  # fallback

def run_or_compile_kibali(file_path):
    """Exécute ou compile un programme .kib selon la plateforme détectée (comme Flutter run)"""
    if not os.path.exists(file_path):
        print(f"❌ Fichier non trouvé: {file_path}")
        return False

    # Détecter la plateforme automatiquement
    target_platform = detect_platform()
    print(f"🎯 Plateforme détectée: {target_platform}")

    # Importer le runtime Kibali
    try:
        file_dir = os.path.dirname(os.path.abspath(file_path))
        sys.path.insert(0, file_dir)

        from kibali import KibaliRuntime

        runtime = KibaliRuntime()

        # Pour desktop, on peut exécuter directement
        if target_platform == 'desktop':
            print("💻 Exécution directe sur desktop...")
            result = runtime.run_program(file_path)
            print("Résultat de l'exécution Kibali:")
            print(result)
            return True
        else:
            # Pour mobile/web, on compile
            print(f"📱 Compilation automatique vers {target_platform}...")
            result = runtime.compile_to_platform(file_path, target_platform)

            if "error" in result:
                print(f"❌ Erreur compilation: {result['error']}")
                return False
            else:
                print("✅ Compilation terminée avec succès!")
                print(f"📁 Répertoire de sortie: {result['output_dir']}")
                print(f"🏗️ Commande de build: {result['build_command']}")

                # Proposer de lancer le build
                response = input(f"\n🚀 Voulez-vous lancer le build maintenant ? (y/N): ").lower().strip()
                if response in ['y', 'yes', 'oui']:
                    build_dir = result['output_dir']
                    build_cmd = result['build_command']

                    print(f"🏗️ Lancement du build dans {build_dir}...")
                    os.chdir(build_dir)

                    # Extraire et exécuter la commande
                    if '&&' in build_cmd:
                        cmd_parts = build_cmd.split('&&', 1)[1].strip()
                    else:
                        cmd_parts = build_cmd

                    print(f"Exécution: {cmd_parts}")
                    subprocess.run(cmd_parts, shell=True)

                return True

    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Assurez-vous que kibali.py est dans le même répertoire que le fichier .kib")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def compile_kibali_program(file_path, target_platform):
    """Compile explicitement vers une plateforme spécifique"""
    if not os.path.exists(file_path):
        print(f"❌ Fichier non trouvé: {file_path}")
        return False

    try:
        file_dir = os.path.dirname(os.path.abspath(file_path))
        sys.path.insert(0, file_dir)

        from kibali import KibaliRuntime

        runtime = KibaliRuntime()
        result = runtime.compile_to_platform(file_path, target_platform)

        if "error" in result:
            print(f"❌ Erreur compilation: {result['error']}")
            return False
        else:
            print("✅ Compilation terminée avec succès!")
            print(f"📁 Répertoire de sortie: {result['output_dir']}")
            print(f"🏗️ Commande de build: {result['build_command']}")
            return True

    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
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
    parser = argparse.ArgumentParser(description="Commande Kibali - Runtime pour programmes organiques (comme Flutter)")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Commande run (comme flutter run)
    run_parser = subparsers.add_parser("run", help="Exécuter ou compiler automatiquement selon la plateforme détectée")
    run_parser.add_argument("file", help="Chemin vers le fichier .kib")

    # Commande compile (compilation explicite)
    compile_parser = subparsers.add_parser("compile", help="Compiler vers une plateforme spécifique")
    compile_parser.add_argument("file", help="Chemin vers le fichier .kib")
    compile_parser.add_argument("platform", choices=["android", "ios", "web", "desktop"],
                               help="Plateforme cible")

    # Commande launch
    launch_parser = subparsers.add_parser("launch", help="Lancer un projet Kibali")
    launch_parser.add_argument("directory", help="Répertoire du projet Kibali")

    # Parser les arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == "run":
        # Comme Flutter run - détecte automatiquement et compile
        success = run_or_compile_kibali(args.file)
        sys.exit(0 if success else 1)

    elif args.command == "compile":
        # Compilation explicite
        success = compile_kibali_program(args.file, args.platform)
        sys.exit(0 if success else 1)

    elif args.command == "launch":
        success = launch_project(args.directory)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()