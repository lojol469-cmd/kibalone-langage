#!/usr/bin/env python3
"""
Script Python pour exécuter main.kib
Lance le runtime Kibali
"""

import subprocess
import sys
import os

def run_kibali():
    """Exécute le programme principal Kibali"""
    main_file = "main.kib"
    if not os.path.exists(main_file):
        print(f"Erreur: {main_file} non trouvé")
        sys.exit(1)

    # Lance le runtime Kibali
    try:
        result = subprocess.run([sys.executable, "kibali.py", main_file],
                              capture_output=True, text=True, encoding='utf-8')
        print("Résultat de l'exécution Kibali:")
        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")

if __name__ == "__main__":
    run_kibali()