#!/usr/bin/env python3
"""
Script pour télécharger les PDFs de chaque cellule
Chaque cellule aura sa base vectorielle
"""

import requests
import os
from pathlib import Path

def download_pdf(url, filename):
    """Télécharge un PDF depuis une URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"✅ Téléchargé: {filename}")
        return True
    except Exception as e:
        print(f"❌ Erreur téléchargement {filename}: {e}")
        return False

def download_cell_pdfs():
    """Télécharge les PDFs pour chaque cellule"""

    pdf_dir = Path("./rag/data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # URLs des PDFs (sources publiques accessibles)
    pdfs = {
        "arbres_biology.pdf": "https://www.epa.gov/sites/default/files/2016-09/documents/climate-change-basic-info.pdf",  # Temporaire, remplacer par PDF arbres
        "climat_science.pdf": "https://www.epa.gov/sites/default/files/2016-09/documents/climate-change-basic-info.pdf",
        "ecureuil_behavior.pdf": "https://www.epa.gov/sites/default/files/2016-09/documents/climate-change-basic-info.pdf",  # Temporaire
        "fleur_biology.pdf": "https://www.epa.gov/sites/default/files/2016-09/documents/climate-change-basic-info.pdf"  # Temporaire
    }

    for filename, url in pdfs.items():
        filepath = pdf_dir / filename
        if not filepath.exists():
            print(f"📥 Téléchargement de {filename}...")
            download_pdf(url, filepath)
        else:
            print(f"📄 {filename} existe déjà")

if __name__ == "__main__":
    download_cell_pdfs()