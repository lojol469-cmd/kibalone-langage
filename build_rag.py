#!/usr/bin/env python3
"""
Script pour construire les bases vectorielles pour chaque cellule
Utilise les métadonnées JSON pour créer les index FAISS
"""

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_vector_base(metadata_file, index_file):
    """Construit la base vectorielle à partir des métadonnées"""

    # Charger les métadonnées
    with open(metadata_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    if not chunks:
        print(f"⚠️ Aucune donnée dans {metadata_file}")
        return

    # Modèle d'embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extraire les textes
    texts = [chunk['chunk'] for chunk in chunks]
    metadata = chunks

    # Créer les embeddings
    print(f"🔄 Création des embeddings pour {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Créer l'index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Index de similarité cosinus

    # Normaliser les embeddings pour la similarité cosinus
    faiss.normalize_L2(embeddings)

    # Ajouter à l'index
    index.add(embeddings)

    # Sauvegarder l'index
    faiss.write_index(index, index_file)

    # Sauvegarder les métadonnées
    metadata_file_out = index_file.replace('.faiss', '_metadata.json')
    with open(metadata_file_out, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Base vectorielle créée: {index_file}")

def build_all_bases():
    """Construit les bases pour toutes les cellules"""

    metadata_dir = Path("./rag/metadata")
    index_dir = Path("./rag/indexes")
    index_dir.mkdir(exist_ok=True)

    cell_rags = {
        "arbres_biology": "arbres_biology.json",
        "climat_science": "climat_science.json",
        "ecureuil_behavior": "ecureuil_behavior.json",
        "fleur_biology": "fleur_biology.json"
    }

    for rag_name, json_file in cell_rags.items():
        metadata_file = metadata_dir / json_file
        index_file = index_dir / f"{rag_name}.faiss"

        if metadata_file.exists():
            print(f"🏗️ Construction de la base pour {rag_name}...")
            build_vector_base(str(metadata_file), str(index_file))
        else:
            print(f"❌ Métadonnées manquantes: {metadata_file}")

if __name__ == "__main__":
    build_all_bases()