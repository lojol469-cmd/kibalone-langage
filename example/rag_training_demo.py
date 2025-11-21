#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Exemple d'Entraînement RAG - Démonstration du système KIBALI

Ce script démontre l'utilisation du système d'entraînement RAG
pour indexer des documents biologiques et créer une base de connaissances.
"""

import sys
import os
from pathlib import Path

# Ajout du chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

from lib.src.ai.trainer import RAGTrainer, ConfigurationEntrainement
from lib.src.cells import charger_cellule
from lib.src.shared.logger import get_logger

def main():
    """Fonction principale de démonstration"""
    logger = get_logger("Demo.RAG")

    print("🌱 Démonstration du système d'entraînement RAG KIBALI")
    print("=" * 60)

    # Chargement de la cellule RAGTrainer
    print("\n📖 Chargement de la cellule RAGTrainer...")
    cellule = charger_cellule('train')

    if cellule:
        print(f"✅ Cellule chargée: {cellule['nom']}")
        print(f"   Propriétés: {cellule['proprietes']}")
        print(f"   Actions: {cellule['actions']}")
    else:
        print("❌ Erreur: Cellule RAGTrainer non trouvée")
        return

    # Configuration de l'entraînement
    print("\n⚙️ Configuration de l'entraînement...")
    config_entrainement = ConfigurationEntrainement(
        chunk_size=cellule['proprietes'].get('chunk_size', 512),
        chunk_overlap=cellule['proprietes'].get('chunk_overlap', 50),
        embedding_model=cellule['proprietes'].get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2').strip('"'),
        chemin_sortie_index=cellule['proprietes'].get('output_index', 'data/rag/indexes/arbres_biology.index').strip('"'),
        chemin_sortie_metadata=cellule['proprietes'].get('output_metadata', 'data/rag/metadata/arbres_biology.json').strip('"')
    )

    # Initialisation du trainer
    print("\n🤖 Initialisation du RAGTrainer...")
    trainer = RAGTrainer()

    # Vérification des documents
    chemin_docs = cellule['proprietes'].get('pdf_path', 'data/pdfs/arbres_biology.pdf').strip('"')
    chemin_docs = Path(chemin_docs).parent  # Dossier parent

    if not chemin_docs.exists():
        print(f"\n📁 Création du dossier documents: {chemin_docs}")
        chemin_docs.mkdir(parents=True, exist_ok=True)

        # Création d'un document d'exemple
        doc_exemple = chemin_docs / "arbres_biology.txt"
        contenu_exemple = """
        BIOLOGIE DES ARBRES ET LEUR MORPHOLOGIE

        Les arbres sont des organismes vivants fascinants qui jouent un rôle crucial
        dans l'écosystème terrestre. Leur morphologie complexe leur permet d'adapter
        à divers environnements et conditions climatiques.

        MORPHOLOGIE DES FEUILLES

        Les feuilles des arbres sont des organes spécialisés dans la photosynthèse.
        Elles présentent diverses formes et adaptations :

        1. Feuilles aciculaires (conifères) : Longues, étroites, persistantes
        2. Feuilles laminaires (feuillus) : Largés, plates, souvent caduques
        3. Feuilles composées : Divisées en plusieurs folioles

        ADAPTATIONS ENVIRONMENTALES

        Les arbres développent diverses stratégies d'adaptation :
        - Racines profondes pour l'accès à l'eau souterraine
        - Écorce épaisse pour la protection contre les incendies
        - Feuillage dense pour optimiser la captation solaire
        - Reproduction par graines pour la dispersion

        RÔLE ECOSYSTEMIQUE

        Les arbres contribuent à :
        - Production d'oxygène par photosynthèse
        - Stockage du carbone atmosphérique
        - Régulation du cycle de l'eau
        - Habitat pour la biodiversité
        - Stabilisation des sols
        """

        doc_exemple.write_text(contenu_exemple, encoding='utf-8')
        print(f"✅ Document d'exemple créé: {doc_exemple}")

    # Construction de l'index
    print("\n🔍 Construction de l'index RAG...")
    try:
        resultats = trainer.construire_index(
            chemin_documents=str(chemin_docs),
            domaine="arbres_biology",
            forcer_reconstruction=True
        )

        print("✅ Index construit avec succès !")
        print(f"   Documents traités: {resultats['documents_traitees']}")
        print(f"   Chunks créés: {resultats['chunks_crees']}")
        print(f"   Durée: {resultats['duree_secondes']:.2f}s")
        print(f"   Chemin index: {resultats['chemin_index']}")

    except Exception as e:
        print(f"❌ Erreur lors de la construction: {e}")
        return

    # Test de l'index
    print("\n🧪 Test de l'index...")
    try:
        resultats_test = trainer.tester_index("arbres_biology", "morphologie des feuilles")

        print("✅ Test réussi !")
        print(f"   Documents indexés: {resultats_test['documents_indexes']}")
        print(f"   Résultats trouvés: {resultats_test['resultats_trouves']}")
        print(f"   Score moyen: {resultats_test['score_moyen']:.2f}")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")

    # Statistiques finales
    print("\n📊 Statistiques finales...")
    stats = trainer.obtenir_statistiques()
    print(f"   Documents indexés: {stats['documents_indexes']}")
    print(f"   Chunks totaux: {stats['chunks_totaux']}")
    print(f"   Termes indexés: {stats['termes_indexes']}")
    print(f"   Embeddings stockés: {stats['embeddings_stockes']}")

    print("\n🎉 Démonstration terminée avec succès !")
    print("\nPour utiliser le système RAG dans votre code:")
    print("from lib.src.ai.trainer import RAGTrainer")
    print("trainer = RAGTrainer()")
    print("resultats = trainer.construire_index('data/documents', 'mon_domaine')")

if __name__ == "__main__":
    main()