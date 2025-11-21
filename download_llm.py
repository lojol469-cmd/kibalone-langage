#!/usr/bin/env python3
"""
Script de téléchargement du modèle LLM pour Kibali
Modèle choisi: microsoft/phi-1_5 (1.3GB, rapide, bon pour le raisonnement)
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_phi_model():
    """Télécharge le modèle Phi-1.5"""
    model_name = "microsoft/phi-1_5"
    local_path = Path("./models/phi-1_5")

    print(f"🚀 Téléchargement du modèle: {model_name}")
    print(f"📁 Destination: {local_path.absolute()}")
    print("⏳ Cela peut prendre quelques minutes...")

    # Créer le dossier si nécessaire
    local_path.parent.mkdir(exist_ok=True)

    try:
        # Télécharger le tokenizer
        print("📥 Téléchargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Télécharger le modèle
        print("📥 Téléchargement du modèle...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Sauvegarder localement
        print("💾 Sauvegarde locale...")
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

        print("✅ Modèle téléchargé avec succès!")
        print(f"📊 Taille approximative: 1.3GB")
        print(f"🔧 Modèle prêt pour l'intégration dans le cerveau des cellules")

    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return False

    return True

if __name__ == "__main__":
    success = download_phi_model()
    if success:
        print("\n🎉 Prêt pour l'intégration dans kibali.py!")
    else:
        print("\n❌ Échec du téléchargement. Vérifiez votre connexion internet.")