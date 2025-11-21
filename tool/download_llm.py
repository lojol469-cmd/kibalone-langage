#!/usr/bin/env python3
"""
Téléchargeur de modèles LLM pour cellules autonomes
Permet de choisir entre différents modèles selon les besoins
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_phi_15():
    """Télécharge Phi-1.5 (1.3GB) - modèle actuel"""
    print("📥 Téléchargement de Microsoft Phi-1.5 (1.3GB)...")
    model_path = Path("./ia/phi-1_5")

    if model_path.exists():
        print("✅ Phi-1.5 déjà présent")
        return str(model_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("✅ Phi-1.5 téléchargé avec succès")
        return str(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de Phi-1.5: {e}")
        return None

def download_phi_2():
    """Télécharge Phi-2 (2.7GB) - modèle plus puissant pour génération de code"""
    print("📥 Téléchargement de Microsoft Phi-2 (2.7GB)...")
    model_path = Path("./ia/phi-2")

    if model_path.exists():
        print("✅ Phi-2 déjà présent")
        return str(model_path)

    try:
        snapshot_download(
            repo_id="microsoft/phi-2",
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        print("✅ Phi-2 téléchargé avec succès")
        return str(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de Phi-2: {e}")
        return None

def download_starcoder_1b():
    """Télécharge StarCoder-1B (environ 1.8GB) - modèle de code optimisé"""
    print("📥 Téléchargement de StarCoder-1B (1.8GB)...")
    model_path = Path("./ia/starcoder-1b")

    if model_path.exists():
        print("✅ StarCoder-1B déjà présent")
        return str(model_path)

    try:
        snapshot_download(
            repo_id="bigcode/starcoder",
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        print("✅ StarCoder-1B téléchargé avec succès")
        return str(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de StarCoder-1B: {e}")
        return None

def download_deepseek_coder_1b():
    """Télécharge DeepSeek-Coder-1.3B (environ 2.6GB)"""
    print("📥 Téléchargement de DeepSeek-Coder-1.3B (2.6GB)...")
    model_path = Path("./ia/deepseek-coder-1b")

    if model_path.exists():
        print("✅ DeepSeek-Coder-1.3B déjà présent")
        return str(model_path)

    try:
        snapshot_download(
            repo_id="deepseek-ai/deepseek-coder-1.3b-base",
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        print("✅ DeepSeek-Coder-1.3B téléchargé avec succès")
        return str(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de DeepSeek-Coder-1.3B: {e}")
        return None

def download_codellama_7b():
    """Télécharge CodeLlama-7B-Instruct (environ 14GB)"""
    print("📥 Téléchargement de CodeLlama-7B-Instruct (14GB)...")
    model_path = Path("./ia/codellama-7b")

    if model_path.exists():
        print("✅ CodeLlama-7B déjà présent")
        return str(model_path)

    try:
        snapshot_download(
            repo_id="codellama/CodeLlama-7b-Instruct-hf",
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        print("✅ CodeLlama-7B téléchargé avec succès")
        return str(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de CodeLlama-7B: {e}")
        return None

def test_model_loading(model_path, model_name):
    """Teste le chargement du modèle"""
    print(f"🧪 Test du chargement de {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Test simple de génération
        test_prompt = "def modify_parameter(value, factor):"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"✅ {model_name} chargé et testé avec succès")
        print(f"   Test génération: {response[len(test_prompt):].strip()[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test de {model_name}: {e}")
        return False

def main():
    print("🤖 Téléchargeur de modèles LLM pour cellules autonomes")
    print("=" * 60)

    models = {
        "1": ("Phi-1.5 (actuel)", download_phi_15, "~1.3GB"),
        "2": ("Phi-2 (recommandé)", download_phi_2, "~2.7GB"),
        "3": ("StarCoder-1B", download_starcoder_1b, "~1.8GB"),
        "4": ("DeepSeek-Coder-1.3B", download_deepseek_coder_1b, "~2.6GB"),
        "5": ("CodeLlama-7B (puissant)", download_codellama_7b, "~14GB")
    }

    print("Modèles disponibles pour modification autonome de paramètres:")
    for key, (name, func, size) in models.items():
        print(f"  {key}. {name} ({size}) - {getattr(func, '__doc__', '').split('-')[0].strip()}")

    choice = input("\nChoisissez un modèle (1-5) ou 'q' pour quitter: ").strip()

    if choice.lower() == 'q':
        return

    if choice not in models:
        print("❌ Choix invalide")
        return

    model_name, download_func, size = models[choice]
    print(f"\n📥 Téléchargement de {model_name} ({size})...")
    print("⚠️  Cela peut prendre du temps selon votre connexion internet")
    print("💡 Le modèle permettra aux cellules de modifier leurs paramètres en temps réel")

    model_path = download_func()

    if model_path and test_model_loading(model_path, model_name):
        print(f"\n✅ {model_name} prêt à être utilisé par les cellules autonomes!")
        print(f"   Chemin: {model_path}")
        print("\n🔧 Pour utiliser ce modèle, modifiez ecosystem_simulation.py:")
        print("   Dans AutonomousBrain.__init__(), changez le model_path")
        print("   pour pointer vers ce nouveau modèle.")
    else:
        print(f"\n❌ Échec du téléchargement/test de {model_name}")

if __name__ == "__main__":
    main()