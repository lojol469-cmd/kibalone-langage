#!/usr/bin/env python3
"""
Chargeur optimisé pour Code Llama 7B avec quantification 4-bit
Réduit l'usage mémoire de ~14GB à ~3.5GB
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
import json
import time

class OptimizedCodeLlamaLoader:
    """Chargeur optimisé pour Code Llama avec quantification"""

    def __init__(self, model_path="./ia/codellama-7b"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        """Charge le modèle avec quantification 4-bit"""
        print("🚀 Chargement de Code Llama 7B avec quantification 4-bit...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")

        # Configuration de quantification 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        try:
            # Charger le tokenizer
            print("📝 Chargement du tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Charger le modèle avec quantification
            print("🤖 Chargement du modèle (quantification 4-bit)...")
            start_time = time.time()

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",  # Utilise automatiquement GPU/CPU
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            load_time = time.time() - start_time
            self.is_loaded = True

            print(f"✅ Modèle chargé en {load_time:.1f} secondes")
            print(f"💾 Mémoire GPU utilisée: {self.get_gpu_memory_usage()}")

            return True

        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False

    def get_gpu_memory_usage(self):
        """Retourne l'usage mémoire GPU"""
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        return "N/A (pas de GPU)"

    def generate_code(self, prompt, max_length=512, temperature=0.7):
        """Génère du code à partir d'un prompt"""
        if not self.is_loaded:
            return "Erreur: modèle non chargé"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            return f"Erreur génération: {e}"

    def analyze_and_modify_parameters(self, cell_type, current_params, environmental_data):
        """Analyse et propose des modifications de paramètres pour une cellule"""

        prompt = f"""[INST] Analyse cette situation pour une cellule {cell_type} et retourne seulement du JSON.

Paramètres actuels: {current_params}
Environnement: {environmental_data}

Retourne exactement ce format JSON:
{{
    "internal_states": {{"photosynthèse_rate": 1.2, "résistance_stress": 0.8}},
    "physical_objects": {{"feuilles": {{"efficacité": 0.95}}}},
    "reasoning": "Explication de l'adaptation"
}}

Pas d'autre texte, juste le JSON. [/INST]"""

        response = self.generate_code(prompt, max_length=512, temperature=0.1)  # Très basse température pour plus de précision

        try:
            # Nettoyer la réponse
            response = response.strip()

            # Supprimer les balises si présentes
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

            # Chercher le JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                modifications = json.loads(json_str)
                return modifications

        except Exception as e:
            return {"error": f"Erreur parsing: {str(e)[:100]}", "raw_response": response[:300]}

        return {"error": "Aucun JSON trouvé", "raw_response": response[:300]}

def test_codellama_integration():
    """Test l'intégration de Code Llama dans le système cellulaire"""

    print("🧪 Test d'intégration Code Llama pour cellules autonomes\n")

    # Initialiser le chargeur
    loader = OptimizedCodeLlamaLoader()

    # Charger le modèle
    if not loader.load_model():
        print("❌ Échec du chargement du modèle")
        return

    # Test avec des données d'exemple pour un arbre
    test_params = {
        "photosynthèse_rate": 1.0,
        "absorption_eau": 0.8,
        "résistance_stress": 0.6
    }

    test_environment = {
        "temperature": 30.0,
        "light_level": 85.0,
        "soil_moisture": 35.0,
        "wind_speed": 8.0
    }

    print("🌳 Test d'adaptation pour un Arbre:")
    print(f"Paramètres actuels: {test_params}")
    print(f"Environnement: {test_environment}\n")

    # Générer des adaptations
    adaptations = loader.analyze_and_modify_parameters("Arbre", test_params, test_environment)

    print("🔄 Adaptations proposées:")
    print(json.dumps(adaptations, indent=2))

if __name__ == "__main__":
    test_codellama_integration()