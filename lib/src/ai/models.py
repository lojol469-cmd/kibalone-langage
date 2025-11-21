# 🧠 Models - Gestion des Modèles d'IA

"""Module de gestion des modèles d'intelligence artificielle

Gère le chargement, l'inférence et l'optimisation des modèles :
- Code Llama 7B (quantization 4-bit)
- Phi-1.5
- Optimisations mémoire et performance
- Interface unifiée pour l'inférence
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline
)
import logging
from datetime import datetime
import time

from ..shared.config import Config
from ..shared.logger import get_logger

@dataclass
class InferenceResult:
    """Résultat d'une inférence de modèle"""
    text: str
    tokens_generated: int
    inference_time: float
    model_name: str
    temperature: float
    max_tokens: int
    metadata: Optional[Dict[str, Any]] = None

    @property
    def tokens_per_second(self) -> float:
        """Calcule les tokens par seconde"""
        if self.inference_time > 0:
            return self.tokens_generated / self.inference_time
        return 0.0

class BaseModel:
    """Classe de base pour tous les modèles d'IA"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(f"IA.{self.__class__.__name__}")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> bool:
        """Charge le modèle en mémoire"""
        raise NotImplementedError("Méthode à implémenter dans les sous-classes")

    def unload_model(self) -> None:
        """Décharge le modèle de la mémoire"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.loaded = False
        self.logger.info("Modèle déchargé de la mémoire")

    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Génère du texte à partir d'un prompt"""
        raise NotImplementedError("Méthode à implémenter dans les sous-classes")

    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé"""
        return self.loaded

    def get_memory_usage(self) -> Dict[str, float]:
        """Retourne l'utilisation mémoire du modèle"""
        if not self.loaded:
            return {"total": 0.0, "cuda": 0.0, "cpu": 0.0}

        memory = {}

        if torch.cuda.is_available():
            memory["cuda"] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory["cuda_reserved"] = torch.cuda.memory_reserved() / 1024**3
        else:
            memory["cuda"] = 0.0
            memory["cuda_reserved"] = 0.0

        # Estimation de la mémoire CPU (approximative)
        memory["cpu"] = 0.0  # Difficile à mesurer précisément

        memory["total"] = memory["cuda"] + memory["cpu"]

        return memory

class CodeLlamaModel(BaseModel):
    """Gestionnaire du modèle Code Llama 7B"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.model_path = config.ia.codellama_model_path
        self.quantization = config.ia.codellama_quantization
        self.max_tokens = config.ia.codellama_max_tokens
        self.temperature = config.ia.codellama_temperature

    def load_model(self) -> bool:
        """Charge Code Llama avec quantization 4-bit"""
        try:
            self.logger.info(f"Chargement de Code Llama depuis {self.model_path}")

            # Configuration de quantization
            if self.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None

            # Chargement du tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Chargement du modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.quantization == "4bit" else torch.float32
            )

            # Création du pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            self.loaded = True
            self.logger.info("Code Llama chargé avec succès")

            # Log de l'utilisation mémoire
            memory = self.get_memory_usage()
            self.logger.info(".2f")

            return True

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de Code Llama: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Génère du code avec Code Llama"""
        if not self.loaded:
            raise RuntimeError("Modèle Code Llama non chargé")

        # Paramètres de génération
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        do_sample = kwargs.get('do_sample', True)

        # Formatage du prompt pour la génération de code
        formatted_prompt = f"```python\n{prompt}\n```"

        try:
            start_time = time.time()

            # Génération
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            inference_time = time.time() - start_time

            # Extraction du texte généré
            generated_text = outputs[0]['generated_text']
            # Suppression du prompt du résultat
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()

            # Comptage des tokens
            tokens_generated = len(self.tokenizer.encode(generated_text))

            result = InferenceResult(
                text=generated_text,
                tokens_generated=tokens_generated,
                inference_time=inference_time,
                model_name="CodeLlama-7B",
                temperature=temperature,
                max_tokens=max_tokens,
                metadata={
                    "quantization": self.quantization,
                    "device": self.device,
                    "tokens_per_second": tokens_generated / inference_time if inference_time > 0 else 0
                }
            )

            self.logger.debug(f"Génération Code Llama: {tokens_generated} tokens en {inference_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération Code Llama: {e}")
            raise

class PhiModel(BaseModel):
    """Gestionnaire du modèle Phi-1.5"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.model_path = config.ia.phi_model_path
        self.max_tokens = config.ia.phi_max_tokens
        self.temperature = config.ia.phi_temperature

    def load_model(self) -> bool:
        """Charge Phi-1.5"""
        try:
            self.logger.info(f"Chargement de Phi-1.5 depuis {self.model_path}")

            # Chargement du tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Chargement du modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Création du pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            self.loaded = True
            self.logger.info("Phi-1.5 chargé avec succès")

            # Log de l'utilisation mémoire
            memory = self.get_memory_usage()
            self.logger.info(".2f")

            return True

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de Phi-1.5: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Génère du texte avec Phi-1.5"""
        if not self.loaded:
            raise RuntimeError("Modèle Phi-1.5 non chargé")

        # Paramètres de génération
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        do_sample = kwargs.get('do_sample', True)

        try:
            start_time = time.time()

            # Génération
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1,
            )

            inference_time = time.time() - start_time

            # Extraction du texte généré
            generated_text = outputs[0]['generated_text']
            # Suppression du prompt du résultat
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            # Comptage des tokens
            tokens_generated = len(self.tokenizer.encode(generated_text))

            result = InferenceResult(
                text=generated_text,
                tokens_generated=tokens_generated,
                inference_time=inference_time,
                model_name="Phi-1.5",
                temperature=temperature,
                max_tokens=max_tokens,
                metadata={
                    "device": self.device,
                    "tokens_per_second": tokens_generated / inference_time if inference_time > 0 else 0
                }
            )

            self.logger.debug(f"Génération Phi-1.5: {tokens_generated} tokens en {inference_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération Phi-1.5: {e}")
            raise

class ModelManager:
    """Gestionnaire central des modèles d'IA"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger("IA.ModelManager")

        # Instances des modèles
        self.models: Dict[str, BaseModel] = {
            "codellama": CodeLlamaModel(self.config),
            "phi": PhiModel(self.config)
        }

        # État des modèles
        self.model_status: Dict[str, bool] = {name: False for name in self.models.keys()}

        # Cache des résultats récents
        self.result_cache: Dict[str, InferenceResult] = {}
        self.cache_max_size = 50

        self.logger.info("ModelManager initialisé")

    def load_model(self, model_name: str) -> bool:
        """Charge un modèle spécifique"""
        if model_name not in self.models:
            self.logger.error(f"Modèle inconnu: {model_name}")
            return False

        try:
            success = self.models[model_name].load_model()
            self.model_status[model_name] = success

            if success:
                self.logger.info(f"Modèle {model_name} chargé avec succès")
            else:
                self.logger.error(f"Échec du chargement du modèle {model_name}")

            return success

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {model_name}: {e}")
            self.model_status[model_name] = False
            return False

    def unload_model(self, model_name: str) -> None:
        """Décharge un modèle spécifique"""
        if model_name in self.models:
            self.models[model_name].unload_model()
            self.model_status[model_name] = False
            self.logger.info(f"Modèle {model_name} déchargé")

    def generate_code(self, prompt: str, model: str = "codellama", **kwargs) -> InferenceResult:
        """Génère du code en utilisant le modèle spécifié"""
        return self.generate(prompt, model, **kwargs)

    def generate_text(self, prompt: str, model: str = "phi", **kwargs) -> InferenceResult:
        """Génère du texte en utilisant le modèle spécifié"""
        return self.generate(prompt, model, **kwargs)

    def generate(self, prompt: str, model: str = "codellama", **kwargs) -> InferenceResult:
        """Génère du contenu en utilisant le modèle spécifié"""
        if model not in self.models:
            raise ValueError(f"Modèle inconnu: {model}")

        if not self.model_status.get(model, False):
            # Tentative de chargement automatique
            if not self.load_model(model):
                raise RuntimeError(f"Impossible de charger le modèle {model}")

        # Vérification du cache
        cache_key = f"{model}_{hash(prompt)}_{kwargs.get('temperature', 'default')}"
        if cache_key in self.result_cache:
            self.logger.debug(f"Résultat trouvé en cache pour {cache_key}")
            return self.result_cache[cache_key]

        try:
            result = self.models[model].generate(prompt, **kwargs)

            # Mise en cache
            if len(self.result_cache) >= self.cache_max_size:
                # Suppression d'une entrée aléatoire
                self.result_cache.pop(next(iter(self.result_cache)))
            self.result_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération avec {model}: {e}")
            raise

    def generate_embeddings(self, texts: List[str], model: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
        """Génère des embeddings pour une liste de textes

        Args:
            texts: Liste de textes à encoder
            model: Modèle d'embedding à utiliser

        Returns:
            List[List[float]]: Liste d'embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Chargement du modèle (avec cache)
            if not hasattr(self, '_embedding_model') or self._embedding_model_name != model:
                self.logger.info(f"Chargement du modèle d'embedding: {model}")
                self._embedding_model = SentenceTransformer(model)
                self._embedding_model_name = model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._embedding_model.to(device)

            # Génération des embeddings
            embeddings = self._embedding_model.encode(texts, convert_to_tensor=False)

            # Conversion en liste de listes
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            else:
                embeddings = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]

            return embeddings

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération d'embeddings: {e}")
            raise

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Retourne le statut de tous les modèles"""
        status = {}

        for name, model in self.models.items():
            status[name] = {
                "loaded": self.model_status.get(name, False),
                "memory_usage": model.get_memory_usage() if model.is_loaded() else {},
                "device": model.device if hasattr(model, 'device') else "unknown"
            }

        return status

    def optimize_memory(self) -> None:
        """Optimise l'utilisation mémoire"""
        self.logger.info("Optimisation de la mémoire...")

        # Déchargement des modèles non utilisés récemment
        # (Logique simplifiée - à étendre selon les besoins)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Cache CUDA vidé")

    def preload_models(self, models: Optional[List[str]] = None) -> Dict[str, bool]:
        """Précharge plusieurs modèles"""
        if models is None:
            models = list(self.models.keys())

        results = {}
        for model_name in models:
            results[model_name] = self.load_model(model_name)

        return results

    def get_available_models(self) -> List[str]:
        """Retourne la liste des modèles disponibles"""
        return list(self.models.keys())

    def is_model_available(self, model_name: str) -> bool:
        """Vérifie si un modèle est disponible"""
        return model_name in self.models

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retourne les informations sur un modèle"""
        if model_name not in self.models:
            return None

        model = self.models[model_name]
        return {
            "name": model_name,
            "class": model.__class__.__name__,
            "loaded": model.is_loaded(),
            "device": getattr(model, 'device', 'unknown'),
            "memory_usage": model.get_memory_usage(),
            "config": {
                "max_tokens": getattr(model, 'max_tokens', None),
                "temperature": getattr(model, 'temperature', None)
            }
        }