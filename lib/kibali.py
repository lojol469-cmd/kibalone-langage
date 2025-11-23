#!/usr/bin/env python3
"""
Kibali Runtime - Framework Multi-Plateforme pour Nano-IA Vivantes
Supporte Android, iOS, Web, Desktop avec compilation croisée
"""

import json
import os
import sys
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import platform
import subprocess
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# Import des outils
from lib.src.core.tools import Tools

# Imports pour compilation multi-plateforme
try:
    import kivy
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False

try:
    import buildozer
    BUILDOZER_AVAILABLE = True
except ImportError:
    BUILDOZER_AVAILABLE = False

try:
    import briefcase
    BRIEFCASE_AVAILABLE = True
except ImportError:
    BRIEFCASE_AVAILABLE = False

try:
    import transcrypt
    TRANSCRYPT_AVAILABLE = True
except ImportError:
    TRANSCRYPT_AVAILABLE = False

try:
    import PyInstaller
    PYINSTALLER_AVAILABLE = True
except ImportError:
    PYINSTALLER_AVAILABLE = False

class KibaliRAGSystem:
    """Système RAG intégré pour les cellules Kibali"""

    def __init__(self, config_path="rag/config.json"):
        self.config = self.load_config(config_path)
        self.encoder = None
        self.index = None
        self.metadata = []

    def load_config(self, config_path):
        """Charger la configuration RAG"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "similarity_threshold": 0.7,
            "max_results": 5
        }

    def initialize_encoder(self):
        """Initialiser le modèle d'embedding"""
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.config["embedding_model"])
        return self.encoder

    def chunk_text(self, text, chunk_size=None, overlap=None):
        """Découper le texte en chunks"""
        if chunk_size is None:
            chunk_size = self.config["chunk_size"]
        if overlap is None:
            overlap = self.config["chunk_overlap"]

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(" ".join(chunk))

        return chunks

    def encode_chunks(self, chunks):
        """Encoder les chunks en embeddings"""
        if self.encoder is None:
            self.initialize_encoder()
        return self.encoder.encode(chunks)

    def build_index(self, embeddings):
        """Construire l'index FAISS"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        return self.index

    def search(self, query, top_k=None):
        """Rechercher dans l'index"""
        if top_k is None:
            top_k = self.config["max_results"]

        if self.encoder is None or self.index is None:
            return []

        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "chunk": self.metadata[idx]["chunk"],
                    "document": self.metadata[idx]["document"],
                    "distance": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                })

        return results

    def save_index(self, index_path):
        """Sauvegarder l'index"""
        if self.index is not None:
            faiss.write_index(self.index, index_path)

    def load_index(self, index_path):
        """Charger l'index"""
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            return self.index
        return None

    def save_metadata(self, metadata_path):
        """Sauvegarder les métadonnées"""
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load_metadata(self, metadata_path):
        """Charger les métadonnées"""
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            return self.metadata
        return []

class KibaliRuntime:
    def __init__(self):
        self.cells = {}
        self.ia_dependencies = {}
        self.memories = {}
        self.tokenizer_phi = None
        self.model_phi = None
        self.tokenizer_codellama = None
        self.model_codellama = None
        self.current_brain = "phi-1_5"  # cerveau par défaut
        self.tools = Tools(self)
        self.rag_system = KibaliRAGSystem()

        # Chargement automatique des composants critiques
        self.initialize_brain_and_knowledge()

    def initialize_brain_and_knowledge(self):
        """Initialise automatiquement le cerveau et les connaissances"""
        print("🧠 Initialisation du système Kibali...")

        # Charger le cerveau (LLM)
        self.load_brain()

        # Charger les index RAG existants
        self.load_existing_indexes()

        print("✅ Système Kibali initialisé avec cerveau et connaissances")

    def load_existing_indexes(self):
        """Charge automatiquement tous les index RAG disponibles"""
        indexes_dir = "rag/indexes"
        metadata_dir = "rag/metadata"

        if os.path.exists(indexes_dir):
            for file in os.listdir(indexes_dir):
                if file.endswith('.index'):
                    index_name = file.replace('.index', '')
                    index_path = os.path.join(indexes_dir, file)
                    metadata_path = os.path.join(metadata_dir, f"{index_name}.json")

                    try:
                        # Charger l'index dans le système RAG
                        self.rag_system.load_index(index_path)
                        self.rag_system.load_metadata(metadata_path)
                        print(f"📚 Index RAG chargé: {index_name}")
                    except Exception as e:
                        print(f"⚠️ Erreur chargement index {index_name}: {e}")

    def query_brain_with_knowledge(self, query, context="", cell_state=None):
        """Interroge le cerveau enrichi avec les connaissances RAG"""
        tokenizer, model = self.get_current_tokenizer_and_model()
        if model is None:
            return "Cerveau non disponible"

        # Enrichir la requête avec des connaissances pertinentes
        knowledge_context = self.get_relevant_knowledge(query, cell_state)

        # Construire le contexte complet
        full_context = f"""
Contexte de la cellule: {context}
Connaissances pertinentes: {knowledge_context}
État actuel de la cellule: {json.dumps(cell_state, ensure_ascii=False) if cell_state else 'N/A'}
"""

        # Construire le prompt avec les outils et connaissances
        tool_descriptions = "\n".join([f"- {name}: {desc[:100]}..." for name, desc in self.tools.items()])
        prompt = f"""Tu es un cerveau de nano-IA dans le système Kibali.
Tu as accès à une base de connaissances extensive et peux influencer le comportement des cellules de manière autonome.

Outils disponibles:
{tool_descriptions}

{full_context}

Question/Action: {query}

Réponds de manière intelligente en tenant compte des connaissances disponibles et guide le comportement de la cellule:"""

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=200, num_return_sequences=1, temperature=0.8, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Nettoyer la réponse
        response = response.replace(prompt, "").strip()
        return response

    def get_relevant_knowledge(self, query, cell_state=None):
        """Extrait les connaissances pertinentes de la base RAG"""
        # Vérifier si la cellule a une connaissance RAG spécialisée
        if cell_state and "connaissance_rag" in cell_state.get("champs", {}):
            rag_config = cell_state["champs"]["connaissance_rag"].strip('"')
            # Format attendu: "type:nom" ou juste "nom"
            if ':' in rag_config:
                parts = rag_config.split(':', 1)
                rag_type, rag_name = parts[0], parts[1]
            else:
                rag_type, rag_name = "document", rag_config

            # Construire le chemin vers l'index spécialisé
            index_path = f"cells/rag_knowledge/{rag_name}.index"
            metadata_path = f"cells/rag_knowledge/{rag_name}.json"

            # Essayer de charger l'index spécialisé
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                try:
                    # Sauvegarder l'index actuel
                    current_index = self.rag_system.index
                    current_metadata = self.rag_system.metadata

                    # Charger l'index spécialisé
                    self.rag_system.load_index(index_path)
                    self.rag_system.load_metadata(metadata_path)

                    # Rechercher
                    results = self.rag_system.search(query, top_k=2)
                    knowledge = []
                    if results:
                        for result in results:
                            knowledge.append(f"• {result['chunk'][:150]}...")

                    # Restaurer l'index général
                    self.rag_system.index = current_index
                    self.rag_system.metadata = current_metadata

                    if knowledge:
                        return f"Connaissances spécialisées ({rag_name}):\n" + "\n".join(knowledge)

                except Exception as e:
                    print(f"⚠️ Erreur chargement RAG spécialisé {rag_name}: {e}")

        # Fallback vers la base RAG générale
        if self.rag_system.index is None:
            return "Aucune connaissance disponible"

        try:
            results = self.rag_system.search(query, top_k=2)
            if not results:
                return "Aucune connaissance pertinente trouvée"

            knowledge = []
            for result in results:
                knowledge.append(f"• {result['chunk'][:150]}...")

            return "\n".join(knowledge)
        except Exception as e:
            return f"Erreur extraction connaissances: {e}"

    def evolve_cell_behavior(self, cell_name, experience_data):
        """Fait évoluer le comportement d'une cellule basé sur l'expérience"""
        if cell_name not in self.cells:
            return False

        cell = self.cells[cell_name]

        # Utiliser le cerveau pour analyser l'expérience et suggérer des évolutions
        evolution_query = f"""
Analyse cette expérience de la cellule {cell_name} et suggère comment elle devrait évoluer:

Expérience: {json.dumps(experience_data, ensure_ascii=False)}

État actuel: {json.dumps(cell, ensure_ascii=False)}

Suggère des modifications de comportement ou de paramètres:
"""

        evolution_suggestion = self.query_brain_with_knowledge(evolution_query, cell_state=cell)

        # Pour l'instant, stocker la suggestion dans la mémoire
        memory_key = f"evolution_{cell_name}"
        memory_data = {
            "timestamp": "2025-11-21",
            "experience": experience_data,
            "suggestion": evolution_suggestion,
            "applied": False
        }

        # Sauvegarder dans la mémoire
        os.makedirs("memories", exist_ok=True)
        with open(f"memories/{memory_key}.json", 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)

        print(f"🧬 Évolution suggérée pour {cell_name}: {evolution_suggestion[:100]}...")
        return True

    def load_tools(self):
        """Charge les outils nano IA depuis le dossier ia/"""
        tools = {}
        ia_dir = "ia"
        if os.path.exists(ia_dir):
            for file in os.listdir(ia_dir):
                if file.endswith('.llm'):
                    tool_name = file.replace('.llm', '')
                    tool_path = os.path.join(ia_dir, file)
                    with open(tool_path, 'r', encoding='utf-8') as f:
                        tools[tool_name] = f.read()
        return tools

    def load_brain(self):
        """Charge les modèles LLM pour les cerveaux (Phi-1.5 et CodeLlama-7B)"""
        if self.model_phi is None or self.model_codellama is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                # Charger Phi-1.5 depuis le dossier local assets/ia/phi-1_5/
                phi_local_path = os.path.join(os.path.dirname(__file__), "..", "assets", "ia", "phi-1_5")
                if os.path.exists(phi_local_path):
                    try:
                        print("🧠 Chargement de Phi-1.5 depuis assets/ia/phi-1_5/ (local)...")
                        self.tokenizer_phi = AutoTokenizer.from_pretrained(phi_local_path)
                        self.model_phi = AutoModelForCausalLM.from_pretrained(phi_local_path)
                        print("✅ Cerveau Phi-1.5 chargé depuis assets/ia/phi-1_5/ et prêt à utiliser les connaissances RAG")
                    except Exception as e_phi:
                        print(f"⚠️ Erreur chargement Phi-1.5 local: {e_phi}")

                # Charger CodeLlama-7B depuis le dossier local assets/ia/codellama-7b/
                codellama_local_path = os.path.join(os.path.dirname(__file__), "..", "assets", "ia", "codellama-7b")
                if os.path.exists(codellama_local_path):
                    try:
                        print("🧠 Chargement de CodeLlama-7B depuis assets/ia/codellama-7b/ (local)...")
                        self.tokenizer_codellama = AutoTokenizer.from_pretrained(codellama_local_path)
                        self.model_codellama = AutoModelForCausalLM.from_pretrained(codellama_local_path)
                        print("✅ Cerveau CodeLlama-7B chargé depuis assets/ia/codellama-7b/ et prêt pour la génération créative")
                    except Exception as e_codellama:
                        print(f"⚠️ Erreur chargement CodeLlama-7B local: {e_codellama}")

                # Essayer Phi-1.5 depuis HuggingFace si pas local
                if self.model_phi is None:
                    try:
                        print("🧠 Chargement de Phi-1.5 depuis HuggingFace...")
                        self.tokenizer_phi = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
                        self.model_phi = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
                        print("✅ Cerveau Phi-1.5 chargé depuis HuggingFace")
                    except Exception as e_phi:
                        print(f"⚠️ Phi-1.5 non disponible: {e_phi}")

                # Essayer CodeLlama-7B depuis HuggingFace si pas local
                if self.model_codellama is None:
                    try:
                        print("🧠 Chargement de CodeLlama-7B depuis HuggingFace...")
                        self.tokenizer_codellama = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
                        self.model_codellama = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
                        print("✅ Cerveau CodeLlama-7B chargé depuis HuggingFace")
                    except Exception as e_codellama:
                        print(f"⚠️ CodeLlama-7B non disponible: {e_codellama}")

                if self.model_phi is None and self.model_codellama is None:
                    print("❌ Aucun modèle LLM disponible")
                elif self.model_phi is None:
                    print("⚠️ Phi-1.5 non disponible, seul CodeLlama-7B sera utilisé")
                elif self.model_codellama is None:
                    print("⚠️ CodeLlama-7B non disponible, seul Phi-1.5 sera utilisé")

            except Exception as e:
                print(f"❌ Erreur chargement cerveaux: {e}")
                print("💡 Le système fonctionnera en mode dégradé sans LLM")

    def switch_brain(self, brain_name):
        """Bascule entre les cerveaux disponibles"""
        if brain_name == "phi-1_5" and self.model_phi is not None:
            self.current_brain = "phi-1_5"
            print("🧠 Cerveau basculé vers Phi-1.5 (rapide, temps réel)")
        elif brain_name == "codellama-7b" and self.model_codellama is not None:
            self.current_brain = "codellama-7b"
            print("🧠 Cerveau basculé vers CodeLlama-7B (créatif, génération)")
        else:
            print(f"⚠️ Cerveau '{brain_name}' non disponible")
            return False
        return True

    def get_current_tokenizer_and_model(self):
        """Retourne le tokenizer et modèle du cerveau actuel"""
        if self.current_brain == "phi-1_5":
            return self.tokenizer_phi, self.model_phi
        elif self.current_brain == "codellama-7b":
            return self.tokenizer_codellama, self.model_codellama
        return None, None

    def query_brain(self, query, context=""):
        """Interroge le cerveau avec les outils disponibles (méthode legacy)"""
        return self.query_brain_with_knowledge(query, context)

    def parse_cell(self, content):
        """Parse une cellule Kibali et extrait les informations"""
        cell_info = {
            "name": None,
            "imports": [],
            "fields": {},
            "actions": [],
            "evolution": False
        }

        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('cellule '):
                match = re.match(r'cellule\s+([A-Za-z_][A-Za-z0-9_]*)', line)
                if match:
                    cell_info["name"] = match.group(1)
            elif line.startswith('importe '):
                # Ex: importe IA:vision.feuilles
                match = re.match(r'importe\s+(.*)', line)
                if match:
                    cell_info["imports"].append(match.group(1))
            elif ':' in line and not line.startswith('//'):
                # Champs comme couleur: "vert"
                key, value = line.split(':', 1)
                cell_info["fields"][key.strip()] = value.strip()
            elif line.startswith('action '):
                match = re.match(r'action\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)', line)
                if match:
                    cell_info["actions"].append(match.group(1))
            elif line.startswith('evolution:'):
                cell_info["evolution"] = True

        return cell_info

    def load_cell(self, file_path):
        """Charge et parse une cellule .kib"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        cell_info = self.parse_cell(content)
        self.cells[cell_info["name"]] = cell_info
        return cell_info

    def load_memory(self, memory_name):
        """Charge une mémoire depuis les fichiers JSON"""
        # Gérer les formats "type:nom" ou juste "nom"
        if ':' in memory_name:
            parts = memory_name.split(':', 1)
            memory_file = f"memories/{parts[1]}.json"
        else:
            memory_file = f"memories/{memory_name}.json"

        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def process_pdf_for_rag(self, pdf_path):
        """Traiter un PDF pour le système RAG"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""

            for page in doc:
                text += page.get_text() + "\n"

            doc.close()

            # Découper en chunks
            chunks = self.rag_system.chunk_text(text)

            # Encoder les chunks
            embeddings = self.rag_system.encode_chunks(chunks)

            # Créer les métadonnées
            metadata = []
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "document": pdf_path,
                    "chunk": chunk,
                    "chunk_id": i,
                    "timestamp": "2025-11-21"
                })

            return {
                "chunks": chunks,
                "embeddings": embeddings,
                "metadata": metadata
            }

        except ImportError:
            print("PyMuPDF non installé. Installation recommandée: pip install PyMuPDF")
            return None
        except Exception as e:
            print(f"Erreur traitement PDF: {e}")
            return None

    def build_rag_index(self, pdf_path, index_path="rag/indexes/document.index", metadata_path="rag/metadata/document.json"):
        """Construire un index RAG à partir d'un PDF"""
        # Traiter le PDF
        pdf_data = self.process_pdf_for_rag(pdf_path)
        if pdf_data is None:
            return False

        # Construire l'index
        self.rag_system.metadata = pdf_data["metadata"]
        self.rag_system.build_index(pdf_data["embeddings"])

        # Sauvegarder
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        self.rag_system.save_index(index_path)
        self.rag_system.save_metadata(metadata_path)

        print(f"✅ Index RAG créé: {len(pdf_data['chunks'])} chunks, {pdf_data['embeddings'].shape[0]} embeddings")
        return True

    def query_rag(self, query, index_path="rag/indexes/document.index", metadata_path="rag/metadata/document.json", top_k=3):
        """Interroger le système RAG"""
        # Charger l'index si nécessaire
        if self.rag_system.index is None:
            self.rag_system.load_index(index_path)
            self.rag_system.load_metadata(metadata_path)

        if self.rag_system.index is None:
            return "Index RAG non trouvé"

        # Rechercher
        results = self.rag_system.search(query, top_k=top_k)

        if not results:
            return "Aucun résultat trouvé"

        # Formater les résultats
        response = f"Résultats pour '{query}':\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result['chunk'][:200]}...\n"
            response += f"   (Distance: {result['distance']:.3f})\n\n"

        return response

    def simulate_cell(self, cell_name):
        """Simule l'exécution d'une cellule avec mémoire, cerveau et connaissances RAG"""
        if cell_name not in self.cells:
            return {"error": "Cellule non trouvée"}

        cell = self.cells[cell_name]

        # Charger la mémoire si spécifiée
        memory_data = {}
        if "memoire" in cell["fields"]:
            mem_key = cell["fields"]["memoire"].strip('"')
            memory_data = self.load_memory(mem_key)

        # État actuel de la cellule pour le contexte
        cell_state = {
            "nom": cell["name"],
            "champs": cell["fields"],
            "actions": cell["actions"],
            "evolution_active": cell["evolution"],
            "memoire": memory_data
        }

        # Déterminer le cerveau à utiliser pour cette cellule
        brain_to_use = self.current_brain  # cerveau par défaut
        if "cerveau_principal" in cell["fields"]:
            brain_config = cell["fields"]["cerveau_principal"].strip('"')
            if brain_config == "phi-1_5":
                brain_to_use = "phi-1_5"
            elif brain_config == "codellama-7b":
                brain_to_use = "codellama-7b"

        # Sauvegarder le cerveau actuel et basculer si nécessaire
        original_brain = self.current_brain
        if brain_to_use != self.current_brain:
            self.switch_brain(brain_to_use)
            print(f"🧠 Cerveau basculé vers {brain_to_use} pour la cellule {cell_name}")

        # Traiter les actions spéciales pour RAGTrainer
        if cell_name == "RAGTrainer":
            results = {}
            for action in cell["actions"]:
                if action == "construire_index":
                    pdf_path = cell["fields"].get("pdf_path", "").strip('"')
                    if pdf_path and os.path.exists(pdf_path):
                        index_path = cell["fields"].get("output_index", "rag/indexes/document.index").strip('"')
                        metadata_path = cell["fields"].get("output_metadata", "rag/metadata/document.json").strip('"')
                        success = self.build_rag_index(pdf_path, index_path, metadata_path)
                        results[action] = "succès" if success else "échec"

                        # Notifier l'évolution basée sur cette expérience
                        if success:
                            experience = {"action": "construction_index", "resultat": "succès", "pdf": pdf_path}
                            self.evolve_cell_behavior(cell_name, experience)
                    else:
                        results[action] = f"PDF non trouvé: {pdf_path}"
                elif action == "tester_index":
                    index_path = cell["fields"].get("output_index", "rag/indexes/document.index").strip('"')
                    metadata_path = cell["fields"].get("output_metadata", "rag/metadata/document.json").strip('"')
                    test_query = "Quelle est la structure des arbres ?"
                    response = self.query_rag(test_query, index_path, metadata_path, top_k=3)
                    results[action] = "index valide" if "Résultats pour" in response else "index invalide"
                else:
                    results[action] = f"action {action} non reconnue"

            # Simulation : générer un état basé sur les champs et mémoire
            etat = {
                "nom": cell["name"],
                "imports": cell["imports"],
                "champs": cell["fields"],
                "actions": cell["actions"],
                "evolution_active": cell["evolution"],
                "memoire_chargee": memory_data,
                "resultats_actions": results,
                "status": "vivant",
                "temperature": 25,
                "mouvement": "actif",
                "reaction": "adaptation en cours"
            }

            # Restaurer le cerveau original
            if brain_to_use != original_brain:
                self.switch_brain(original_brain)

            return etat

        # Pour les autres cellules : utiliser le cerveau enrichi avec connaissances
        brain_responses = {}
        autonomous_decisions = {}

        for action in cell["actions"]:
            # Interroger le cerveau avec connaissances pour chaque action
            query = f"Comment la cellule {cell_name} devrait-elle exécuter l'action '{action}' ?"
            context = f"Cellule: {cell_name}, Température: 25°C, Mémoire: {json.dumps(memory_data, ensure_ascii=False)}"

            brain_response = self.query_brain_with_knowledge(query, context, cell_state)
            brain_responses[action] = brain_response

            # Extraire des décisions autonomes de la réponse du cerveau
            if "devrait" in brain_response.lower() or "doit" in brain_response.lower():
                autonomous_decisions[action] = self.extract_decision_from_response(brain_response)

        # Évolution automatique si activée
        if cell["evolution"]:
            # Collecter l'expérience de cette simulation
            experience = {
                "timestamp": "2025-11-21",
                "actions_executees": cell["actions"],
                "reponses_cerveau": brain_responses,
                "decisions_autonomes": autonomous_decisions,
                "connaissances_utilisees": len(self.get_relevant_knowledge("test")) > 0
            }
            self.evolve_cell_behavior(cell_name, experience)

        # Simulation : générer un état basé sur les champs, mémoire et décisions autonomes
        etat = {
            "nom": cell["name"],
            "imports": cell["imports"],
            "champs": cell["fields"],
            "actions": cell["actions"],
            "evolution_active": cell["evolution"],
            "memoire_chargee": memory_data,
            "reponses_cerveau": brain_responses,
            "decisions_autonomes": autonomous_decisions,
            "connaissances_integrees": len(self.get_relevant_knowledge("test")) > 0,
            "status": "vivant",
            "temperature": 25,
            "mouvement": "actif",
            "reaction": "adaptation en cours"
        }

        # Restaurer le cerveau original
        if brain_to_use != original_brain:
            self.switch_brain(original_brain)

        return etat

    def extract_decision_from_response(self, brain_response):
        """Extrait les décisions autonomes de la réponse du cerveau"""
        decisions = []

        # Analyser la réponse pour extraire des décisions
        response_lower = brain_response.lower()

        if "bouger" in response_lower:
            decisions.append("modifier_mouvement")
        if "adapter" in response_lower or "adaptation" in response_lower:
            decisions.append("adapter_comportement")
        if "apprendre" in response_lower:
            decisions.append("collecter_connaissances")
        if "évoluer" in response_lower:
            decisions.append("declencher_evolution")

        return decisions

    def animate_scene(self, description):
        """Génère et anime une scène 3D complète à partir d'une description"""
        print("🎬 Démarrage de l'animation Kibali...")
        print(f"📝 Description: {description}")

        # Charger le Directeur IA
        director_file = "cells/3d/Director.kib"
        if not os.path.exists(director_file):
            return json.dumps({"error": "Directeur IA non trouvé. Créez cells/3d/Director.kib"})

        director_info = self.load_cell(director_file)
        print(f"🎭 Directeur chargé: {director_info['name']}")

        # Basculer vers CodeLlama pour la génération créative
        self.switch_brain("codellama-7b")

        # Générer le scénario avec le Directeur
        scenario_query = f"Écris un scénario détaillé pour cette animation: {description}"
        scenario = self.query_brain_with_knowledge(scenario_query, cell_state=director_info)
        print("📜 Scénario généré par le Directeur IA")

        # Analyser le scénario pour extraire les éléments
        elements = self.extract_scene_elements(scenario)
        print(f"🎭 Éléments identifiés: {len(elements)} cellules à créer")

        # Créer dynamiquement les cellules nécessaires
        created_cells = []
        for element in elements:
            cell_file = self.create_dynamic_cell(element)
            if cell_file:
                created_cells.append(cell_file)
                print(f"🧬 Cellule créée: {element['type']} - {element['name']}")

        # Charger toutes les cellules créées
        for cell_file in created_cells:
            self.load_cell(cell_file)

        # Basculer vers Phi-1.5 pour l'animation temps réel
        self.switch_brain("phi-1_5")

        # Simuler l'animation scène par scène
        animation_result = {
            "description": description,
            "scenario": scenario,
            "cells_creees": created_cells,
            "scenes": []
        }

        # Simuler quelques scènes
        for i in range(3):  # 3 scènes pour l'exemple
            scene_result = self.simulate_scene(i + 1, created_cells)
            animation_result["scenes"].append(scene_result)
            print(f"🎬 Scène {i+1} simulée")

        # Générer un résumé final
        summary = self.generate_animation_summary(animation_result)
        animation_result["resume"] = summary

        print("✅ Animation terminée!")
        return json.dumps(animation_result, indent=2, ensure_ascii=False)

    def extract_scene_elements(self, scenario):
        """Extrait les éléments de scène du scénario généré"""
        elements = []

        # Analyse simple du texte pour identifier les éléments
        lines = scenario.lower().split('\n')
        for line in lines:
            if 'personnage' in line or 'héros' in line or 'héroïne' in line:
                elements.append({
                    "type": "Character",
                    "name": "Protagoniste",
                    "template": "cells/3d/Character.kib"
                })
            elif 'arbre' in line or 'plante' in line:
                elements.append({
                    "type": "OrganicTree",
                    "name": "Arbre Ancien",
                    "template": "cells/3d/Tree.kib"
                })
            elif 'caméra' in line or 'vue' in line:
                elements.append({
                    "type": "Camera",
                    "name": "Main Camera",
                    "template": "cells/3d/Camera.kib"
                })

        # Éliminer les doublons
        unique_elements = []
        seen = set()
        for element in elements:
            key = (element["type"], element["name"])
            if key not in seen:
                unique_elements.append(element)
                seen.add(key)

        return unique_elements

    def create_dynamic_cell(self, element):
        """Crée une cellule dynamique basée sur un template"""
        template_file = element["template"]
        if not os.path.exists(template_file):
            print(f"⚠️ Template non trouvé: {template_file}")
            return None

        # Lire le template
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # Personnaliser le contenu
        cell_content = template_content.replace(
            f'cellule {element["type"]}',
            f'cellule {element["name"].replace(" ", "")}'
        )
        cell_content = cell_content.replace(
            f'nom: "{element["type"]}"',
            f'nom: "{element["name"]}"'
        )

        # Créer le fichier de cellule
        cell_filename = f"cells/3d/{element['name'].replace(' ', '')}.kib"
        os.makedirs(os.path.dirname(cell_filename), exist_ok=True)

        with open(cell_filename, 'w', encoding='utf-8') as f:
            f.write(cell_content)

        return cell_filename

    def simulate_scene(self, scene_number, cell_files):
        """Simule une scène avec toutes les cellules"""
        scene_result = {
            "scene": scene_number,
            "cellules_actives": [],
            "actions_executees": [],
            "etat_global": "animé"
        }

        for cell_file in cell_files:
            cell_name = os.path.basename(cell_file).replace('.kib', '')
            if cell_name in self.cells:
                # Simuler la cellule
                cell_result = self.simulate_cell(cell_name)
                scene_result["cellules_actives"].append({
                    "nom": cell_name,
                    "etat": cell_result
                })

                # Collecter les actions
                if "reponses_cerveau" in cell_result:
                    for action, response in cell_result["reponses_cerveau"].items():
                        scene_result["actions_executees"].append({
                            "cellule": cell_name,
                            "action": action,
                            "reponse": response[:100] + "..." if len(response) > 100 else response
                        })

        return scene_result

    def generate_animation_summary(self, animation_result):
        """Génère un résumé de l'animation"""
        total_scenes = len(animation_result["scenes"])
        total_cells = len(animation_result["cells_creees"])

        summary = f"""
🎬 Animation Kibali - Résumé

📝 Description originale: {animation_result["description"]}

📊 Statistiques:
- Scènes générées: {total_scenes}
- Cellules créées: {total_cells}
- Cerveaux utilisés: Phi-1.5 (temps réel) + CodeLlama-7B (créatif)

🎭 Cellules actives:
"""

        for cell in animation_result["cells_creees"]:
            cell_name = os.path.basename(cell).replace('.kib', '')
            summary += f"- {cell_name}\n"

        summary += "\n✅ Animation terminée avec succès!"

        return summary

    def run_program(self, main_file):
        """Exécute un programme Kibali"""
        cell_info = self.load_cell(main_file)
        result = self.simulate_cell(cell_info["name"])
        return json.dumps(result, indent=2, ensure_ascii=False)

    def compile_to_platform(self, cell_file, platform, output_dir=None):
        """Compile une cellule vers une plateforme spécifique"""
        compiler = KibaliCompiler()
        return compiler.compile_cell_to_platform(cell_file, platform, output_dir)


class KibaliCompiler:
    """Compilateur multi-plateforme pour cellules Kibali"""

    def __init__(self):
        self.supported_platforms = {
            "android": self.compile_android,
            "ios": self.compile_ios,
            "web": self.compile_web,
            "desktop": self.compile_desktop
        }
        self.runtime = KibaliRuntime()

    def detect_platform(self):
        """Détecte la plateforme actuelle"""
        system = platform.system().lower()
        if system == "linux":
            return "desktop"
        elif system == "darwin":
            return "desktop"  # macOS peut compiler pour iOS
        elif system == "windows":
            return "desktop"
        return "unknown"

    def compile_cell_to_platform(self, cell_file, target_platform, output_dir=None):
        """Compile une cellule Kibali vers une plateforme cible"""
        if target_platform not in self.supported_platforms:
            return {"error": f"Plateforme non supportée: {target_platform}"}

        if not os.path.exists(cell_file):
            return {"error": f"Fichier cellule non trouvé: {cell_file}"}

        # Charger et analyser la cellule
        cell_info = self.runtime.load_cell(cell_file)

        # Créer le répertoire de sortie
        if output_dir is None:
            output_dir = f"build/{target_platform}/{cell_info['name']}"

        os.makedirs(output_dir, exist_ok=True)

        # Compiler selon la plateforme
        try:
            result = self.supported_platforms[target_platform](cell_info, output_dir)
            result["platform"] = target_platform
            result["cell_name"] = cell_info["name"]
            result["output_dir"] = output_dir
            return result
        except Exception as e:
            return {
                "error": f"Erreur compilation {target_platform}: {str(e)}",
                "platform": target_platform,
                "cell_name": cell_info["name"]
            }

    def compile_android(self, cell_info, output_dir):
        """Compile vers Android avec Kivy/Buildozer"""
        if not KIVY_AVAILABLE:
            return {"error": "Kivy non installé. Installez avec: pip install kivy"}

        if not BUILDOZER_AVAILABLE:
            return {"error": "Buildozer non installé. Installez avec: pip install buildozer"}

        # Générer le code Kivy pour la cellule
        kivy_code = self.generate_kivy_app(cell_info)

        # Écrire le fichier main.py
        main_py = os.path.join(output_dir, "main.py")
        with open(main_py, 'w', encoding='utf-8') as f:
            f.write(kivy_code)

        # Générer le fichier buildozer.spec
        spec_content = self.generate_buildozer_spec(cell_info)
        spec_file = os.path.join(output_dir, "buildozer.spec")
        with open(spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)

        # Générer le fichier requirements.txt
        requirements = [
            "kivy",
            "sentence-transformers",
            "faiss-cpu",
            "transformers",
            "torch",
            "numpy"
        ]
        req_file = os.path.join(output_dir, "requirements.txt")
        with open(req_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(requirements))

        return {
            "success": True,
            "message": "Application Android générée avec Kivy",
            "files_created": [main_py, spec_file, req_file],
            "build_command": f"cd {output_dir} && buildozer android debug"
        }

    def compile_ios(self, cell_info, output_dir):
        """Compile vers iOS avec BeeWare/Briefcase"""
        if not BRIEFCASE_AVAILABLE:
            return {"error": "Briefcase non installé. Installez avec: pip install briefcase"}

        # Générer le code pour iOS
        ios_code = self.generate_ios_app(cell_info)

        # Écrire le fichier main.py
        main_py = os.path.join(output_dir, "main.py")
        with open(main_py, 'w', encoding='utf-8') as f:
            f.write(ios_code)

        # Générer pyproject.toml pour Briefcase
        toml_content = self.generate_briefcase_toml(cell_info)
        toml_file = os.path.join(output_dir, "pyproject.toml")
        with open(toml_file, 'w', encoding='utf-8') as f:
            f.write(toml_content)

        return {
            "success": True,
            "message": "Application iOS générée avec Briefcase",
            "files_created": [main_py, toml_file],
            "build_command": f"cd {output_dir} && briefcase build ios"
        }

    def compile_web(self, cell_info, output_dir):
        """Compile vers Web avec Transcrypt"""
        if not TRANSCRYPT_AVAILABLE:
            return {"error": "Transcrypt non installé. Installez avec: pip install transcrypt"}

        # Générer le code JavaScript pour le web
        web_code = self.generate_web_app(cell_info)

        # Écrire le fichier main.py
        main_py = os.path.join(output_dir, "main.py")
        with open(main_py, 'w', encoding='utf-8') as f:
            f.write(web_code)

        # Générer index.html
        html_content = self.generate_web_html(cell_info)
        html_file = os.path.join(output_dir, "index.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return {
            "success": True,
            "message": "Application Web générée avec Transcrypt",
            "files_created": [main_py, html_file],
            "build_command": f"cd {output_dir} && transcrypt -b -m -n main.py"
        }

    def compile_desktop(self, cell_info, output_dir):
        """Compile vers Desktop avec PyInstaller"""
        if not PYINSTALLER_AVAILABLE:
            return {"error": "PyInstaller non installé. Installez avec: pip install pyinstaller"}

        # Générer le code desktop
        desktop_code = self.generate_desktop_app(cell_info)

        # Écrire le fichier main.py
        main_py = os.path.join(output_dir, "main.py")
        with open(main_py, 'w', encoding='utf-8') as f:
            f.write(desktop_code)

        # Générer le fichier spec pour PyInstaller
        spec_content = self.generate_pyinstaller_spec(cell_info)
        spec_file = os.path.join(output_dir, f"{cell_info['name']}.spec")
        with open(spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)

        return {
            "success": True,
            "message": "Application Desktop générée avec PyInstaller",
            "files_created": [main_py, spec_file],
            "build_command": f"cd {output_dir} && pyinstaller {cell_info['name']}.spec"
        }

    def generate_kivy_app(self, cell_info):
        """Génère le code Kivy pour une cellule"""
        app_name = cell_info["name"]
        fields = cell_info["fields"]

        code = f'''#!/usr/bin/env python3
"""
Application Android Kibali - {app_name}
Générée automatiquement depuis cellule Kibali
"""

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
import json
import os
import sys

# Ajouter le chemin vers kibali.py
sys.path.insert(0, os.path.dirname(__file__))

class KibaliCellApp(App):
    def build(self):
        self.title = "Kibali - {app_name}"
        self.runtime = None

        # Layout principal
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Titre
        title = Label(
            text=f"🧬 Cellule {app_name}",
            font_size=24,
            size_hint_y=0.1
        )
        layout.add_widget(title)

        # Zone d'état
        self.status_label = Label(
            text="Initialisation...",
            font_size=16,
            size_hint_y=0.2
        )
        layout.add_widget(self.status_label)

        # Zone d'actions
        actions_layout = BoxLayout(orientation='horizontal', size_hint_y=0.2)
        for action in {cell_info["actions"]}:
            btn = Button(text=action.replace('_', ' ').title())
            btn.bind(on_press=lambda instance, act=action: self.execute_action(act))
            actions_layout.add_widget(btn)
        layout.add_widget(actions_layout)

        # Zone de sortie
        self.output_text = TextInput(
            text="",
            readonly=True,
            size_hint_y=0.4
        )
        layout.add_widget(self.output_text)

        # Bouton de simulation
        simulate_btn = Button(
            text="Simuler Cellule",
            size_hint_y=0.1
        )
        simulate_btn.bind(on_press=self.simulate_cell)
        layout.add_widget(simulate_btn)

        # Initialiser après le build
        Clock.schedule_once(self.initialize_runtime, 0.1)

        return layout

    def initialize_runtime(self, dt):
        try:
            from kibali import KibaliRuntime
            self.runtime = KibaliRuntime()
            self.status_label.text = "✅ Runtime initialisé"
        except Exception as e:
            self.status_label.text = f"❌ Erreur: {{e}}"

    def simulate_cell(self, instance):
        if self.runtime is None:
            self.output_text.text = "Runtime non initialisé"
            return

        try:
            result = self.runtime.simulate_cell("{app_name}")
            self.output_text.text = json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            self.output_text.text = f"Erreur simulation: {{e}}"

    def execute_action(self, action):
        if self.runtime is None:
            self.output_text.text = "Runtime non initialisé"
            return

        try:
            # Interroger le cerveau pour cette action
            query = f"Comment exécuter l'action '{{action}}' pour la cellule {app_name}?"
            response = self.runtime.query_brain_with_knowledge(query)
            self.output_text.text = f"Action {{action}}:\\n{{response}}"
        except Exception as e:
            self.output_text.text = f"Erreur action: {{e}}"

if __name__ == '__main__':
    KibaliCellApp().run()
'''
        return code

    def generate_ios_app(self, cell_info):
        """Génère le code iOS pour une cellule"""
        app_name = cell_info["name"]

        code = f'''#!/usr/bin/env python3
"""
Application iOS Kibali - {app_name}
Générée automatiquement depuis cellule Kibali
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import json
import os
import sys

# Ajouter le chemin vers kibali.py
sys.path.insert(0, os.path.dirname(__file__))

class KibaliCellApp(toga.App):
    def startup(self):
        self.main_window = toga.MainWindow(title=f"Kibali - {app_name}")

        # Layout principal
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))

        # Titre
        title_label = toga.Label(
            f"🧬 Cellule {app_name}",
            style=Pack(padding=(0, 5))
        )
        main_box.add(title_label)

        # Zone d'état
        self.status_label = toga.Label(
            "Initialisation...",
            style=Pack(padding=(0, 5))
        )
        main_box.add(self.status_label)

        # Zone d'actions
        actions_box = toga.Box(style=Pack(direction=ROW, padding=(0, 5)))
        for action in {cell_info["actions"]}:
            btn = toga.Button(
                action.replace('_', ' ').title(),
                on_press=lambda widget, act=action: self.execute_action(act)
            )
            actions_box.add(btn)
        main_box.add(actions_box)

        # Zone de sortie
        self.output_text = toga.MultilineTextInput(
            readonly=True,
            style=Pack(flex=1, padding=(0, 5))
        )
        main_box.add(self.output_text)

        # Bouton de simulation
        simulate_btn = toga.Button(
            "Simuler Cellule",
            on_press=self.simulate_cell,
            style=Pack(padding=(5, 0))
        )
        main_box.add(simulate_btn)

        self.main_window.content = main_box
        self.main_window.show()

        # Initialiser le runtime
        self.initialize_runtime()

    def initialize_runtime(self):
        try:
            from kibali import KibaliRuntime
            self.runtime = KibaliRuntime()
            self.status_label.text = "✅ Runtime initialisé"
        except Exception as e:
            self.status_label.text = f"❌ Erreur: {{e}}"

    def simulate_cell(self, widget):
        if not hasattr(self, 'runtime') or self.runtime is None:
            self.output_text.value = "Runtime non initialisé"
            return

        try:
            result = self.runtime.simulate_cell("{app_name}")
            self.output_text.value = json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            self.output_text.value = f"Erreur simulation: {{e}}"

    def execute_action(self, action, widget=None):
        if not hasattr(self, 'runtime') or self.runtime is None:
            self.output_text.value = "Runtime non initialisé"
            return

        try:
            query = f"Comment exécuter l'action '{{action}}' pour la cellule {app_name}?"
            response = self.runtime.query_brain_with_knowledge(query)
            self.output_text.value = f"Action {{action}}:\\n{{response}}"
        except Exception as e:
            self.output_text.value = f"Erreur action: {{e}}"

def main():
    return KibaliCellApp(app_id="org.kibali.{app_name.lower()}", app_name="{app_name}")

if __name__ == '__main__':
    main().main_loop()
'''
        return code

    def generate_web_app(self, cell_info):
        """Génère le code Web pour une cellule"""
        app_name = cell_info["name"]

        code = f'''#!/usr/bin/env python3
"""
Application Web Kibali - {app_name}
Générée automatiquement depuis cellule Kibali
"""

import json
import os
import sys

# Simulation d'interface web pour Transcrypt
class WebInterface:
    def __init__(self):
        self.runtime = None
        self.output_element = None

    def initialize(self):
        try:
            # En mode web, on simule le runtime
            print("Initialisation interface web...")
            self.runtime = WebKibaliRuntime()
            return True
        except Exception as e:
            print(f"Erreur initialisation: {{e}}")
            return False

    def simulate_cell(self):
        if self.runtime is None:
            return "Runtime non initialisé"

        try:
            result = self.runtime.simulate_cell("{app_name}")
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Erreur simulation: {{e}}"

    def execute_action(self, action):
        if self.runtime is None:
            return "Runtime non initialisé"

        try:
            query = f"Comment exécuter l'action '{{action}}' pour la cellule {app_name}?"
            response = self.runtime.query_brain_with_knowledge(query)
            return f"Action {{action}}:\\n{{response}}"
        except Exception as e:
            return f"Erreur action: {{e}}"

class WebKibaliRuntime:
    """Runtime Kibali simplifié pour le web"""

    def __init__(self):
        self.cells = {{
            "{app_name}": {{
                "name": "{app_name}",
                "actions": {cell_info["actions"]},
                "fields": {cell_info["fields"]}
            }}
        }}

    def simulate_cell(self, cell_name):
        cell = self.cells.get(cell_name)
        if not cell:
            return {{"error": "Cellule non trouvée"}}

        return {{
            "nom": cell["name"],
            "actions": cell["actions"],
            "champs": cell["fields"],
            "status": "vivant (web)",
            "temperature": 25,
            "mouvement": "actif",
            "reaction": "interface web"
        }}

    def query_brain_with_knowledge(self, query):
        return f"Réponse simulée pour: {{query}} (mode web sans LLM complet)"

# Interface globale pour JavaScript
web_interface = WebInterface()

def initialize_web_app():
    """Fonction appelée depuis JavaScript pour initialiser"""
    return web_interface.initialize()

def simulate_cell_web():
    """Fonction appelée depuis JavaScript pour simuler"""
    return web_interface.simulate_cell()

def execute_action_web(action):
    """Fonction appelée depuis JavaScript pour exécuter une action"""
    return web_interface.execute_action(action)

# Point d'entrée principal
if __name__ == '__main__':
    print("Application Web Kibali - {app_name}")
    print("Utilisez les fonctions JavaScript pour interagir")
'''
        return code

    def generate_desktop_app(self, cell_info):
        """Génère le code Desktop pour une cellule"""
        app_name = cell_info["name"]

        code = f'''#!/usr/bin/env python3
"""
Application Desktop Kibali - {app_name}
Générée automatiquement depuis cellule Kibali
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import os
import sys
import threading

# Ajouter le chemin vers kibali.py
sys.path.insert(0, os.path.dirname(__file__))

class KibaliDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Kibali - {app_name}")
        self.root.geometry("800x600")

        self.runtime = None
        self.create_widgets()
        self.initialize_runtime()

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Titre
        title_label = ttk.Label(
            main_frame,
            text=f"🧬 Cellule {app_name}",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Zone d'état
        ttk.Label(main_frame, text="État:").grid(row=1, column=0, sticky=tk.W)
        self.status_label = ttk.Label(main_frame, text="Initialisation...")
        self.status_label.grid(row=1, column=1, sticky=tk.W)

        # Zone d'actions
        ttk.Label(main_frame, text="Actions:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        actions_frame = ttk.Frame(main_frame)
        actions_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))

        for i, action in enumerate({cell_info["actions"]}):
            btn = ttk.Button(
                actions_frame,
                text=action.replace('_', ' ').title(),
                command=lambda act=action: self.execute_action(act)
            )
            btn.grid(row=0, column=i, padx=(0, 5))

        # Zone de sortie
        ttk.Label(main_frame, text="Sortie:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.output_text = scrolledtext.ScrolledText(
            main_frame,
            width=70,
            height=15,
            wrap=tk.WORD
        )
        self.output_text.grid(row=5, column=0, columnspan=2, pady=(0, 10))

        # Bouton de simulation
        simulate_btn = ttk.Button(
            main_frame,
            text="Simuler Cellule",
            command=self.simulate_cell
        )
        simulate_btn.grid(row=6, column=0, columnspan=2)

        # Configuration grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def initialize_runtime(self):
        def init_thread():
            try:
                from kibali import KibaliRuntime
                self.runtime = KibaliRuntime()
                self.status_label.config(text="✅ Runtime initialisé")
            except Exception as e:
                self.status_label.config(text=f"❌ Erreur: {{e}}")

        thread = threading.Thread(target=init_thread)
        thread.daemon = True
        thread.start()

    def simulate_cell(self):
        if self.runtime is None:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Runtime non initialisé")
            return

        def simulate_thread():
            try:
                result = self.runtime.simulate_cell("{app_name}")
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as e:
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"Erreur simulation: {{e}}")

        thread = threading.Thread(target=simulate_thread)
        thread.daemon = True
        thread.start()

    def execute_action(self, action):
        if self.runtime is None:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Runtime non initialisé")
            return

        def action_thread():
            try:
                query = f"Comment exécuter l'action '{{action}}' pour la cellule {app_name}?"
                response = self.runtime.query_brain_with_knowledge(query)
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"Action {{action}}:\\n{{response}}")
            except Exception as e:
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"Erreur action: {{e}}")

        thread = threading.Thread(target=action_thread)
        thread.daemon = True
        thread.start()

def main():
    root = tk.Tk()
    app = KibaliDesktopApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
'''
        return code

    def generate_buildozer_spec(self, cell_info):
        """Génère le fichier buildozer.spec pour Android"""
        app_name = cell_info["name"]

        spec = f'''[app]

# (str) Title of your application
title = Kibali - {app_name}

# (str) Package name
package.name = kibali_{app_name.lower()}

# (str) Package domain (needed for android/ios packaging)
package.domain = org.kibali

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas

# (list) List of inclusions using pattern matching
#source.include_patterns = assets/*,images/*.png

# (list) Source files to exclude (let empty to not exclude anything)
#source.exclude_exts = spec

# (list) List of directory to exclude (let empty to not exclude anything)
#source.exclude_dirs = tests, bin

# (list) List of exclusions using pattern matching
#source.exclude_patterns = license,images/*/*.jpg

# (str) Application versioning (method 1)
version = 1.0.0

# (str) Application versioning (method 2)
# version.regex = __version__ = ['"](.*)['"]
# version.filename = %(source.dir)s/main.py

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,sentence-transformers,faiss-cpu,transformers,torch,numpy

# (str) Custom source folders for requirements
# Sets custom source for any requirements with recipes
# requirements.source.kivy = ../../kivy

# (list) Garden requirements
#garden_requirements =

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/data/icon.png

# (str) Supported orientation (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (list) List of service to declare
#services = NAME:ENTRYPOINT_TO_PY,NAME2:ENTRYPOINT_TO_PY

#
# OSX Specific
#

#
# author = © Copyright Info

# change the major version of python used by the app
osx.python_version = 3

# Kivy version to use
osx.kivy_version = 2.0.0

#
# Android specific
#

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (string) Presplash background color (for new android toolchain)
# Supported formats are: #RRGGBB #AARRGGBB or one of the following names:
# red, blue, green, black, white, gray, cyan, magenta, yellow, lightgray,
# darkgray, grey, lightgrey, darkgrey, aqua, fuchsia, lime, maroon, navy,
# olive, purple, silver, teal.
#android.presplash_color = #FFFFFF

# (list) Permissions
android.permissions = INTERNET,ACCESS_NETWORK_STATE

# (int) Target Android API, should be as high as possible.
android.api = 31

# (int) Minimum API your APK will support.
android.minapi = 21

# (int) Android SDK version to use
#android.sdk = 20

# (str) Android NDK version to use
#android.ndk = 19b

# (int) Android NDK API to use. This is the minimum API your app will support, it should usually match android.minapi.
#android.ndk_api = 21

# (bool) Use --private data storage (True) or --dir public storage (False)
#android.private_storage = True

# (str) Android NDK directory (if empty, it will be automatically downloaded.)
#android.ndk_path =

# (str) Android SDK directory (if empty, it will be automatically downloaded.)
#android.sdk_path =

# (str) ANT directory (if empty, it will be automatically downloaded.)
#android.ant_path =

# (bool) If True, then skip trying to update the Android sdk
# This can be useful to avoid excess Internet downloads or save time
# when an update is due and you just want to test/build your package
# android.skip_update = False

# (bool) If True, then automatically accept SDK license
# agreements. This is intended for automation only. If set to False,
# the default, you will be shown the license when first running
# buildozer.
android.accept_sdk_license = True

# (str) Android entry point, default is ok for Kivy apps
#android.entrypoint = org.renpy.android.PythonActivity

# (list) Pattern to whitelist for the whole project
#android.whitelist =

# (str) Path to a custom whitelist file
#android.whitelist_src =

# (str) Path to a custom blacklist file
#android.blacklist_src =

# (list) List of Java .jar files to add to the libs so that pyjnius can access
# their classes. Don't add jars that you do not need, since extra jars can slow
# down the build process. Allows wildcards matching, for example:
# OUYA-ODK/libs/*.jar
#android.add_jar =

# (list) List of Java files to add to the android project (can be java or a
# directory containing the files)
#android.add_src =

# (list) Android AAR archives to add (currently works only with a single aar)
#android.add_aar =

# (list) Gradle dependencies to add (currently works only with a single gradle)
#android.gradle_dependencies =

# (list) Java classes to add as activities to the manifest
#android.add_activites = com.example.thing.Thing

# (str) python-for-android branch to use, defaults to master
#p4a.branch = master

# (str) OUYA Console category. Should be one of GAME or APP
# If you leave this blank, OUYA support will not be enabled
#android.ouya.category = GAME

# (str) Filename of OUYA Console icon. It must be a 732x412 png image.
#android.ouya.icon.filename = %(source.dir)s/data/ouya_icon.png

# (str) XML file to include as an intent filters in <activity> tag
#android.manifest.intent_filters =

# (str) launchMode to set for the main activity
#android.manifest.launch_mode = standard

# (str) screenOrientation to set for the main activity
# ["unspecified", "behind", "landscape", "portrait",
#    "reverseLandscape", "reversePortrait", "sensorLandscape",
#    "sensorPortrait", "sensor", "fullSensor", "nosensor", "user", "fullUser",
#    "locked", "reverseUser"]
#android.manifest.orientation = portrait

# (list) Android additional libraries to copy into libs/armeabi
#android.add_libs_armeabi = libs/android/*.so
#android.add_libs_armeabi_v7a = libs/android-v7/*.so
#android.add_libs_arm64_v8a = libs/android-v8/*.so
#android.add_libs_x86 = libs/android-x86/*.so
#android.add_libs_mips = libs/android-mips/*.so

# (bool) Indicate whether the screen should stay on
# Don't forget to add the WAKE_LOCK permission if you set this to True
#android.wake_lock = False

# (list) Android application meta-data to set (key=value format)
#android.meta_data =

# (list) Android library project to add (will be added in the
# project.properties automatically.)
#android.library_references =

# (list) Android shared libraries which will be added to AndroidManifest.xml using <uses-library> tag
#android.uses_library =

# (str) Android logcat filters to use
#android.logcat_filters = *:S python:D

# (bool) Copy library instead of making a libpymodules.so
#android.copy_libs = 1

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.arch = armeabi-v7a

# (int) overrides automatic versionCode computation (used in build.gradle)
# this is not the same as app version and should only overwrite the value if you know what you're doing
# android.numeric_version = 1

#
# Python for android (p4a) specific
#

# (str) python-for-android git clone directory (if empty, it will be automatically cloned from github)
#p4a.source_dir =

# (str) The directory in which python-for-android should look for your own build recipes (if any)
#p4a.local_recipes =

# (str) Filename to the hook for p4a
#p4a.hook =

# (str) Bootstrap to use for android builds
# p4a.bootstrap = sdl2

# (int) port number to specify an explicit --port parameter to adb connect for non-default connections
#adb.port =

#
# iOS specific
#

# (str) Path to a custom kivy-ios folder
#ios.kivy_ios_dir =
# Alternately, specify the directory in a custom build of kivy-ios
#ios.kivy_ios_url =

# (str) Name of the certificate to use for signing the debug version
# Get a list of available identities: buildozer ios list_identities
#ios.codesign.debug = "iPhone Developer: Python for iOS"

# (str) Name of the certificate to use for signing the release version
#ios.codesign.release = "iPhone Distribution: Python for iOS"


[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

# (str) Path to build artifact storage, absolute or relative to spec file
# build_dir = ./.buildozer

# (str) Path to build output (i.e. .apk, .ipa) storage
# bin_dir = ./bin

#    -----------------------------------------------------------------------------
#    List as sections
#
#    You can define all the "list" as [section:key].
#    Each line will be considered as a option to the list.
#    Let's take [app] / source.exclude_exts.
#    Instead of doing:
#
#    [app]
#    source.exclude_exts = spec
#
#    This can be translated into:
#
#    [app:source.exclude_exts]
#    spec
#
#    -----------------------------------------------------------------------------
#    Profiles
#
#    You can extend section / key with a profile
#    For example, you can add a desktop profile to produce a desktop version of your application
#    You can do this by extending the section / key with a new section
#
#    [app]
#    environment = production
#
#    [app:desktop]
#    environment = debug
#
#    -----------------------------------------------------------------------------
'''
        return spec

    def generate_briefcase_toml(self, cell_info):
        """Génère le fichier pyproject.toml pour iOS"""
        app_name = cell_info["name"]

        toml = f'''[build-system]
requires = ["briefcase"]

[tool.briefcase]
project_name = "Kibali - {app_name}"
bundle = "org.kibali.{app_name.lower()}"
version = "1.0.0"
url = "https://kibali.org"
license = "MIT"
author = "Kibali Team"
author_email = "team@kibali.org"

[tool.briefcase.app.{app_name.lower()}]
formal_name = "Kibali - {app_name}"
description = "Application Kibali générée automatiquement"
icon = "icon"
sources = ["main.py"]
requires = [
    "toga",
    "sentence-transformers",
    "faiss-cpu",
    "transformers",
    "torch",
    "numpy"
]

[tool.briefcase.app.{app_name.lower()}.macOS]
requires = []

[tool.briefcase.app.{app_name.lower()}.linux]
requires = []

[tool.briefcase.app.{app_name.lower()}.windows]
requires = []

[tool.briefcase.app.{app_name.lower()}.iOS]
requires = []
'''
        return toml

    def generate_web_html(self, cell_info):
        """Génère le fichier HTML pour l'application web"""
        app_name = cell_info["name"]

        html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kibali - {app_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        .status {{
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }}
        button {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background-color: #2980b9;
        }}
        .output {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            white-space: pre-wrap;
            font-family: monospace;
            max-height: 400px;
            overflow-y: auto;
        }}
        .simulate-btn {{
            background-color: #27ae60 !important;
        }}
        .simulate-btn:hover {{
            background-color: #229954 !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Kibali - {app_name}</h1>

        <div class="status" id="status">
            Initialisation...
        </div>

        <div class="actions" id="actions">
            <!-- Les boutons d'actions seront ajoutés par JavaScript -->
        </div>

        <div class="output" id="output">
            <!-- La sortie sera affichée ici -->
        </div>

        <button class="simulate-btn" onclick="simulateCell()">
            Simuler Cellule
        </button>
    </div>

    <script>
        // Interface avec le code Python transpilé
        let webInterface = null;

        // Initialisation
        window.onload = function() {{
            initializeApp();
        }};

        async function initializeApp() {{
            try {{
                // Simulation de l'initialisation (en vrai, cela appellerait le code Transcrypt)
                updateStatus("✅ Interface web initialisée");
                createActionButtons({cell_info["actions"]});
            }} catch (error) {{
                updateStatus("❌ Erreur d'initialisation: " + error.message);
            }}
        }}

        function createActionButtons(actions) {{
            const actionsDiv = document.getElementById('actions');
            actions.forEach(action => {{
                const button = document.createElement('button');
                button.textContent = action.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                button.onclick = () => executeAction(action);
                actionsDiv.appendChild(button);
            }});
        }}

        function updateStatus(message) {{
            document.getElementById('status').textContent = message;
        }}

        function updateOutput(content) {{
            document.getElementById('output').textContent = content;
        }}

        function simulateCell() {{
            updateStatus("🔄 Simulation en cours...");
            // Simulation de la réponse (en vrai, cela appellerait simulate_cell_web())
            setTimeout(() => {{
                const mockResult = {{
                    "nom": "{app_name}",
                    "actions": {json.dumps(cell_info["actions"])},
                    "champs": {json.dumps(cell_info["fields"])},
                    "status": "vivant (web)",
                    "temperature": 25,
                    "mouvement": "actif",
                    "reaction": "interface web simulée"
                }};
                updateOutput(JSON.stringify(mockResult, null, 2));
                updateStatus("✅ Simulation terminée");
            }}, 1000);
        }}

        function executeAction(action) {{
            updateStatus(`🔄 Exécution de ${{action}}...`);
            // Simulation de la réponse (en vrai, cela appellerait execute_action_web(action))
            setTimeout(() => {{
                const mockResponse = `Action ${{action}} exécutée avec succès dans l'interface web simulée`;
                updateOutput(mockResponse);
                updateStatus("✅ Action terminée");
            }}, 500);
        }}
    </script>
</body>
</html>'''
        return html

    def generate_pyinstaller_spec(self, cell_info):
        """Génère le fichier spec pour PyInstaller"""
        app_name = cell_info["name"]

        spec = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'sentence_transformers',
        'faiss',
        'transformers',
        'torch',
        'numpy',
        'tkinter',
        'json',
        'os',
        'sys',
        'threading'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Kibali_{app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emacs=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
        return spec


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python kibali.py <fichier.kib>                    # Exécuter une cellule")
        print("  python kibali.py animate \"<description>\"         # Générer et animer une scène 3D")
        print("  python kibali.py compile <fichier.kib> <platform> # Compiler vers une plateforme")
        print("Plateformes supportées: android, ios, web, desktop")
        sys.exit(1)

    if sys.argv[1] == "animate":
        if len(sys.argv) < 3:
            print("Usage: python kibali.py animate \"<description de la scène>\"")
            sys.exit(1)

        description = sys.argv[2]
        runtime = KibaliRuntime()
        result = runtime.animate_scene(description)
        print("Résultat de l'animation Kibali:")
        print(result)

    elif sys.argv[1] == "compile":
        if len(sys.argv) < 4:
            print("Usage: python kibali.py compile <fichier.kib> <platform>")
            print("Plateformes supportées: android, ios, web, desktop")
            sys.exit(1)

        cell_file = sys.argv[2]
        platform = sys.argv[3].lower()

        if platform not in ["android", "ios", "web", "desktop"]:
            print(f"Plateforme non supportée: {platform}")
            print("Plateformes supportées: android, ios, web, desktop")
            sys.exit(1)

        runtime = KibaliRuntime()
        result = runtime.compile_to_platform(cell_file, platform)

        if "error" in result:
            print(f"❌ Erreur: {result['error']}")
            sys.exit(1)
        else:
            print("✅ Compilation terminée avec succès!")
            print(f"📁 Répertoire de sortie: {result['output_dir']}")
            print(f"🏗️ Commande de build: {result['build_command']}")

    else:
        # Mode exécution normal
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Fichier {file_path} non trouvé")
            sys.exit(1)

        runtime = KibaliRuntime()
        output = runtime.run_program(file_path)
        print("Résultat de l'exécution Kibali:")
        print(output)

if __name__ == "__main__":
    main()