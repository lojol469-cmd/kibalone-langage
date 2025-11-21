#!/usr/bin/env python3
"""
Kibali Runtime - Framework pour exécuter des programmes Kibali
Langage organique pour nano-IA vivantes
"""

import json
import os
import sys
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

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
        self.tokenizer = None
        self.model = None
        self.tools = self.load_tools()
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
        if self.model is None:
            return "Cerveau non disponible"

        # Enrichir la requête avec des connaissances pertinentes
        knowledge_context = self.get_relevant_knowledge(query)

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

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=200, num_return_sequences=1, temperature=0.8, do_sample=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Nettoyer la réponse
        response = response.replace(prompt, "").strip()
        return response

    def get_relevant_knowledge(self, query, top_k=2):
        """Extrait les connaissances pertinentes de la base RAG"""
        if self.rag_system.index is None:
            return "Aucune connaissance disponible"

        try:
            results = self.rag_system.search(query, top_k=top_k)
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
        """Charge le modèle LLM pour le cerveau (Phi-1.5 local ou Mistral)"""
        if self.model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                # Essayer d'abord Phi-1.5 depuis le dossier local models/
                phi_local_path = os.path.join(os.path.dirname(__file__), "models", "phi-1_5")
                if os.path.exists(phi_local_path):
                    try:
                        print("🧠 Chargement de Phi-1.5 depuis models/phi-1_5/ (local)...")
                        self.tokenizer = AutoTokenizer.from_pretrained(phi_local_path)
                        self.model = AutoModelForCausalLM.from_pretrained(phi_local_path)
                        print("✅ Cerveau Phi-1.5 chargé depuis models/phi-1_5/ et prêt à utiliser les connaissances RAG")
                        return
                    except Exception as e_phi:
                        print(f"⚠️ Erreur chargement Phi-1.5 local: {e_phi}")

                # Essayer Phi-1.5 depuis HuggingFace
                try:
                    print("🧠 Chargement de Phi-1.5 depuis HuggingFace...")
                    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
                    self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
                    print("✅ Cerveau Phi-1.5 chargé depuis HuggingFace et prêt à utiliser les connaissances RAG")
                    return
                except Exception as e_phi:
                    print(f"⚠️ Phi-1.5 non disponible: {e_phi}")

                # Sinon essayer Mistral local
                mistral_path = os.path.join(os.path.dirname(__file__), "ia", "mistral-7b")
                if os.path.exists(mistral_path):
                    # Vérifier si le modèle est complet
                    if os.path.exists(os.path.join(mistral_path, "pytorch_model.bin")) or os.path.exists(os.path.join(mistral_path, "model.safetensors")):
                        self.tokenizer = AutoTokenizer.from_pretrained(mistral_path)
                        self.model = AutoModelForCausalLM.from_pretrained(mistral_path)
                        print("✅ Cerveau Mistral 7B chargé depuis ia/")
                        return

                print("❌ Aucun modèle LLM disponible")

            except Exception as e:
                print(f"❌ Erreur chargement cerveau: {e}")
                print("💡 Le système fonctionnera en mode dégradé sans LLM")

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

    def run_program(self, main_file):
        """Exécute un programme Kibali"""
        cell_info = self.load_cell(main_file)
        result = self.simulate_cell(cell_info["name"])
        return json.dumps(result, indent=2, ensure_ascii=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: python kibali.py <fichier.kib>")
        sys.exit(1)

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