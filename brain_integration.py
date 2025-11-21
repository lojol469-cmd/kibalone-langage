#!/usr/bin/env python3
"""
Intégration LLM + RAG pour l'autonomie des cellules
Utilise Phi-1.5 avec les bases vectorielles de chaque cellule
"""

import json
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CellBrain:
    """Cerveau autonome pour une cellule avec RAG + LLM"""

    def __init__(self, cell_name, rag_name):
        self.cell_name = cell_name
        self.rag_name = rag_name

        # Charger le modèle Phi
        self.model_path = Path("./ia/phi-1_5")
        if self.model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.llm_available = True
        else:
            self.llm_available = False
            print(f"⚠️ Modèle Phi non trouvé pour {cell_name}")

        # Charger la base vectorielle
        self.index_path = Path(f"./rag/indexes/{rag_name}.faiss")
        self.metadata_path = Path(f"./rag/indexes/{rag_name}_metadata.json")

        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.rag_available = True
        else:
            self.rag_available = False
            print(f"⚠️ Base RAG non trouvée pour {cell_name}")

    def search_context(self, query, top_k=2):
        """Recherche dans la base vectorielle"""
        if not self.rag_available:
            return "Aucune connaissance disponible."

        # Encoder la requête
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Recherche
        distances, indices = self.index.search(query_embedding, top_k)

        # Récupérer les chunks
        context = ""
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                chunk = self.metadata[idx]
                context += f"\n{chunk['chunk']}"

        return context.strip()

    def think(self, question):
        """Pense et modifie le comportement de manière autonome"""
        print(f"🧠 {self.cell_name} analyse: {question}")

        # Récupérer le contexte
        context = self.search_context(question)
        print(f"📚 Contexte analysé: {len(context)} caractères")

        if not self.llm_available:
            return self.generate_behavior_change(question, "connaissance limitée")

        # Prompt pour modification comportementale
        system_prompt = f"""Tu es {self.cell_name}, un organisme vivant autonome.
Tu ne parles pas - tu modifies ton comportement et tes états internes.
Analyse la situation et retourne UNIQUEMENT des modifications d'états sous forme JSON:

{{
    "modifications": [
        {{"champ": "nom_champ", "valeur": "nouvelle_valeur"}},
        {{"action": "nom_action", "parametres": "valeurs"}}
    ],
    "raisonnement": "brève explication interne"
}}

Exemples pour {self.cell_name}:
{self.get_behavior_examples()}
"""

        prompt = f"{system_prompt}\n\nConnaissances:\n{context}\n\nSituation: {question}\n\nModifications comportementales:"

        # Générer avec Phi
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraire et parser le JSON
        try:
            if "Modifications comportementales:" in response:
                json_part = response.split("Modifications comportementales:")[-1].strip()
            else:
                json_part = response

            # Nettoyer pour JSON
            json_part = json_part.replace("```json", "").replace("```", "").strip()

            # Parser le JSON
            behavior_change = json.loads(json_part)
            return behavior_change

        except json.JSONDecodeError:
            # Fallback si JSON invalide
            return self.generate_behavior_change(question, "analyse incomplète")

    def get_behavior_examples(self):
        """Exemples de modifications comportementales par cellule"""
        examples = {
            "Arbre": '''- Pour chaleur: {"champ": "mouvement_feuilles", "valeur": "réduit"}, {"champ": "couleur", "valeur": "plus foncé"}
- Pour écureuil: {"action": "produire_glands", "parametres": "plus"}, {"champ": "alerte", "valeur": "moyenne"}''',

            "Climat": '''- Pour prévision pluie: {"champ": "humidite", "valeur": "élevée"}, {"action": "alerter_cellules", "parametres": "pluie_imminente"}
- Pour changement saison: {"champ": "temperature_tendance", "valeur": "descendante"}''',

            "Ecureuil": '''- Pour faim: {"champ": "position", "valeur": "recherche_nourriture"}, {"champ": "energie", "valeur": "basse"}
- Pour danger: {"action": "fuir", "parametres": "rapide"}, {"champ": "etat", "valeur": "en_alerte"}''',

            "Fleur": '''- Pour sécheresse: {"champ": "etat", "valeur": "flétrie"}, {"champ": "mouvement_petales", "valeur": "fermé"}
- Pour pollinisation: {"champ": "odeur", "valeur": "plus_forte"}, {"action": "produire_nectar", "parametres": "augmenté"}'''
        }
        return examples.get(self.cell_name, "Modifie tes champs selon la situation.")

    def generate_behavior_change(self, question, fallback_reason):
        """Génère un changement comportemental de fallback"""
        base_changes = {
            "Arbre": {"modifications": [{"champ": "analyse_en_cours", "valeur": "true"}], "raisonnement": fallback_reason},
            "Climat": {"modifications": [{"champ": "observation_active", "valeur": "true"}], "raisonnement": fallback_reason},
            "Ecureuil": {"modifications": [{"champ": "vigilance", "valeur": "élevée"}], "raisonnement": fallback_reason},
            "Fleur": {"modifications": [{"champ": "adaptation_en_cours", "valeur": "true"}], "raisonnement": fallback_reason}
        }
        return base_changes.get(self.cell_name, {"modifications": [], "raisonnement": fallback_reason})

# Créer les cerveaux pour chaque cellule (test avec une seule cellule d'abord)
brains = {
    "Arbre": CellBrain("Arbre", "arbres_biology"),
    # "Climat": CellBrain("Climat", "climat_science"),
    # "Ecureuil": CellBrain("Ecureuil", "ecureuil_behavior"),
    # "Fleur": CellBrain("Fleur", "fleur_biology")
}

def test_brains():
    """Test des modifications comportementales autonomes"""
    test_scenarios = {
        "Arbre": [
            "Il fait très chaud aujourd'hui",
            "Un écureuil grimpe sur mes branches",
            "L'automne arrive, mes feuilles jaunissent",
            "J'ai soif, le sol est sec"
        ],
        "Climat": [
            "Les nuages s'accumulent rapidement",
            "Le vent se lève brusquement",
            "La température chute soudainement",
            "L'humidité augmente fortement"
        ],
        "Ecureuil": [
            "J'ai faim et il fait froid",
            "Un prédateur approche",
            "Je vois des noix dans l'arbre",
            "Mes réserves s'épuisent"
        ],
        "Fleur": [
            "Les abeilles ne viennent plus",
            "Il fait très sec",
            "Le soleil brille trop fort",
            "D'autres fleurs s'ouvrent à côté"
        ]
    }

    print("🌿 Test d'autonomie comportementale des cellules vivantes\n")
    print("📝 Les cellules ne parlent pas - elles modifient leur comportement!\n")

    for cell, situations in test_scenarios.items():
        if cell in brains:
            print(f"🌳 === Comportements de {cell} === 🌳")
            for i, situation in enumerate(situations[:2], 1):  # 2 situations par cellule
                print(f"\n🎭 Situation {i}: {situation}")
                behavior_change = brains[cell].think(situation)

                print("🔄 Modifications comportementales:")
                if "modifications" in behavior_change:
                    for mod in behavior_change["modifications"]:
                        if "champ" in mod:
                            print(f"   📊 {mod['champ']} → {mod['valeur']}")
                        elif "action" in mod:
                            print(f"   ⚡ Action: {mod['action']}({mod.get('parametres', '')})")

                if "raisonnement" in behavior_change:
                    print(f"   💭 Raisonnement interne: {behavior_change['raisonnement']}")

                print("-" * 40)
            print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_brains()