#!/usr/bin/env python3
"""
Agent Kibali intelligent utilisant Code Llama et Phi comme outils
Cet agent orchestre l'influence sur les cellules et objets de manière fluide
Version simplifiée sans LangChain pour commencer
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Nos modèles optimisés
from codellama_loader import OptimizedCodeLlamaLoader

# Import des classes existantes
from ecosystem_simulation import Environment, AutonomousCell, CellKnowledge

@dataclass
class CellInfluence:
    """Représente une influence sur une cellule"""
    cell_name: str
    cell_type: str
    modifications: Dict[str, Any]
    reasoning: str
    priority: int = 1

@dataclass
class ObjectInfluence:
    """Représente une influence sur un objet"""
    object_name: str
    cell_name: str
    modifications: Dict[str, Any]
    reasoning: str
    priority: int = 1

class CodeLlamaTool:
    """Outil pour Code Llama"""

    def __init__(self):
        self.loader = None

    def load_model(self):
        """Charge le modèle si pas déjà chargé"""
        if self.loader is None:
            self.loader = OptimizedCodeLlamaLoader()
            self.loader.load_model()
        return self.loader

    def analyze_cell(self, cell_type: str, current_params: Dict, env_data: Dict) -> Dict:
        """Analyse une cellule avec Code Llama"""
        try:
            loader = self.load_model()
            result = loader.analyze_and_modify_parameters(cell_type, current_params, env_data)
            return result
        except Exception as e:
            return {"error": f"Erreur Code Llama: {str(e)}"}

class PhiTool:
    """Outil pour Phi (fallback et analyse rapide)"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Charge le modèle Phi"""
        try:
            # Import dynamique pour éviter les erreurs si pas installé
            import sys
            if 'transformers' in sys.modules:
                from transformers import AutoTokenizer, AutoModelForCausalLM
            else:
                return

            model_path = Path("./ia/phi-1_5")
            if model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            print(f"Phi non disponible: {e}")

    def analyze_quick(self, prompt: str) -> Dict:
        """Analyse rapide avec Phi"""
        if not self.model or not self.tokenizer:
            return {"error": "Modèle Phi non disponible"}

        try:
            # Import dynamique
            import sys
            if 'torch' in sys.modules:
                import torch
            else:
                return {"error": "PyTorch non disponible"}

            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Essayer de parser du JSON
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except:
                pass

            return {"response": response[:200]}

        except Exception as e:
            return {"error": f"Erreur Phi: {str(e)}"}

class EnvironmentInfluenceTool:
    """Outil pour influencer l'environnement"""

    def __init__(self, environment: Environment):
        self.environment = environment

    def modify_environment(self, modifications: Dict[str, Any]) -> str:
        """Applique des modifications à l'environnement"""
        try:
            mod_list = []
            for prop, change in modifications.items():
                if isinstance(change, dict) and "type" in change and "value" in change:
                    mod_list.append({
                        "property": prop,
                        "type": change["type"],
                        "value": change["value"]
                    })

            self.environment.modify(mod_list)
            return f"Environnement modifié: {len(mod_list)} changements appliqués"
        except Exception as e:
            return f"Erreur modification environnement: {str(e)}"

class KibaliIntelligentAgent:
    """Agent Kibali intelligent utilisant Code Llama et Phi comme outils"""

    def __init__(self, environment: Environment, cells: List[AutonomousCell]):
        self.environment = environment
        self.cells = {cell.name: cell for cell in cells}

        # Outils disponibles
        self.code_llama_tool = CodeLlamaTool()
        self.phi_tool = PhiTool()
        self.env_tool = EnvironmentInfluenceTool(environment)

        # État de l'agent
        self.influence_history = []
        self.knowledge_base = self._load_knowledge_base()

        print("🤖 Agent KIBALI initialisé avec succès!")
        print(f"📊 Cellules surveillées: {list(self.cells.keys())}")
        print(f"🛠️ Outils disponibles: Code Llama, Phi, Influence Environnementale")

    def _load_knowledge_base(self):
        """Charge la base de connaissances générale"""
        kb = {}
        for cell in self.cells.values():
            kb[cell.cell_type] = cell.knowledge_base
        return kb

    def _analyze_global_situation(self) -> Dict[str, Any]:
        """Analyse la situation globale de l'écosystème"""
        env_state = self.environment.get_state()

        cell_states = {}
        for name, cell in self.cells.items():
            cell_states[name] = {
                "type": cell.cell_type,
                "internal_states": cell.internal_states,
                "physical_objects": cell.physical_objects,
                "last_adaptation": cell.adaptation_history[-1] if cell.adaptation_history else None
            }

        return {
            "environment": env_state,
            "cells": cell_states,
            "timestamp": time.time()
        }

    def _generate_influence_strategy(self, situation: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Génère une stratégie d'influence intelligente"""

        # Analyser les problèmes potentiels
        issues = self._identify_issues(situation)

        # Générer des influences correctives
        influences = []

        for issue in issues:
            if issue["type"] == "stress_thermique":
                # Utiliser Code Llama pour analyser l'arbre
                tree_cell = next((cell for cell in self.cells.values() if cell.cell_type == "Arbre"), None)
                if tree_cell:
                    analysis = self.code_llama_tool.analyze_cell(
                        "Arbre",
                        tree_cell.internal_states,
                        situation["environment"]
                    )

                    if "error" not in analysis:
                        influences.append(CellInfluence(
                            cell_name=tree_cell.name,
                            cell_type="Arbre",
                            modifications=analysis,
                            reasoning="Adaptation aux conditions thermiques via Code Llama",
                            priority=3
                        ))

            elif issue["type"] == "déséquilibre_hydrique":
                # Influencer l'environnement pour réguler l'humidité
                influences.append(ObjectInfluence(
                    object_name="environnement",
                    cell_name="global",
                    modifications={
                        "humidity": {"type": "add", "value": 5.0}
                    },
                    reasoning="Régulation hydrique environnementale",
                    priority=2
                ))

        return {
            "issues": issues,
            "influences": influences,
            "strategy": "Approche proactive et coordonnée"
        }

    def _identify_issues(self, situation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifie les problèmes nécessitant une intervention"""
        issues = []

        env = situation["environment"]

        # Stress thermique
        if env["temperature"] > 28:
            issues.append({
                "type": "stress_thermique",
                "severity": "high",
                "description": f"Température élevée ({env['temperature']}°C)"
            })

        # Déséquilibre hydrique
        if env["humidity"] < 40 or env["soil_moisture"] < 30:
            issues.append({
                "type": "déséquilibre_hydrique",
                "severity": "medium",
                "description": "Conditions sèches détectées"
            })

        # Stress lumineux
        if env["light_level"] > 85:
            issues.append({
                "type": "stress_lumineux",
                "severity": "medium",
                "description": "Luminosité excessive"
            })

        return issues

    def _apply_influences(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Applique les influences planifiées"""
        applied_influences = []

        for influence in strategy["influences"]:
            if isinstance(influence, CellInfluence):
                cell = self.cells.get(influence.cell_name)
                if cell:
                    # Appliquer les modifications à la cellule
                    if "internal_states" in influence.modifications:
                        cell.apply_internal_adaptations(influence.modifications)

                    if "physical_objects" in influence.modifications:
                        cell.modify_physical_objects(influence.modifications)

                    applied_influences.append({
                        "type": "cell_influence",
                        "target": influence.cell_name,
                        "modifications": influence.modifications,
                        "reasoning": influence.reasoning
                    })

            elif isinstance(influence, ObjectInfluence):
                if influence.object_name == "environnement":
                    # Modifier l'environnement
                    env_mods = []
                    for prop, change in influence.modifications.items():
                        env_mods.append({
                            "property": prop,
                            "type": change["type"],
                            "value": change["value"]
                        })

                    self.environment.modify(env_mods)

                    applied_influences.append({
                        "type": "environment_influence",
                        "modifications": influence.modifications,
                        "reasoning": influence.reasoning
                    })

        return applied_influences

    def _generate_report(self, influences: List[Dict[str, Any]]) -> str:
        """Génère un rapport des actions effectuées"""
        report = "🤖 RAPPORT DE L'AGENT KIBALI 🤖\n\n"

        if influences:
            report += f"✅ {len(influences)} influences appliquées:\n\n"

            for i, influence in enumerate(influences, 1):
                report += f"{i}. {influence['type'].upper()}\n"
                if 'target' in influence:
                    report += f"   Cible: {influence['target']}\n"
                report += f"   Modifications: {json.dumps(influence['modifications'], indent=2, ensure_ascii=False)}\n"
                report += f"   Raison: {influence['reasoning']}\n\n"
        else:
            report += "🔍 Aucune influence nécessaire - écosystème équilibré\n"

        return report

    def influence_ecosystem(self, user_instruction: str = "") -> str:
        """Point d'entrée principal pour influencer l'écosystème"""

        # Analyser la situation actuelle
        situation = self._analyze_global_situation()

        # Générer une stratégie d'influence
        strategy = self._generate_influence_strategy(situation, user_instruction)

        # Appliquer les influences
        influences = self._apply_influences(strategy)

        # Retourner le rapport
        result = self._generate_report(influences)

        # Enregistrer dans l'historique
        self.influence_history.append({
            "timestamp": time.time(),
            "instruction": user_instruction,
            "result": result
        })

        return result

    def get_agent_status(self) -> Dict[str, Any]:
        """Retourne l'état actuel de l'agent"""
        return {
            "name": "KIBALI Intelligent Agent",
            "tools_available": ["Code Llama", "Phi", "Environment Influence"],
            "cells_monitored": list(self.cells.keys()),
            "influences_applied": len(self.influence_history),
            "last_activity": self.influence_history[-1] if self.influence_history else None
        }

# Fonction principale pour tester l'agent
def test_kibali_agent():
    """Test de l'agent Kibali"""

    print("🚀 Initialisation de l'Agent KIBALI...\n")

    # Créer l'écosystème
    environment = Environment()

    # Créer les cellules
    cells = [
        AutonomousCell("Chêne Millénaire", "Arbre", "arbres_biology", environment),
        AutonomousCell("Système Climatique", "Climat", "climat_science", environment),
        AutonomousCell("Écureuil Agile", "Ecureuil", "ecureuil_behavior", environment),
        AutonomousCell("Rose Sauvage", "Fleur", "fleur_biology", environment)
    ]

    # Créer l'agent Kibali
    kibali_agent = KibaliIntelligentAgent(environment, cells)

    print("📊 État initial de l'agent:")
    print(json.dumps(kibali_agent.get_agent_status(), indent=2, ensure_ascii=False))
    print("\n" + "="*60 + "\n")

    # Simuler quelques cycles avec intervention de l'agent
    for cycle in range(3):
        print(f"🔄 Cycle {cycle + 1}/3 - Agent KIBALI en action\n")

        # Modifier l'environnement pour créer des défis
        if cycle == 0:
            environment.temperature += 15  # Stress thermique
            environment.events.append("canicule")
            print("🌡️ Agent KIBALI détecte: Canicule soudaine!")
        elif cycle == 1:
            environment.humidity -= 30  # Sécheresse
            environment.events.append("sécheresse")
            print("🏜️ Agent KIBALI détecte: Période de sécheresse!")
        elif cycle == 2:
            environment.light_level += 20  # Forte luminosité
            environment.events.append("ensoleillement_intense")
            print("☀️ Agent KIBALI détecte: Ensoleillement intense!")

        # L'agent analyse et influence
        influence_report = kibali_agent.influence_ecosystem(
            f"Cycle {cycle + 1}: Optimise l'écosystème face aux conditions actuelles"
        )

        print(influence_report)

        # Les cellules s'adaptent normalement
        print("🦠 Les cellules s'adaptent également:")
        for cell in cells:
            print(f"   • {cell.name} ({cell.cell_type})")
            cell.adapt_autonomously()

        print(f"\n📊 État de l'environnement après cycle {cycle + 1}:")
        print(json.dumps(environment.get_state(), indent=2))
        print("\n" + "="*60 + "\n")

        time.sleep(1)

    # Rapport final
    print("🏁 RAPPORT FINAL DE L'AGENT KIBALI")
    print("="*60)
    print(json.dumps(kibali_agent.get_agent_status(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_kibali_agent()