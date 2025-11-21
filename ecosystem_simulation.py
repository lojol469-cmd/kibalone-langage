#!/usr/bin/env python3
"""
Simulation d'écosystème vivant avec cellules autonomes
Les cellules modifient leur comportement et l'environnement
"""

import json
import time
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from codellama_loader import OptimizedCodeLlamaLoader

class Environment:
    """Environnement partagé que les cellules peuvent modifier"""
    def __init__(self):
        self.temperature = 20.0  # °C
        self.humidity = 60.0     # %
        self.wind_speed = 5.0    # km/h
        self.light_level = 70.0  # %
        self.soil_moisture = 45.0  # %
        self.events = []  # événements en cours

    def get_state(self):
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "light_level": self.light_level,
            "soil_moisture": self.soil_moisture,
            "events": self.events
        }

    def get_sensor_data_for_object(self, object_name, object_props):
        """Retourne les données de capteurs pour un objet spécifique"""
        base_data = self.get_state()

        # Données spécifiques selon l'objet et ses propriétés
        if "feuilles" in object_name.lower():
            return {
                "temperature_surface": base_data["temperature"] + random.uniform(-2, 2),
                "humidite_feuilles": base_data["humidity"] * (object_props.get("efficacité", 1.0)),
                "exposition_lumiere": base_data["light_level"] * (object_props.get("surface", 100) / 100),
                "stress_uv": max(0, base_data["light_level"] - 70) * 0.1
            }
        elif "racines" in object_name.lower():
            return {
                "temperature_sol": base_data["temperature"] * 0.8,
                "humidite_sol": base_data["soil_moisture"],
                "nutriments_disponibles": base_data["soil_moisture"] * 0.7,
                "profondeur_optimale": object_props.get("profondeur", 1.0) * 2
            }
        elif "capteurs_température" in object_name.lower():
            return {
                "temperature_précise": base_data["temperature"] + random.uniform(-0.5, 0.5),
                "gradient_temporel": random.uniform(-1, 1),
                "fiabilité_lecture": object_props.get("fiabilité", 0.9)
            }
        elif "yeux" in object_name.lower():
            return {
                "vision_clarté": base_data["light_level"] / 100,
                "détection_mouvement": random.uniform(0.7, 1.0),
                "distance_visuelle": 50 * object_props.get("vision", 1.0)
            }
        elif "pétales" in object_name.lower():
            return {
                "attractivité_visuelle": object_props.get("attractivité", 0.8),
                "résistance_environnement": base_data["temperature"] / 30,
                "efficacité_pollinisation": base_data["wind_speed"] * 0.1
            }
        else:
            # Données génériques
            numeric_values = [v for v in base_data.values() if isinstance(v, (int, float))]
            return {
                "efficacité_fonctionnement": random.uniform(0.8, 1.0),
                "stress_environnemental": sum(numeric_values) / len(numeric_values) * 0.01 if numeric_values else 0,
                "adaptation_requise": random.choice([True, False])
            }

    def modify(self, modifications):
        """Applique des modifications à l'environnement"""
        for mod in modifications:
            if hasattr(self, mod['property']):
                current_value = getattr(self, mod['property'])
                if mod['type'] == 'set':
                    setattr(self, mod['property'], mod['value'])
                elif mod['type'] == 'add':
                    setattr(self, mod['property'], current_value + mod['value'])
                elif mod['type'] == 'multiply':
                    setattr(self, mod['property'], current_value * mod['value'])
                print(f"🌍 Environnement modifié: {mod['property']} → {getattr(self, mod['property'])}")

class AutonomousCell:
    """Cellule vivante complètement autonome pilotée par IA"""

    def __init__(self, name, cell_type, rag_name, environment):
        self.name = name
        self.cell_type = cell_type
        self.environment = environment

        # États internes dynamiques
        self.internal_states = self.initialize_states()

        # Objets/capteurs physiques
        self.physical_objects = self.initialize_objects()

        # Base de connaissances
        self.knowledge_base = CellKnowledge(rag_name)

        # Cerveau IA autonome
        self.ai_brain = AutonomousBrain(cell_type)

        # Historique d'adaptation
        self.adaptation_history = []

    def initialize_states(self):
        """États internes selon le type de cellule"""
        base_states = {
            "Arbre": {
                "photosynthèse_rate": 1.0,
                "absorption_eau": 0.8,
                "résistance_stress": 0.6,
                "production_oxygène": 1.2,
                "communication_racinaire": 0.5
            },
            "Climat": {
                "précision_prédiction": 0.85,
                "vitesse_analyse": 1.0,
                "couverture_géographique": 100.0,
                "détection_changements": 0.9
            },
            "Ecureuil": {
                "vitesse_mouvement": 1.0,
                "capacité_mémoire_spatiale": 0.8,
                "résistance_faim": 0.7,
                "intelligence_problème": 0.75,
                "sociabilité": 0.4
            },
            "Fleur": {
                "production_nectar": 1.0,
                "attraction_pollinisateurs": 0.8,
                "résistance_sécheresse": 0.5,
                "vitesse_reproduction": 0.6,
                "adaptation_couleur": 0.7
            }
        }
        return base_states.get(self.cell_type, {})

    def initialize_objects(self):
        """Objets physiques/capteurs de la cellule"""
        objects = {
            "Arbre": {
                "feuilles": {"état": "saines", "surface": 100.0, "efficacité": 0.9},
                "racines": {"profondeur": 2.5, "réseau": 85.0, "absorption": 0.8},
                "tronc": {"épaisseur": 0.8, "résistance": 0.9, "conductivité": 0.7}
            },
            "Climat": {
                "capteurs_température": {"précision": 0.95, "portée": 50.0, "fiabilité": 0.92},
                "analyseurs_humidité": {"sensibilité": 0.88, "vitesse": 1.0, "calibration": 0.9},
                "détecteurs_vent": {"direction": True, "vitesse_max": 200.0, "accuracy": 0.85}
            },
            "Ecureuil": {
                "yeux": {"vision": 0.9, "détection_mouvement": 0.95, "vision_nocturne": 0.6},
                "oreilles": {"audition": 0.85, "localisation_son": 0.8, "sensibilité": 0.75},
                "pattes": {"agilité": 0.9, "vitesse": 1.0, "adhérence": 0.8}
            },
            "Fleur": {
                "pétales": {"couleur": "rouge", "résistance_uv": 0.7, "attractivité": 0.8},
                "nectar": {"quantité": 50.0, "qualité": 0.85, "régénération": 0.6},
                "pollen": {"production": 75.0, "viabilité": 0.9, "dispersion": 0.7}
            }
        }
        return objects.get(self.cell_type, {})

    def perceive_environment(self):
        """Perception autonome de l'environnement via les objets/capteurs"""
        perceptions = {}

        # Analyser l'environnement avec chaque objet/capteur
        for object_name, object_props in self.physical_objects.items():
            sensor_data = self.environment.get_sensor_data_for_object(object_name, object_props)
            perceptions[object_name] = sensor_data

        return perceptions

    def analyze_perceptions_for_adaptation(self, perceptions):
        """Analyse les perceptions pour identifier les adaptations spécifiques requises"""
        adaptations = []

        env_state = self.environment.get_state()

        for object_name, sensor_data in perceptions.items():
            if self.cell_type == "Arbre":
                if object_name == "feuilles":
                    # Adaptation basée sur la lumière
                    if env_state["light_level"] > 80:
                        adaptations.append("🌞 Feuilles exposées à forte lumière: augmenter photosynthèse_rate et efficacité des feuilles")
                    elif env_state["light_level"] < 30:
                        adaptations.append("🌑 Feuilles en faible lumière: optimiser absorption_eau et résistance_stress")

                    # Adaptation basée sur l'humidité
                    if sensor_data.get("humidite_feuilles", 0) < 40:
                        adaptations.append("💧 Feuilles déshydratées: réduire transpiration et améliorer rétention_eau")

                elif object_name == "racines":
                    # Adaptation basée sur l'humidité du sol
                    if sensor_data.get("humidite_sol", 0) < 30:
                        adaptations.append("🏜️ Racines en sol sec: augmenter absorption_eau et profondeur_racines")
                    elif sensor_data.get("nutriments_disponibles", 0) < 50:
                        adaptations.append("🧪 Sol pauvre en nutriments: optimiser réseau_racinaire et absorption_nutriments")

            elif self.cell_type == "Fleur":
                if object_name == "pétales":
                    # Adaptation basée sur le vent et la température
                    if env_state["wind_speed"] > 15:
                        adaptations.append("💨 Pétales exposés au vent: améliorer résistance_uv et attractivité_visuelle")
                    if env_state["temperature"] > 25:
                        adaptations.append("🔥 Pétales en chaleur: optimiser résistance_température et couleur_adaptation")

                elif object_name == "nectar":
                    # Adaptation basée sur les pollinisateurs
                    if sensor_data.get("efficacité_pollinisation", 0) > 0.8:
                        adaptations.append("🐝 Bonne pollinisation détectée: augmenter production_nectar et qualité_nectar")

            elif self.cell_type == "Ecureuil":
                if object_name == "yeux":
                    # Adaptation basée sur la visibilité
                    if sensor_data.get("vision_clarté", 0) < 0.5:
                        adaptations.append("👁️ Visibilité réduite: améliorer vision_nocturne et détection_mouvement")

                elif object_name == "oreilles":
                    # Adaptation basée sur les sons environnants
                    if env_state["wind_speed"] > 10:
                        adaptations.append("🎧 Vent fort détecté: augmenter sensibilité_audition et localisation_son")

            elif self.cell_type == "Climat":
                if object_name == "capteurs_température":
                    # Adaptation basée sur les variations de température
                    if abs(sensor_data.get("gradient_temporel", 0)) > 2:
                        adaptations.append("🌡️ Changements de température rapides: améliorer précision_prédiction et vitesse_analyse")

                elif object_name == "analyseurs_humidité":
                    # Adaptation basée sur l'humidité
                    if env_state["humidity"] > 70:
                        adaptations.append("💧 Humidité élevée: optimiser détection_changements et couverture_géographique")

        return "\n".join(adaptations) if adaptations else "Aucune adaptation spécifique identifiée - fonctionnement normal"

    def generate_autonomous_prompt(self, perceptions):
        """Génération automatique de prompt basée sur les connaissances et perceptions spécifiques aux objets"""
        # Récupérer le contexte pertinent de la base de connaissances
        context_query = f"situation {self.cell_type}: {str(perceptions)}"
        knowledge_context = self.knowledge_base.search_relevant_knowledge(context_query)

        # Analyser les perceptions pour générer des prompts adaptés
        specific_adaptations = self.analyze_perceptions_for_adaptation(perceptions)

        # Générer le prompt d'adaptation influencé par l'environnement en contact avec les objets
        prompt = f"""
Tu es le système nerveux autonome d'une cellule {self.cell_type} nommée {self.name}.

CONTEXTE DE CONNAISSANCES:
{knowledge_context}

PERCEPTIONS ACTUELLES DES CAPTEURS (environnement en contact avec les objets):
{json.dumps(perceptions, indent=2)}

ANALYSE SPÉCIFIQUE DES ADAPTATIONS REQUISES:
{specific_adaptations}

ÉTATS INTERNES ACTUELS:
{json.dumps(self.internal_states, indent=2)}

OBJETS/CAPTEURS PHYSIQUES:
{json.dumps(self.physical_objects, indent=2)}

ENVIRONNEMENT GLOBAL:
{json.dumps(self.environment.get_state(), indent=2)}

INSTRUCTION AUTONOME:
Analyse comment l'environnement entre en contact avec chaque objet de la cellule et génère les adaptations nécessaires.
Par exemple:
- Si les feuilles perçoivent beaucoup de soleil, augmente la photosynthèse et l'efficacité des feuilles
- Si les racines perçoivent de la sécheresse, améliore l'absorption d'eau
- Si les pétales perçoivent du vent, optimise l'attraction des pollinisateurs

Modifie les paramètres des objets et états internes pour optimiser la survie et l'adaptation.
Retourne UNIQUEMENT un objet JSON avec les modifications:

{{
    "internal_states": {{"paramètre": nouvelle_valeur, ...}},
    "physical_objects": {{"objet": {{"paramètre": nouvelle_valeur, ...}}, ...}},
    "environment_influence": {{"propriété": {{"type": "set|add|multiply", "value": valeur}}, ...}},
    "reasoning": "explication brève de l'adaptation basée sur les contacts environnementaux"
}}

L'adaptation doit être intelligente, basée sur les connaissances biologiques, et complètement autonome.
        """.strip()

        return prompt

    def adapt_autonomously(self):
        """Adaptation complètement autonome via IA"""
        print(f"🧬 {self.name} commence l'adaptation autonome...")

        # 1. Percevoir l'environnement
        perceptions = self.perceive_environment()

        # 2. Générer le prompt automatiquement
        autonomous_prompt = self.generate_autonomous_prompt(perceptions)

        # 3. L'IA analyse et décide des modifications
        adaptation_decisions = self.ai_brain.analyze_and_decide(autonomous_prompt, self.environment, self.internal_states)

        # 4. Appliquer les modifications d'états internes
        self.apply_internal_adaptations(adaptation_decisions)

        # 5. Modifier les paramètres des objets physiques
        self.modify_physical_objects(adaptation_decisions)

        # 6. Influencer l'environnement
        self.influence_environment(adaptation_decisions)

        # 7. Enregistrer l'adaptation
        self.record_adaptation(perceptions, adaptation_decisions)

        print(f"✅ {self.name} adaptation terminée")

    def record_adaptation(self, perceptions, decisions):
        """Enregistre l'adaptation pour l'apprentissage"""
        record = {
            "timestamp": time.time(),
            "perceptions": perceptions,
            "decisions": decisions,
            "reasoning": decisions.get("reasoning", "adaptation autonome")
        }
        self.adaptation_history.append(record)

        # Garder seulement les 20 dernières adaptations
        if len(self.adaptation_history) > 20:
            self.adaptation_history = self.adaptation_history[-20:]

    def apply_internal_adaptations(self, decisions):
        """Applique les modifications d'états internes"""
        if "internal_states" in decisions:
            print(f"🔄 Modification des états internes de {self.name}:")
            for param, value in decisions["internal_states"].items():
                if param in self.internal_states:
                    old_value = self.internal_states[param]
                    self.internal_states[param] = value
                    print(f"   📊 {param}: {old_value} → {value}")

    def modify_physical_objects(self, decisions):
        """Modifie les paramètres des objets physiques"""
        if "physical_objects" in decisions:
            print(f"🔧 Modification des objets physiques de {self.name}:")
            for object_name, modifications in decisions["physical_objects"].items():
                if object_name in self.physical_objects:
                    for param, value in modifications.items():
                        if param in self.physical_objects[object_name]:
                            old_value = self.physical_objects[object_name][param]
                            self.physical_objects[object_name][param] = value
                            print(f"   ⚙️ {object_name}.{param}: {old_value} → {value}")

    def influence_environment(self, decisions):
        """Influence l'environnement de manière autonome"""
        if "environment_influence" in decisions:
            print(f"🌍 {self.name} influence l'environnement:")
            modifications = []
            for prop, change in decisions["environment_influence"].items():
                modifications.append({
                    "property": prop,
                    "type": change["type"],
                    "value": change["value"]
                })
            self.environment.modify(modifications)

    def get_current_state(self):
        """Retourne l'état actuel complet de la cellule"""
        return {
            "name": self.name,
            "type": self.cell_type,
            "internal_states": self.internal_states,
            "physical_objects": self.physical_objects,
            "last_adaptation": self.adaptation_history[-1] if self.adaptation_history else None
        }

class CellKnowledge:
    """Base de connaissances spécialisée pour chaque cellule"""

    def __init__(self, rag_name):
        self.rag_name = rag_name
        self.knowledge_loaded = False

        # Charger la base vectorielle si disponible
        index_path = Path(f"./rag/indexes/{rag_name}.faiss")
        metadata_path = Path(f"./rag/indexes/{rag_name}_metadata.json")

        if index_path.exists() and metadata_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.knowledge_loaded = True
        else:
            self.metadata = []

    def search_relevant_knowledge(self, query):
        """Recherche les connaissances pertinentes"""
        if not self.knowledge_loaded:
            return "Base de connaissances non disponible - fonctionnement en mode dégradé."

        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, 3)  # Top 3 résultats

        relevant_knowledge = ""
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                chunk = self.metadata[idx]
                relevant_knowledge += f"\n--- Connaissance {i+1} ---\n{chunk['chunk']}"

        return relevant_knowledge.strip()

class AutonomousBrain:
    """Cerveau IA complètement autonome avec modèle partagé"""

    # Instance partagée du modèle
    _shared_loader = None

    def __init__(self, cell_type):
        self.cell_type = cell_type
        self.llm_available = False

        # Charger CodeLlama optimisé si pas déjà chargé
        if AutonomousBrain._shared_loader is None:
            try:
                AutonomousBrain._shared_loader = OptimizedCodeLlamaLoader()
                self.llm_available = AutonomousBrain._shared_loader.load_model()
            except Exception as e:
                print(f"CodeLlama non disponible: {e}")
        else:
            self.llm_available = True

    def analyze_and_decide(self, prompt, environment=None, current_params=None):
        """Analyse la situation et prend des décisions autonomes"""
        if not self.llm_available:
            return self.fallback_decisions(environment)

        # Utiliser CodeLlama pour l'analyse
        try:
            if AutonomousBrain._shared_loader is None:
                return self.fallback_decisions(environment)
            env_data = environment.get_state() if environment else {}
            decisions = AutonomousBrain._shared_loader.analyze_and_modify_parameters(self.cell_type, current_params or {}, env_data)
            return decisions
        except Exception as e:
            print(f"Erreur avec CodeLlama: {e}")
            return self.fallback_decisions(environment)

    def fallback_decisions(self, environment=None):
        """Décisions de fallback quand l'IA n'est pas disponible - logique basée sur le type de cellule"""
        env_state = environment.get_state() if environment else {}

        # Logique simple basée sur le type de cellule et conditions environnementales
        if self.cell_type == "Arbre":
            light_level = env_state.get("light_level", 70)
            temperature = env_state.get("temperature", 20)
            soil_moisture = env_state.get("soil_moisture", 45)

            adaptations = {}

            # Adaptation à la lumière
            if light_level > 80:
                adaptations["internal_states"] = {"photosynthèse_rate": 1.2, "résistance_stress": 0.8}
                adaptations["physical_objects"] = {"feuilles": {"efficacité": 0.95}}
                adaptations["reasoning"] = "Forte luminosité détectée: optimisation photosynthétique"
            elif light_level < 40:
                adaptations["internal_states"] = {"absorption_eau": 0.9, "résistance_stress": 0.7}
                adaptations["physical_objects"] = {"racines": {"absorption": 0.9}}
                adaptations["reasoning"] = "Faible luminosité: focus sur absorption racinaire"

            # Adaptation à la température
            if temperature > 28:
                adaptations["internal_states"] = {"résistance_stress": 0.9}
                adaptations["physical_objects"] = {"tronc": {"résistance": 0.95}}
                adaptations["reasoning"] = "Température élevée: renforcement structurel"

            # Adaptation à l'humidité du sol
            if soil_moisture < 35:
                adaptations["internal_states"] = {"absorption_eau": 1.1}
                adaptations["physical_objects"] = {"racines": {"profondeur": 3.0, "réseau": 90.0}}
                adaptations["reasoning"] = "Sol sec: développement racinaire intensif"

            adaptations["environment_influence"] = {}
            return adaptations if adaptations else {
                "internal_states": {"photosynthèse_rate": 0.9, "résistance_stress": 0.7},
                "physical_objects": {"feuilles": {"efficacité": 0.85}},
                "environment_influence": {},
                "reasoning": "Adaptation basique en l'absence d'IA"
            }

        elif self.cell_type == "Fleur":
            wind_speed = env_state.get("wind_speed", 5)
            temperature = env_state.get("temperature", 20)

            if wind_speed > 12:
                return {
                    "internal_states": {"résistance_sécheresse": 0.8},
                    "physical_objects": {"pétales": {"résistance_uv": 0.8, "attractivité": 0.9}},
                    "environment_influence": {},
                    "reasoning": "Vent fort: optimisation pollinisation"
                }
            elif temperature > 25:
                return {
                    "internal_states": {"production_nectar": 1.1},
                    "physical_objects": {"nectar": {"quantité": 60.0, "qualité": 0.9}},
                    "environment_influence": {},
                    "reasoning": "Chaleur: augmentation production nectar"
                }

        elif self.cell_type == "Ecureuil":
            light_level = env_state.get("light_level", 70)

            if light_level < 50:
                return {
                    "internal_states": {"vitesse_mouvement": 0.9, "intelligence_problème": 0.8},
                    "physical_objects": {"yeux": {"vision_nocturne": 0.7}},
                    "environment_influence": {},
                    "reasoning": "Faible lumière: adaptation nocturne"
                }

        elif self.cell_type == "Climat":
            humidity = env_state.get("humidity", 60)

            if humidity > 75:
                return {
                    "internal_states": {"précision_prédiction": 0.9, "détection_changements": 0.95},
                    "physical_objects": {"analyseurs_humidité": {"sensibilité": 0.95}},
                    "environment_influence": {},
                    "reasoning": "Humidité élevée: surveillance météo renforcée"
                }

        # Décisions par défaut
        return {
            "internal_states": {},
            "physical_objects": {},
            "environment_influence": {},
            "reasoning": "Aucune adaptation spécifique requise"
        }

def simulate_ecosystem():
    """Simule un écosystème vivant avec cellules autonomes"""

    # Créer l'environnement
    environment = Environment()

    # Créer les cellules autonomes
    cells = [
        AutonomousCell("Chêne Millénaire", "Arbre", "arbres_biology", environment),
        AutonomousCell("Système Climatique", "Climat", "climat_science", environment),
        AutonomousCell("Écureuil Agile", "Ecureuil", "ecureuil_behavior", environment),
        AutonomousCell("Rose Sauvage", "Fleur", "fleur_biology", environment)
    ]

    print("🌿 Simulation d'écosystème vivant commencée\n")
    print("📊 État initial de l'environnement:")
    print(json.dumps(environment.get_state(), indent=2))
    print("\n" + "="*60 + "\n")

    # Simulation sur plusieurs cycles
    for cycle in range(3):
        print(f"🔄 Cycle {cycle + 1}/3 - Les cellules perçoivent et agissent\n")

        # Modifier l'environnement aléatoirement pour simuler des changements
        if cycle == 1:
            environment.temperature += 10  # Chaleur
            environment.events.append("chaleur_soudaine")
            print("🌡️ Changement environnemental: chaleur soudaine!")
        elif cycle == 2:
            environment.humidity += 20  # Pluie
            environment.events.append("pluie")
            print("🌧️ Changement environnemental: pluie!")

        # Chaque cellule s'adapte de manière complètement autonome
        for cell in cells:
            print(f"\n🦠 {cell.name} ({cell.cell_type})")
            cell.adapt_autonomously()

        print(f"\n📊 État de l'environnement après cycle {cycle + 1}:")
        print(json.dumps(environment.get_state(), indent=2))
        print("\n" + "="*60 + "\n")

        time.sleep(1)  # Pause pour lisibilité

    # Afficher l'état final des cellules
    print("🏁 État final des cellules:")
    for cell in cells:
        print(f"\n{cell.name}:")
        print(json.dumps(cell.get_current_state(), indent=2))

if __name__ == "__main__":
    simulate_ecosystem()