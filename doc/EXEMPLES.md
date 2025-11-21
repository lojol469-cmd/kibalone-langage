# 💡 Exemples de Code - Écosystème KIBALI

## Exemples Pratiques d'Utilisation

### 1. Démarrage Rapide

```python
#!/usr/bin/env python3
"""
Exemple de démarrage rapide de l'écosystème KIBALI
"""

from kibali import Ecosysteme, AgentKibali, Environnement
from kibali.cells import Arbre, Climat, Ecureuil, Fleur

def main():
    # Création de l'écosystème
    ecosysteme = Ecosysteme()

    # Ajout de cellules
    ecosysteme.ajouter_cellule(Arbre("Chêne Millénaire", age=100))
    ecosysteme.ajouter_cellule(Climat("Forêt Tempérée"))
    ecosysteme.ajouter_cellule(Ecureuil("Noisette"))
    ecosysteme.ajouter_cellule(Fleur("Rose", saison="printemps"))

    # Configuration de l'environnement
    environnement = Environnement(
        temperature=22,
        humidite=65,
        luminosite=70
    )
    ecosysteme.definir_environnement(environnement)

    # Création et démarrage de l'agent
    agent = AgentKibali()
    agent.connecter_ecosysteme(ecosysteme)

    # Exécution de cycles d'orchestration
    for cycle in range(10):
        rapport = agent.cycle_orchestration()
        print(f"Cycle {cycle}: {len(rapport.adaptations)} adaptations")

    print("Écosystème stabilisé !")

if __name__ == "__main__":
    main()
```

### 2. Simulation de Stress Climatique

```python
#!/usr/bin/env python3
"""
Simulation d'une canicule et adaptation des cellules
"""

from kibali import Ecosysteme, AgentKibali
from kibali.cells import Arbre, Fleur
from kibali.environment import EnvironnementStressant
import time

def simuler_canicule():
    # Écosystème initial
    ecosysteme = Ecosysteme()
    ecosysteme.ajouter_cellule(Arbre("Chêne", age=50))
    ecosysteme.ajouter_cellule(Fleur("Tulipe", saison="printemps"))

    # Agent intelligent
    agent = AgentKibali(strategie="survie_maximale")

    # Conditions normales
    env_normal = EnvironnementStressant(temperature=22, humidite=65)
    ecosysteme.definir_environnement(env_normal)

    print("=== Conditions Normales ===")
    for cycle in range(5):
        rapport = agent.cycle_orchestration()
        print(f"Cycle {cycle}: Stress moyen = {rapport.stress_moyen:.2f}")

    # Déclenchement de la canicule
    print("\n=== Canicule Déclenchée ===")
    env_canicule = EnvironnementStressant(
        temperature=38,  # Canicule
        humidite=25,     # Sécheresse
        vent=15          # Vent fort
    )
    ecosysteme.definir_environnement(env_canicule)

    # Observation des adaptations
    for cycle in range(10):
        rapport = agent.cycle_orchestration()

        if rapport.urgence_declaree:
            print(f"🚨 URGENCE Cycle {cycle}: {len(rapport.adaptations_urgence)} adaptations d'urgence")

        # Affichage des adaptations cellulaires
        for adaptation in rapport.adaptations:
            print(f"  {adaptation.cellule.nom}: {adaptation.type_adaptation}")

        time.sleep(0.5)  # Pause pour observation

    print("\n=== Résultats de la Simulation ===")
    print(f"Écosystème stabilisé: {agent.evaluer_stabilite()}")

def analyser_adaptations_evolution():
    """Analyse l'évolution des stratégies d'adaptation"""
    agent = AgentKibali()

    # Simulation de différents scénarios
    scenarios = [
        {"temp": 35, "humidite": 30, "description": "Sécheresse modérée"},
        {"temp": 42, "humidite": 15, "description": "Canicule extrême"},
        {"temp": 5, "humidite": 90, "description": "Hiver rigoureux"}
    ]

    for scenario in scenarios:
        print(f"\n--- Scénario: {scenario['description']} ---")

        # Configuration environnementale
        env = EnvironnementStressant(
            temperature=scenario['temp'],
            humidite=scenario['humidite']
        )

        # Analyse des adaptations possibles
        adaptations = agent.analyser_adaptations_possibles(env)
        print(f"Adaptations identifiées: {len(adaptations)}")

        # Évaluation de l'efficacité
        efficacite = agent.evaluer_efficacite_adaptations(adaptations)
        print(f"Efficacité moyenne: {efficacite:.2%}")

if __name__ == "__main__":
    simuler_canicule()
    analyser_adaptations_evolution()
```

### 3. Création de Cellules Personnalisées

```python
#!/usr/bin/env python3
"""
Exemple de création de cellules personnalisées
"""

from kibali.cells.base import CelluleBase
from kibali.core import EtatInterne, ObjetPhysique, Capteur
from typing import Dict, List
import random

class AlgueMarine(CelluleBase):
    """Cellule représentant une algue marine photosynthétique"""

    def __init__(self, nom: str, profondeur: int = 10):
        super().__init__(
            nom=nom,
            type_cellule="algue_marine",
            etats_internes={
                "photosynthese_rate": EtatInterne(1.0, "efficacite", 0.5, 2.0),
                "biomasse": EtatInterne(100, "mg", 10, 1000),
                "stress_salin": EtatInterne(20, "%", 0, 100),
                "profondeur": EtatInterne(profondeur, "m", 0, 100)
            },
            objets_physiques={
                "fronde": ObjetPhysique("structure", "fonctionnel"),
                "racine": ObjetPhysique("ancrage", "fonctionnel"),
                "pigments": ObjetPhysique("photosynthetique", "optimal")
            }
        )

        # Capteurs spécialisés
        self.capteurs = [
            Capteur("luminosite_sous_marine", "lux"),
            Capteur("temperature_eau", "celsius"),
            Capteur("salinite", "ppt"),
            Capteur("courant", "m/s")
        ]

    def percevoir_environnement(self) -> Dict[str, float]:
        """Perception spécialisée pour milieu aquatique"""
        perceptions = super().percevoir_environnement()

        # Ajustements pour l'environnement marin
        profondeur = self.etats_internes["profondeur"].valeur
        perceptions["luminosite_effective"] = max(0, perceptions.get("luminosite", 100) * (1 - profondeur/100))
        perceptions["pression"] = profondeur * 0.1  # bars

        return perceptions

    def adapter_autonomously(self, perceptions: Dict[str, float]) -> List[str]:
        """Adaptations spécifiques aux algues marines"""
        adaptations = []

        # Adaptation à la profondeur
        lumiere = perceptions.get("luminosite_effective", 50)
        if lumiere < 20:
            adaptations.append("augmenter_pigments_photosynthetiques")
            self.etats_internes["photosynthese_rate"].valeur *= 1.2

        # Gestion du stress salin
        salinite = perceptions.get("salinite", 35)
        if salinite > 40:
            adaptations.append("activer_defenses_osmotiques")
            self.etats_internes["stress_salin"].valeur += 5

        # Migration verticale si nécessaire
        if lumiere < 10 and profondeur > 5:
            adaptations.append("migration_vers_surface")
            self.etats_internes["profondeur"].valeur -= 2

        return adaptations

class InsectePollinisateur(CelluleBase):
    """Cellule représentant un insecte pollinisateur"""

    def __init__(self, nom: str, espece: str = "abeille"):
        super().__init__(
            nom=nom,
            type_cellule="pollinisateur",
            etats_internes={
                "energie": EtatInterne(100, "%", 0, 100),
                "charge_pollinique": EtatInterne(0, "grains", 0, 100),
                "distance_parcourue": EtatInterne(0, "km", 0, 1000),
                "stress_thermique": EtatInterne(10, "%", 0, 100)
            },
            objets_physiques={
                "ailes": ObjetPhysique("locomotion", "fonctionnel"),
                "dards": ObjetPhysique("defense", "recharge"),
                "corbeille": ObjetPhysique("collecte", "vide")
            }
        )

        self.espece = espece
        self.fleurs_visitees = []
        self.territoires = []

    def chercher_nourriture(self, fleurs_disponibles: List['Fleur']) -> List[str]:
        """Stratégie de recherche de nourriture"""
        actions = []

        # Sélection des fleurs attractives
        fleurs_attractives = [
            fleur for fleur in fleurs_disponibles
            if fleur.etats_internes["nectar_disponible"].valeur > 10
        ]

        if fleurs_attractives:
            fleur_cible = random.choice(fleurs_attractives)
            actions.append(f"voler_vers_{fleur_cible.nom}")
            actions.append(f"butiner_{fleur_cible.nom}")

            # Collecte de pollen
            pollen = min(20, fleur_cible.etats_internes["pollen_disponible"].valeur)
            self.etats_internes["charge_pollinique"].valeur += pollen
            fleur_cible.etats_internes["pollen_disponible"].valeur -= pollen

            self.fleurs_visitees.append(fleur_cible.nom)

        return actions

    def adapter_autonomously(self, perceptions: Dict[str, float]) -> List[str]:
        """Adaptations comportementales"""
        adaptations = []

        temperature = perceptions.get("temperature", 20)

        # Gestion de la température
        if temperature > 35:
            adaptations.append("rechercher_ombre")
            self.etats_internes["stress_thermique"].valeur += 10
        elif temperature < 10:
            adaptations.append("rechercher_chaleur")
            self.etats_internes["energie"].valeur -= 5

        # Gestion de l'énergie
        if self.etats_internes["energie"].valeur < 30:
            adaptations.append("retour_ruche")
        else:
            adaptations.append("continuer_pollinisation")

        # Optimisation du territoire
        if len(self.fleurs_visitees) > 10:
            adaptations.append("optimiser_itineraire")

        return adaptations

def demonstrer_cellules_personnalisees():
    """Démonstration des cellules personnalisées"""
    from kibali import Ecosysteme, AgentKibali

    # Création de l'écosystème aquatique
    ecosysteme_ocean = Ecosysteme()
    ecosysteme_ocean.ajouter_cellule(AlgueMarine("Ulva lactuca", profondeur=5))
    ecosysteme_ocean.ajouter_cellule(AlgueMarine("Sargassum", profondeur=15))

    # Création de l'écosystème terrestre
    ecosysteme_foret = Ecosysteme()
    ecosysteme_foret.ajouter_cellule(InsectePollinisateur("Maya", "abeille"))
    ecosysteme_foret.ajouter_cellule(InsectePollinisateur("Buzz", "bourdon"))

    # Agent multi-écosystèmes
    agent = AgentKibali()

    print("=== Écosystème Aquatique ===")
    for cycle in range(5):
        rapport = agent.orchestrer_ecosysteme(ecosysteme_ocean)
        print(f"Cycle {cycle}: {len(rapport.adaptations)} adaptations marines")

    print("\n=== Écosystème Terrestre ===")
    for cycle in range(5):
        rapport = agent.orchestrer_ecosysteme(ecosysteme_foret)
        print(f"Cycle {cycle}: {len(rapport.adaptations)} adaptations pollinisatrices")

if __name__ == "__main__":
    demonstrer_cellules_personnalisees()
```

### 4. Utilisation Avancée de l'IA

```python
#!/usr/bin/env python3
"""
Exemples d'utilisation avancée des capacités IA
"""

from kibali.ai import ModeleIA, AnalyseurIA
from kibali.core import SituationComplexe
import asyncio

async def analyser_situation_complexe():
    """Analyse IA d'une situation écologique complexe"""

    # Configuration des modèles IA
    codellama = ModeleIA(
        nom="codellama",
        type_modele="llm",
        chemin_modele="codellama/CodeLlama-7b-hf",
        quantification="4bit"
    )

    phi = ModeleIA(
        nom="phi",
        type_modele="analyse",
        chemin_modele="microsoft/phi-1_5"
    )

    # Création de l'analyseur
    analyseur = AnalyseurIA([codellama, phi])

    # Situation complexe à analyser
    situation = SituationComplexe(
        description="""
        Une forêt temperate fait face à une sécheresse prolongée.
        Les chênes montrent des signes de stress hydrique sévère,
        tandis que les pins semblent mieux résister.
        Les écureuils modifient leurs comportements de stockage.
        Température: 35°C, Humidité: 25%, Précipitations: 0mm depuis 3 semaines.
        """,
        parametres={
            "temperature": 35,
            "humidite": 25,
            "stress_hydrique": 0.8,
            "adaptation_pins": 0.6,
            "adaptation_chênes": 0.3
        },
        cellules_concernees=["chênes", "pins", "écureuils"],
        enjeux=["survie_espèces", "biodiversite", "regeneration"]
    )

    print("=== Analyse par Code Llama ===")
    analyse_codellama = await analyseur.analyser_avec_modele(
        situation, "codellama",
        prompt_specialise="""
        En tant que biologiste computationnel, analyse cette situation
        et propose des adaptations génétiques potentielles sous forme de code.
        """
    )
    print(analyse_codellama.resultat)

    print("\n=== Analyse par Phi ===")
    analyse_phi = await analyseur.analyser_avec_modele(
        situation, "phi",
        prompt_specialise="""
        Fournis une analyse rapide et des recommandations immédiates
        pour la gestion de crise écologique.
        """
    )
    print(analyse_phi.resultat)

    print("\n=== Synthèse Comparative ===")
    synthese = await analyseur.comparer_analyses([analyse_codellama, analyse_phi])
    print(f"Convergences: {synthese.convergences}")
    print(f"Divergences: {synthese.divergences}")
    print(f"Recommandation finale: {synthese.recommandation_principale}")

async def generer_adaptations_ia():
    """Génération d'adaptations cellulaires via IA"""

    generateur = GenerateurAdaptationsIA()

    # Spécifications d'adaptation
    specs = {
        "cellule_cible": "arbre",
        "stress": "thermique",
        "severite": "haute",
        "duree": "long_terme",
        "contraintes": ["photosynthese", "croissance", "reproduction"]
    }

    print("=== Génération d'Adaptations ===")

    # Génération par Code Llama (approche détaillée)
    adaptations_codellama = await generateur.generer_adaptations(
        specs, "codellama",
        style="scientifique_detaille"
    )

    for i, adaptation in enumerate(adaptations_codellama, 1):
        print(f"{i}. {adaptation.nom}")
        print(f"   Mécanisme: {adaptation.mecanisme}")
        print(f"   Efficacité estimée: {adaptation.efficacite:.1%}")
        print(f"   Coûts énergétiques: {adaptation.couts_energetiques}")
        print()

    # Génération par Phi (approche pragmatique)
    adaptations_phi = await generateur.generer_adaptations(
        specs, "phi",
        style="pragmatique_rapide"
    )

    print("=== Validation des Adaptations ===")
    validation = await generateur.valider_adaptations(
        adaptations_codellama + adaptations_phi,
        criteres=["faisabilite", "efficacite", "stabilite"]
    )

    print(f"Adaptations validées: {len(validation.adaptations_validees)}/{len(adaptations_codellama + adaptations_phi)}")
    print(f"Score moyen: {validation.score_moyen:.2f}/5")

async def apprentissage_ia_ecosysteme():
    """Apprentissage continu de l'IA sur l'écosystème"""

    apprentissage = SystemeApprentissageIA()

    # Collecte de données historiques
    donnees_historiques = [
        {
            "situation": "canicule_2023",
            "adaptations": ["augmenter_resistance_thermique", "reduire_transpiration"],
            "resultat": "survie_85_pourcent",
            "lecons": ["adaptation_thermique_efficace", "gestion_eau_critique"]
        },
        {
            "situation": "inondation_2023",
            "adaptations": ["migration_altitude", "stockage_semences"],
            "resultat": "regeneration_complete",
            "lecons": ["mobilite_adaptative", "strategie_reproduction"]
        }
    ]

    print("=== Apprentissage à partir de l'Histoire ===")

    # Apprentissage supervisé
    modeles_entraines = await apprentissage.entrainer_modeles(
        donnees_historiques,
        objectifs=["prediction_adaptations", "evaluation_risques"]
    )

    print(f"Modèles entraînés: {len(modeles_entraines)}")

    # Test de prédiction
    situation_future = {
        "temperature": 40,
        "humidite": 20,
        "duree_prevision": "2_mois"
    }

    prediction = await apprentissage.predire_evolution(situation_future)
    print(f"Prédiction: {prediction.scenario_principal}")
    print(f"Confiance: {prediction.confiance:.1%}")
    print(f"Adaptations recommandées: {prediction.adaptations_recommandees}")

async def main():
    """Démonstration complète des capacités IA"""
    print("🚀 Démonstration des Capacités IA de KIBALI\n")

    await analyser_situation_complexe()
    print("\n" + "="*50 + "\n")

    await generer_adaptations_ia()
    print("\n" + "="*50 + "\n")

    await apprentissage_ia_ecosysteme()

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Intégration avec des Outils Externes

```python
#!/usr/bin/env python3
"""
Exemples d'intégration avec des outils externes
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from kibali import Ecosysteme, AgentKibali
import json
import os

class IntegrationMeteo:
    """Intégration avec des données météorologiques réelles"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def obtenir_conditions_reelles(self, ville: str = "Paris") -> dict:
        """Récupération des conditions météorologiques actuelles"""
        params = {
            "q": ville,
            "appid": self.api_key,
            "units": "metric"
        }

        response = requests.get(f"{self.base_url}/weather", params=params)
        data = response.json()

        return {
            "temperature": data["main"]["temp"],
            "humidite": data["main"]["humidity"],
            "pression": data["main"]["pressure"],
            "vent": data["wind"]["speed"],
            "description": data["weather"][0]["description"]
        }

    def obtenir_previsions(self, ville: str = "Paris", jours: int = 5) -> list:
        """Récupération des prévisions météorologiques"""
        params = {
            "q": ville,
            "appid": self.api_key,
            "units": "metric"
        }

        response = requests.get(f"{self.base_url}/forecast", params=params)
        data = response.json()

        previsions = []
        for i in range(0, min(jours * 8, len(data["list"])), 8):  # Une fois par jour
            item = data["list"][i]
            previsions.append({
                "date": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "humidite": item["main"]["humidity"],
                "description": item["weather"][0]["description"]
            })

        return previsions

class IntegrationBaseDonnees:
    """Intégration avec des bases de données biologiques"""

    def __init__(self):
        self.cache = {}

    def rechercher_espece(self, nom_espece: str) -> dict:
        """Recherche d'informations sur une espèce"""
        if nom_espece in self.cache:
            return self.cache[nom_espece]

        # Simulation d'API biologique (remplacer par API réelle)
        especes_db = {
            "Quercus robur": {
                "nom_scientifique": "Quercus robur",
                "nom_commun": "Chêne pédonculé",
                "longevite": 500,
                "resistance_secheresse": 0.7,
                "adaptation_climatique": "tempere",
                "caracteristiques": ["feuillage_caduque", "glands_comestibles"]
            },
            "Pinus sylvestris": {
                "nom_scientifique": "Pinus sylvestris",
                "nom_commun": "Pin d'Écosse",
                "longevite": 300,
                "resistance_secheresse": 0.9,
                "adaptation_climatique": "montagnard",
                "caracteristiques": ["feuillage_persistant", "cones_serotines"]
            }
        }

        resultat = especes_db.get(nom_espece, {})
        self.cache[nom_espece] = resultat
        return resultat

    def obtenir_donnees_ecologiques(self, region: str) -> dict:
        """Données écologiques par région"""
        regions_db = {
            "foret_temperee": {
                "biome": "forêt tempérée",
                "precipitations_moyennes": 800,
                "temperature_moyenne": 10,
                "especes_endemiques": ["chênes", "hêtres", "écureuils"],
                "menaces": ["fragmentation", "introduction_especes"],
                "indice_biodiversite": 0.8
            }
        }

        return regions_db.get(region, {})

class VisualisationEcosysteme:
    """Outils de visualisation des données écosystémiques"""

    def __init__(self, output_dir: str = "visualisations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def tracer_evolution_stress(self, historique_stress: list, titre: str = "Évolution du Stress Écosystémique"):
        """Graphique de l'évolution du stress"""
        plt.figure(figsize=(12, 6))
        plt.plot(historique_stress, marker='o', linewidth=2, markersize=4)
        plt.title(titre, fontsize=14, fontweight='bold')
        plt.xlabel('Cycles', fontsize=12)
        plt.ylabel('Niveau de Stress', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/evolution_stress.png", dpi=300, bbox_inches='tight')
        plt.close()

    def creer_heatmap_adaptations(self, matrice_adaptations: pd.DataFrame):
        """Heatmap des adaptations cellulaires"""
        plt.figure(figsize=(10, 8))
        plt.imshow(matrice_adaptations.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Intensité d\'adaptation')
        plt.title('Matrice des Adaptations Cellulaires', fontsize=14, fontweight='bold')
        plt.xlabel('Types d\'Adaptation')
        plt.ylabel('Cellules')
        plt.xticks(range(len(matrice_adaptations.columns)), matrice_adaptations.columns, rotation=45, ha='right')
        plt.yticks(range(len(matrice_adaptations.index)), matrice_adaptations.index)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/heatmap_adaptations.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generer_rapport_html(self, donnees_ecosysteme: dict, nom_fichier: str = "rapport_ecosysteme.html"):
        """Génération d'un rapport HTML complet"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport Écosystème KIBALI</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2E7D32; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #2E7D32; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Rapport Écosystème KIBALI</h1>
                <p>Généré le {donnees_ecosysteme.get('date_generation', 'N/A')}</p>
            </div>

            <div class="section">
                <h2>📈 Métriques Globales</h2>
                <div class="metric">Stress Moyen: {donnees_ecosysteme.get('stress_moyen', 'N/A'):.2f}</div>
                <div class="metric">Cellules Actives: {donnees_ecosysteme.get('cellules_actives', 'N/A')}</div>
                <div class="metric">Adaptations: {donnees_ecosysteme.get('total_adaptations', 'N/A')}</div>
            </div>

            <div class="section">
                <h2>🌡️ Conditions Environnementales</h2>
                <table>
                    <tr><th>Paramètre</th><th>Valeur</th><th>Unité</th></tr>
                    <tr><td>Température</td><td>{donnees_ecosysteme.get('temperature', 'N/A')}</td><td>°C</td></tr>
                    <tr><td>Humidité</td><td>{donnees_ecosysteme.get('humidite', 'N/A')}</td><td>%</td></tr>
                    <tr><td>Luminosité</td><td>{donnees_ecosysteme.get('luminosite', 'N/A')}</td><td>%</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>🧬 État des Cellules</h2>
                <table>
                    <tr><th>Cellule</th><th>Type</th><th>Santé</th><th>Stress</th><th>Adaptations</th></tr>
                    {"".join([f"<tr><td>{c['nom']}</td><td>{c['type']}</td><td>{c['sante']:.1f}</td><td>{c['stress']:.1f}</td><td>{c['adaptations']}</td></tr>" for c in donnees_ecosysteme.get('cellules', [])])}
                </table>
            </div>
        </body>
        </html>
        """

        with open(f"{self.output_dir}/{nom_fichier}", 'w', encoding='utf-8') as f:
            f.write(html_content)

def demonstrer_integrations():
    """Démonstration des intégrations externes"""

    # Configuration des intégrations
    meteo = IntegrationMeteo(api_key="VOTRE_CLE_API_OPENWEATHERMAP")
    db_bio = IntegrationBaseDonnees()
    visu = VisualisationEcosysteme()

    # Écosystème KIBALI
    ecosysteme = Ecosysteme()
    agent = AgentKibali()

    print("=== Intégration Météorologique ===")

    # Conditions météorologiques réelles
    conditions = meteo.obtenir_conditions_reelles("Paris")
    print(f"Conditions actuelles à Paris: {conditions}")

    # Application à l'écosystème
    environnement = {
        "temperature": conditions["temperature"],
        "humidite": conditions["humidite"],
        "vent": conditions["vent"] * 3.6  # Conversion m/s vers km/h
    }

    print(f"Environnement appliqué: {environnement}")

    print("\n=== Intégration Base de Données Biologiques ===")

    # Recherche d'espèces
    info_chene = db_bio.rechercher_espece("Quercus robur")
    print(f"Informations Chêne: {info_chene}")

    # Données écologiques
    data_region = db_bio.obtenir_donnees_ecologiques("foret_temperee")
    print(f"Données forêt tempérée: {data_region}")

    print("\n=== Simulation avec Données Réelles ===")

    # Simulation sur plusieurs cycles
    historique_stress = []
    for cycle in range(10):
        rapport = agent.cycle_orchestration()

        # Collecte des métriques
        stress_moyen = sum(c.etats_internes["stress"].valeur for c in ecosysteme.cellules) / len(ecosysteme.cellules)
        historique_stress.append(stress_moyen)

        print(f"Cycle {cycle}: Stress moyen = {stress_moyen:.2f}")

    print("\n=== Génération des Visualisations ===")

    # Création des graphiques
    visu.tracer_evolution_stress(historique_stress)

    # Création du rapport HTML
    donnees_rapport = {
        "date_generation": "2024-12-21",
        "stress_moyen": sum(historique_stress) / len(historique_stress),
        "cellules_actives": len(ecosysteme.cellules),
        "total_adaptations": sum(len(c.adaptations_appliquees) for c in ecosysteme.cellules),
        "temperature": environnement["temperature"],
        "humidite": environnement["humidite"],
        "luminosite": 70,
        "cellules": [
            {
                "nom": c.nom,
                "type": c.type_cellule,
                "sante": c.etats_internes["sante"].valeur,
                "stress": c.etats_internes["stress"].valeur,
                "adaptations": len(c.adaptations_appliquees)
            } for c in ecosysteme.cellules
        ]
    }

    visu.generer_rapport_html(donnees_rapport)

    print("Visualisations générées dans le dossier 'visualisations/'")

if __name__ == "__main__":
    demonstrer_integrations()
```

---

*Ces exemples montrent la flexibilité et la puissance de l'écosystème KIBALI pour diverses applications.*