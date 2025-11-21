# 🌍 Environnement - Conditions Écologiques Globales

"""Module Environnement - Gestion des conditions écologiques

L'environnement représente l'ensemble des conditions externes qui influencent
les cellules de l'écosystème KIBALI :
- Conditions météorologiques (température, humidité, luminosité)
- Qualité de l'air et du sol
- Disponibilité des ressources
- Événements saisonniers et climatiques
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import math
import logging

from ..shared.config import Config

@dataclass
class ConditionEnvironnementale:
    """Représente une condition environnementale spécifique"""
    nom: str
    valeur: float
    unite: str
    minimum: float
    maximum: float
    volatilite: float  # Taux de changement naturel
    description: str = ""

    def fluctuer(self) -> float:
        """Fait fluctuer la valeur selon la volatilité"""
        changement = random.gauss(0, self.volatilite)
        nouvelle_valeur = self.valeur + changement

        # Contrainte dans les limites
        nouvelle_valeur = max(self.minimum, min(self.maximum, nouvelle_valeur))

        ancien_valeur = self.valeur
        self.valeur = nouvelle_valeur

        return nouvelle_valeur - ancien_valeur

    def influencer_cellule(self, type_cellule: str) -> float:
        """Calcule l'influence sur un type de cellule

        Args:
            type_cellule: Type de cellule affectée

        Returns:
            float: Facteur d'influence (-1 à 1, négatif = stress)
        """
        # Logique simplifiée d'influence
        influences = {
            "temperature": {
                "vegetal": lambda x: -abs(x - 20) / 20,  # Optimum à 20°C
                "animal": lambda x: -abs(x - 25) / 25,    # Optimum à 25°C
                "microbien": lambda x: -abs(x - 30) / 30  # Optimum à 30°C
            },
            "humidite": {
                "vegetal": lambda x: (x - 50) / 50,       # Préfère humidité moyenne
                "animal": lambda x: (x - 60) / 60,        # Préfère humidité plus élevée
                "microbien": lambda x: (x - 70) / 70      # Préfère très humide
            },
            "luminosite": {
                "vegetal": lambda x: min(1.0, x / 1000),  # Plus de lumière = mieux
                "animal": lambda x: -min(1.0, x / 2000),  # Trop de lumière = stress
                "microbien": lambda x: -min(1.0, x / 500) # Préfère l'obscurité
            }
        }

        if self.nom in influences and type_cellule in influences[self.nom]:
            return influences[self.nom][type_cellule](self.valeur)
        else:
            return 0.0  # Pas d'influence

@dataclass
class EvenementEcologique:
    """Représente un événement écologique ponctuel"""
    nom: str
    type_evenement: str
    intensite: float  # 0-1
    duree: timedelta
    date_debut: datetime
    description: str = ""
    effets: Dict[str, float] = field(default_factory=dict)  # Effets sur les conditions

    @property
    def actif(self) -> bool:
        """Vérifie si l'événement est actuellement actif"""
        maintenant = datetime.now()
        return self.date_debut <= maintenant <= self.date_debut + self.duree

    @property
    def progression(self) -> float:
        """Retourne la progression de l'événement (0-1)"""
        if not self.actif:
            return 0.0

        maintenant = datetime.now()
        ecoule = maintenant - self.date_debut
        return min(1.0, ecoule.total_seconds() / self.duree.total_seconds())

    def calculer_effet_actuel(self) -> Dict[str, float]:
        """Calcule les effets actuels de l'événement"""
        if not self.actif:
            return {}

        # Effet progressif (montée puis descente)
        progression = self.progression
        facteur_temps = math.sin(progression * math.pi)  # Courbe en cloche

        effets_actuels = {}
        for condition, effet_base in self.effets.items():
            effets_actuels[condition] = effet_base * facteur_temps * self.intensite

        return effets_actuels

@dataclass
class Saison:
    """Représente une saison climatique"""
    nom: str
    duree_moyenne: int  # jours
    caracteristiques: Dict[str, Dict[str, float]]  # conditions -> {moyenne, amplitude}
    description: str = ""

    def generer_conditions(self, jour_annee: int) -> Dict[str, float]:
        """Génère les conditions pour un jour donné de l'année

        Args:
            jour_annee: Jour de l'année (0-364)

        Returns:
            Dict: Conditions générées
        """
        conditions = {}

        for condition, params in self.caracteristiques.items():
            moyenne = params["moyenne"]
            amplitude = params["amplitude"]

            # Variation saisonnière sinusoïdale
            variation = math.sin(2 * math.pi * jour_annee / 365) * amplitude
            conditions[condition] = moyenne + variation

        return conditions

class Environnement:
    """Classe principale de gestion de l'environnement

    Gère l'ensemble des conditions environnementales et leurs évolutions :
    - Conditions météorologiques
    - Événements écologiques
    - Cycles saisonniers
    - Interactions avec les cellules
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialise l'environnement

        Args:
            config: Configuration optionnelle
        """
        self.config = config or Config()

        # Conditions environnementales de base
        self.conditions: Dict[str, ConditionEnvironnementale] = {
            "temperature": ConditionEnvironnementale(
                "temperature", 20.0, "°C", -50.0, 60.0, 2.0,
                "Température ambiante"
            ),
            "humidite": ConditionEnvironnementale(
                "humidite", 60.0, "%", 0.0, 100.0, 5.0,
                "Taux d'humidité relative"
            ),
            "luminosite": ConditionEnvironnementale(
                "luminosite", 1000.0, "lux", 0.0, 10000.0, 200.0,
                "Intensité lumineuse"
            ),
            "qualite_air": ConditionEnvironnementale(
                "qualite_air", 80.0, "%", 0.0, 100.0, 3.0,
                "Qualité de l'air (oxygène, pollution)"
            ),
            "nutriments_sol": ConditionEnvironnementale(
                "nutriments_sol", 50.0, "%", 0.0, 100.0, 2.0,
                "Disponibilité des nutriments dans le sol"
            ),
            "ph_sol": ConditionEnvironnementale(
                "ph_sol", 7.0, "", 0.0, 14.0, 0.5,
                "pH du sol"
            )
        }

        # Événements actifs
        self.evenements_actifs: List[EvenementEcologique] = []

        # Saisons
        self.saisons = self._initialiser_saisons()

        # État temporel
        self.date_courante = datetime.now()
        self.jour_annee = self.date_courante.timetuple().tm_yday - 1  # 0-364
        self.saison_courante = self._determiner_saison()

        # Historique
        self.historique_conditions: List[Dict[str, Any]] = []
        self.historique_evenements: List[Dict[str, Any]] = []

        # Logger
        self.logger = logging.getLogger("Environnement")

        # Initialisation des conditions saisonnières
        self._appliquer_conditions_saisonniere()

        self.logger.info("Environnement KIBALI initialisé")

    def evoluer(self, heures_ecoulees: float = 1.0) -> Dict[str, Any]:
        """Fait évoluer l'environnement

        Args:
            heures_ecoulees: Nombre d'heures écoulées

        Returns:
            Dict: Résultats de l'évolution
        """
        changements = {}
        evenements_apparus = []

        # Évolution naturelle des conditions
        for nom, condition in self.conditions.items():
            changement = condition.fluctuer()
            changements[nom] = changement

        # Évolution des événements
        for evenement in self.evenements_actifs[:]:  # Copie pour modification
            if not evenement.actif:
                self.evenements_actifs.remove(evenement)
                self.historique_evenements.append({
                    "evenement": evenement.nom,
                    "fin": datetime.now(),
                    "duree_reelle": datetime.now() - evenement.date_debut
                })
            else:
                # Application des effets de l'événement
                effets = evenement.calculer_effet_actuel()
                for condition_nom, effet in effets.items():
                    if condition_nom in self.conditions:
                        self.conditions[condition_nom].valeur += effet
                        changements[condition_nom] = changements.get(condition_nom, 0) + effet

        # Vérification des nouveaux événements
        nouvel_evenement = self._verifier_evenements_aleatoires()
        if nouvel_evenement:
            self.evenements_actifs.append(nouvel_evenement)
            evenements_apparus.append(nouvel_evenement.nom)

        # Mise à jour temporelle
        self.date_courante += timedelta(hours=heures_ecoulees)
        ancien_jour = self.jour_annee
        self.jour_annee = self.date_courante.timetuple().tm_yday - 1

        # Changement de saison
        if ancien_jour != self.jour_annee and self.jour_annee % 91 == 0:  # Tous les ~3 mois
            self._appliquer_conditions_saisonniere()

        # Enregistrement dans l'historique
        self.historique_conditions.append({
            "timestamp": self.date_courante,
            "conditions": {nom: c.valeur for nom, c in self.conditions.items()},
            "changements": changements,
            "evenements_actifs": [e.nom for e in self.evenements_actifs]
        })

        # Limitation de l'historique
        if len(self.historique_conditions) > 1000:
            self.historique_conditions.pop(0)

        resultats = {
            "changements_conditions": changements,
            "evenements_apparus": evenements_apparus,
            "evenements_actifs": len(self.evenements_actifs),
            "saison_courante": self.saison_courante,
            "jour_annee": self.jour_annee,
            "timestamp": self.date_courante
        }

        return resultats

    def obtenir_perceptions(self, type_cellule: str) -> Dict[str, float]:
        """Obtient les perceptions environnementales pour un type de cellule

        Args:
            type_cellule: Type de cellule

        Returns:
            Dict: Perceptions avec facteurs d'influence
        """
        perceptions = {}

        for nom, condition in self.conditions.items():
            valeur = condition.valeur
            influence = condition.influencer_cellule(type_cellule)
            perceptions[nom] = {
                "valeur": valeur,
                "influence": influence,
                "description": condition.description
            }

        return perceptions

    def calculer_stress_global(self) -> float:
        """Calcule le niveau de stress global de l'environnement

        Returns:
            float: Niveau de stress (0-1)
        """
        stress_total = 0.0
        poids_conditions = {
            "temperature": 0.3,
            "humidite": 0.2,
            "qualite_air": 0.25,
            "nutriments_sol": 0.15,
            "ph_sol": 0.1
        }

        for nom, poids in poids_conditions.items():
            if nom in self.conditions:
                condition = self.conditions[nom]
                # Stress basé sur l'écart par rapport à l'optimum
                if nom == "temperature":
                    optimum = 22.0
                elif nom == "humidite":
                    optimum = 65.0
                elif nom == "qualite_air":
                    optimum = 85.0
                elif nom == "nutriments_sol":
                    optimum = 60.0
                elif nom == "ph_sol":
                    optimum = 6.5
                else:
                    optimum = (condition.minimum + condition.maximum) / 2

                ecart_relatif = abs(condition.valeur - optimum) / (condition.maximum - condition.minimum)
                stress_total += ecart_relatif * poids

        # Bonus stress des événements
        stress_evenements = len(self.evenements_actifs) * 0.1
        stress_total += min(0.3, stress_evenements)  # Max 30% de stress additionnel

        return min(1.0, stress_total)

    def declencher_evenement(self,
                           nom: str,
                           type_evenement: str,
                           intensite: float,
                           duree_heures: float,
                           effets: Dict[str, float]) -> None:
        """Déclenche un événement écologique

        Args:
            nom: Nom de l'événement
            type_evenement: Type d'événement
            intensite: Intensité (0-1)
            duree_heures: Durée en heures
            effets: Effets sur les conditions
        """
        evenement = EvenementEcologique(
            nom=nom,
            type_evenement=type_evenement,
            intensite=intensite,
            duree=timedelta(hours=duree_heures),
            date_debut=datetime.now(),
            effets=effets
        )

        self.evenements_actifs.append(evenement)
        self.logger.info(f"Événement déclenché: {nom} ({type_evenement})")

    def modifier_condition(self, nom: str, valeur: float, temporaire: bool = False) -> bool:
        """Modifie une condition environnementale

        Args:
            nom: Nom de la condition
            valeur: Nouvelle valeur
            temporaire: Si True, la modification est temporaire

        Returns:
            bool: Succès de la modification
        """
        if nom not in self.conditions:
            return False

        condition = self.conditions[nom]
        ancienne_valeur = condition.valeur

        # Contrainte dans les limites
        valeur = max(condition.minimum, min(condition.maximum, valeur))
        condition.valeur = valeur

        if not temporaire:
            # Ajustement de la volatilité pour maintenir la nouvelle valeur
            condition.volatilite *= 0.8  # Réduction de la volatilité

        self.logger.info(f"Condition {nom}: {ancienne_valeur} -> {valeur}")
        return True

    def exporter_etat(self) -> Dict[str, Any]:
        """Exporte l'état complet de l'environnement

        Returns:
            Dict: État sérialisé
        """
        return {
            "date_courante": self.date_courante.isoformat(),
            "jour_annee": self.jour_annee,
            "saison_courante": self.saison_courante,
            "conditions": {
                nom: {
                    "valeur": c.valeur,
                    "unite": c.unite,
                    "minimum": c.minimum,
                    "maximum": c.maximum,
                    "volatilite": c.volatilite,
                    "description": c.description
                }
                for nom, c in self.conditions.items()
            },
            "evenements_actifs": [
                {
                    "nom": e.nom,
                    "type_evenement": e.type_evenement,
                    "intensite": e.intensite,
                    "duree": e.duree.total_seconds(),
                    "date_debut": e.date_debut.isoformat(),
                    "progression": e.progression,
                    "description": e.description
                }
                for e in self.evenements_actifs
            ],
            "historique_conditions": self.historique_conditions[-50:],  # Derniers 50 éléments
            "historique_evenements": self.historique_evenements[-20:]   # Derniers 20 événements
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Environnement':
        """Crée un environnement à partir d'un dictionnaire

        Args:
            data: Données sérialisées

        Returns:
            Environnement: Nouvelle instance
        """
        instance = cls()

        # Restauration des conditions
        if "conditions" in data:
            for nom, c_data in data["conditions"].items():
                if nom in instance.conditions:
                    instance.conditions[nom].valeur = c_data["valeur"]
                    instance.conditions[nom].volatilite = c_data["volatilite"]

        # Restauration temporelle
        if "date_courante" in data:
            instance.date_courante = datetime.fromisoformat(data["date_courante"])
            instance.jour_annee = data.get("jour_annee", 0)
            instance.saison_courante = data.get("saison_courante", "printemps")

        # Restauration des événements
        if "evenements_actifs" in data:
            for e_data in data["evenements_actifs"]:
                evenement = EvenementEcologique(
                    nom=e_data["nom"],
                    type_evenement=e_data["type_evenement"],
                    intensite=e_data["intensite"],
                    duree=timedelta(seconds=e_data["duree"]),
                    date_debut=datetime.fromisoformat(e_data["date_debut"]),
                    description=e_data.get("description", "")
                )
                instance.evenements_actifs.append(evenement)

        return instance

    def importer_etat(self, data: Dict[str, Any]) -> None:
        """Importe un état sauvegardé

        Args:
            data: Données sérialisées à importer
        """
        # Restauration des conditions
        if "conditions" in data:
            for nom, c_data in data["conditions"].items():
                if nom in self.conditions:
                    self.conditions[nom].valeur = c_data["valeur"]
                    self.conditions[nom].volatilite = c_data["volatilite"]

        # Restauration temporelle
        if "date_courante" in data:
            self.date_courante = datetime.fromisoformat(data["date_courante"])
            self.jour_annee = data.get("jour_annee", 0)
            self.saison_courante = data.get("saison_courante", "printemps")

        # Restauration des événements
        self.evenements_actifs = []
        if "evenements_actifs" in data:
            for e_data in data["evenements_actifs"]:
                evenement = EvenementEcologique(
                    nom=e_data["nom"],
                    type_evenement=e_data["type_evenement"],
                    intensite=e_data["intensite"],
                    duree=timedelta(seconds=e_data["duree"]),
                    date_debut=datetime.fromisoformat(e_data["date_debut"]),
                    description=e_data.get("description", "")
                )
                self.evenements_actifs.append(evenement)

        self.logger.info("État environnement importé")

    def _initialiser_saisons(self) -> Dict[str, Saison]:
        """Initialise les saisons climatiques"""
        return {
            "printemps": Saison(
                "printemps", 91,
                {
                    "temperature": {"moyenne": 15.0, "amplitude": 5.0},
                    "humidite": {"moyenne": 65.0, "amplitude": 10.0},
                    "luminosite": {"moyenne": 800.0, "amplitude": 200.0}
                },
                "Saison de renaissance et croissance"
            ),
            "ete": Saison(
                "ete", 94,
                {
                    "temperature": {"moyenne": 25.0, "amplitude": 8.0},
                    "humidite": {"moyenne": 55.0, "amplitude": 15.0},
                    "luminosite": {"moyenne": 1000.0, "amplitude": 300.0}
                },
                "Saison chaude et ensoleillée"
            ),
            "automne": Saison(
                "automne", 91,
                {
                    "temperature": {"moyenne": 12.0, "amplitude": 6.0},
                    "humidite": {"moyenne": 70.0, "amplitude": 12.0},
                    "luminosite": {"moyenne": 600.0, "amplitude": 150.0}
                },
                "Saison de récolte et préparation"
            ),
            "hiver": Saison(
                "hiver", 89,
                {
                    "temperature": {"moyenne": 2.0, "amplitude": 4.0},
                    "humidite": {"moyenne": 75.0, "amplitude": 8.0},
                    "luminosite": {"moyenne": 400.0, "amplitude": 100.0}
                },
                "Saison froide et de repos"
            )
        }

    def _determiner_saison(self) -> str:
        """Détermine la saison actuelle"""
        if 0 <= self.jour_annee < 91:
            return "printemps"
        elif 91 <= self.jour_annee < 185:
            return "ete"
        elif 185 <= self.jour_annee < 276:
            return "automne"
        else:
            return "hiver"

    def _appliquer_conditions_saisonniere(self) -> None:
        """Applique les conditions saisonnières actuelles"""
        saison = self.saisons[self.saison_courante]
        conditions_saison = saison.generer_conditions(self.jour_annee)

        for condition_nom, valeur in conditions_saison.items():
            if condition_nom in self.conditions:
                # Transition progressive vers les conditions saisonnières
                actuelle = self.conditions[condition_nom].valeur
                nouvelle = (actuelle * 0.7) + (valeur * 0.3)  # Lissage
                self.conditions[condition_nom].valeur = nouvelle

        self.logger.debug(f"Conditions saisonnières appliquées pour {self.saison_courante}")

    def _verifier_evenements_aleatoires(self) -> Optional[EvenementEcologique]:
        """Vérifie si un événement aléatoire doit se produire"""
        # Probabilité basse d'événement aléatoire (1% par évolution)
        if random.random() < 0.01:
            evenements_possibles = [
                {
                    "nom": "Pluie torrentielle",
                    "type": "precipitation",
                    "duree": 6,  # heures
                    "effets": {"humidite": 30.0, "temperature": -5.0}
                },
                {
                    "nom": "Vague de chaleur",
                    "type": "temperature",
                    "duree": 24,
                    "effets": {"temperature": 15.0, "humidite": -20.0}
                },
                {
                    "nom": "Tempête solaire",
                    "type": "radiation",
                    "duree": 2,
                    "effets": {"luminosite": 500.0}
                },
                {
                    "nom": "Pollution atmosphérique",
                    "type": "pollution",
                    "duree": 12,
                    "effets": {"qualite_air": -25.0}
                }
            ]

            evt_data = random.choice(evenements_possibles)
            return EvenementEcologique(
                nom=evt_data["nom"],
                type_evenement=evt_data["type"],
                intensite=random.uniform(0.3, 0.8),
                duree=timedelta(hours=evt_data["duree"]),
                date_debut=datetime.now(),
                effets=evt_data["effets"],
                description=f"Événement {evt_data['type']} aléatoire"
            )

        return None

    def __repr__(self) -> str:
        temp = self.conditions["temperature"].valeur
        hum = self.conditions["humidite"].valeur
        events = len(self.evenements_actifs)
        return f"Environnement(temp={temp:.1f}°C, hum={hum:.1f}%, events={events}, saison='{self.saison_courante}')"