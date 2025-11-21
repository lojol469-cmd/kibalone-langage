# 🧬 Cellule - Unité Biologique Autonome

"""Classe Cellule - Unité fondamentale de l'écosystème KIBALI

Une cellule représente une entité biologique autonome capable de :
- Percer l'environnement
- Prendre des décisions
- S'adapter aux conditions
- Interagir avec d'autres cellules
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging

from ..shared.config import Config

@dataclass
class EtatInterne:
    """Représente un état interne d'une cellule"""
    valeur: Union[int, float]
    unite: str
    minimum: Union[int, float] = 0
    maximum: Union[int, float] = 100
    description: str = ""

    def __post_init__(self):
        """Validation après initialisation"""
        if not (self.minimum <= self.valeur <= self.maximum):
            raise ValueError(f"Valeur {self.valeur} hors limites [{self.minimum}, {self.maximum}]")

    def modifier(self, delta: Union[int, float]) -> bool:
        """Modifie la valeur de l'état"""
        nouvelle_valeur = self.valeur + delta
        if self.minimum <= nouvelle_valeur <= self.maximum:
            self.valeur = nouvelle_valeur
            return True
        return False

@dataclass
class ObjetPhysique:
    """Représente un objet physique d'une cellule"""
    type_objet: str
    etat: str
    proprietes: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

class Cellule(ABC):
    """Classe de base pour toutes les cellules KIBALI

    Une cellule est une entité autonome qui :
    - Possède des états internes (santé, stress, âge, etc.)
    - A des objets physiques (organes, structures)
    - Peut percevoir son environnement
    - Prend des décisions d'adaptation
    - Interagit avec d'autres cellules
    """

    def __init__(self,
                 nom: str,
                 type_cellule: str,
                 etats_internes: Optional[Dict[str, EtatInterne]] = None,
                 objets_physiques: Optional[Dict[str, ObjetPhysique]] = None,
                 config: Optional[Config] = None):
        """Initialise une nouvelle cellule

        Args:
            nom: Nom unique de la cellule
            type_cellule: Type de cellule (arbre, fleur, animal, etc.)
            etats_internes: États internes initiaux
            objets_physiques: Objets physiques initiaux
            config: Configuration optionnelle
        """
        self.nom = nom
        self.type_cellule = type_cellule
        self.age = 0
        self.date_creation = datetime.now()

        # États internes par défaut
        self.etats_internes: Dict[str, EtatInterne] = {
            "sante": EtatInterne(100, "%", 0, 100, "État de santé général"),
            "stress": EtatInterne(0, "%", 0, 100, "Niveau de stress"),
            "energie": EtatInterne(100, "%", 0, 100, "Réserves d'énergie")
        }

        # Mise à jour avec les états fournis
        if etats_internes:
            self.etats_internes.update(etats_internes)

        # Objets physiques
        self.objets_physiques: Dict[str, ObjetPhysique] = objets_physiques or {}

        # Historique et mémoire
        self.historique_actions: List[Dict[str, Any]] = []
        self.memoires: List[Dict[str, Any]] = []

        # Configuration
        self.config = config or Config()

        # Logger
        self.logger = logging.getLogger(f"Cellule.{nom}")

        self.logger.info(f"Cellule {nom} ({type_cellule}) créée")

    @abstractmethod
    def percevoir_environnement(self, environnement: Any) -> Dict[str, float]:
        """Perçoit l'environnement et retourne les perceptions

        Args:
            environnement: Conditions environnementales

        Returns:
            Dict: Perceptions sous forme de valeurs numériques
        """
        pass

    @abstractmethod
    def adapter_autonomously(self, perceptions: Dict[str, float]) -> List[str]:
        """Adapte la cellule de manière autonome

        Args:
            perceptions: Perceptions environnementales

        Returns:
            List[str]: Liste des adaptations appliquées
        """
        pass

    def evoluer(self) -> Dict[str, Any]:
        """Fait évoluer la cellule naturellement

        Returns:
            Dict: Résultats de l'évolution
        """
        self.age += 1

        # Vieillissement naturel
        vieillissement = self._calculer_vieillissement()
        self.etats_internes["sante"].modifier(-vieillissement)

        # Régénération
        regeneration = self._calculer_regeneration()
        self.etats_internes["sante"].modifier(regeneration)

        # Gestion de l'énergie
        consommation = self._calculer_consommation_energetique()
        self.etats_internes["energie"].modifier(-consommation)

        # Vérification de survie
        survie = self._verifier_survie()

        resultats = {
            "age": self.age,
            "vieillissement": vieillissement,
            "regeneration": regeneration,
            "consommation_energie": consommation,
            "survie": survie,
            "timestamp": datetime.now()
        }

        self.historique_actions.append({
            "action": "evolution_naturelle",
            "resultats": resultats
        })

        return resultats

    def interagir_avec(self, autre_cellule: 'Cellule') -> Optional[Dict[str, Any]]:
        """Interagit avec une autre cellule

        Args:
            autre_cellule: La cellule cible de l'interaction

        Returns:
            Dict or None: Résultats de l'interaction ou None si pas d'interaction
        """
        # Distance et proximité (simplifié)
        distance = self._calculer_distance(autre_cellule)

        if distance > self._distance_interaction_max():
            return None  # Trop loin pour interagir

        # Type d'interaction basé sur les types cellulaires
        interaction = self._determiner_type_interaction(autre_cellule)

        if interaction:
            resultats = self._executer_interaction(autre_cellule, interaction)

            # Enregistrement
            self.historique_actions.append({
                "action": "interaction",
                "cible": autre_cellule.nom,
                "type": interaction,
                "resultats": resultats
            })

            return resultats

        return None

    def appliquer_adaptation(self, type_adaptation: str, parametres: Optional[Dict[str, Any]] = None) -> bool:
        """Applique une adaptation spécifique

        Args:
            type_adaptation: Type d'adaptation à appliquer
            parametres: Paramètres optionnels

        Returns:
            bool: True si adaptation réussie
        """
        try:
            methode_adaptation = getattr(self, f"_adapter_{type_adaptation}", None)
            if methode_adaptation:
                succes = methode_adaptation(parametres or {})
                if succes:
                    self.historique_actions.append({
                        "action": "adaptation",
                        "type": type_adaptation,
                        "parametres": parametres,
                        "succes": True
                    })
                return succes
            else:
                self.logger.warning(f"Adaptation {type_adaptation} non supportée")
                return False

        except Exception as e:
            self.logger.error(f"Erreur lors de l'adaptation {type_adaptation}: {e}")
            return False

    def appliquer_action_urgence(self, action: str) -> Dict[str, Any]:
        """Applique une action d'urgence

        Args:
            action: Action d'urgence à appliquer

        Returns:
            Dict: Résultats de l'action
        """
        try:
            # Actions d'urgence prioritaires
            actions_urgence = {
                "activer_defenses": self._urgence_defenses,
                "economiser_energie": self._urgence_energie,
                "protege_vital": self._urgence_protection
            }

            if action in actions_urgence:
                resultats = actions_urgence[action]()
                self.historique_actions.append({
                    "action": "urgence",
                    "type": action,
                    "resultats": resultats
                })
                return {"succes": True, "resultats": resultats}
            else:
                return {"succes": False, "erreur": "action_non_supportee"}

        except Exception as e:
            self.logger.error(f"Erreur action urgence {action}: {e}")
            return {"succes": False, "erreur": str(e)}

    def memoriser(self, information: Dict[str, Any]) -> None:
        """Mémorise une information importante

        Args:
            information: Information à mémoriser
        """
        memoire = {
            "timestamp": datetime.now(),
            "information": information
        }
        self.memoires.append(memoire)

        # Limitation de la mémoire
        if len(self.memoires) > 100:  # Taille max de mémoire
            self.memoires.pop(0)

    def rappeler(self, critere: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rappelle des souvenirs basés sur un critère

        Args:
            critere: Critères de recherche dans la mémoire

        Returns:
            List[Dict]: Souvenirs correspondants
        """
        souvenirs = []
        for memoire in self.memoires:
            correspond = True
            for key, value in critere.items():
                if key not in memoire["information"] or memoire["information"][key] != value:
                    correspond = False
                    break
            if correspond:
                souvenirs.append(memoire)

        return souvenirs

    def exporter_etat(self) -> Dict[str, Any]:
        """Exporte l'état complet de la cellule

        Returns:
            Dict: État sérialisé
        """
        return {
            "nom": self.nom,
            "type_cellule": self.type_cellule,
            "age": self.age,
            "date_creation": self.date_creation.isoformat(),
            "etats_internes": {
                nom: {
                    "valeur": etat.valeur,
                    "unite": etat.unite,
                    "minimum": etat.minimum,
                    "maximum": etat.maximum,
                    "description": etat.description
                }
                for nom, etat in self.etats_internes.items()
            },
            "objets_physiques": {
                nom: {
                    "type_objet": obj.type_objet,
                    "etat": obj.etat,
                    "proprietes": obj.proprietes,
                    "description": obj.description
                }
                for nom, obj in self.objets_physiques.items()
            },
            "historique_actions": self.historique_actions[-50:],  # Derniers 50 éléments
            "memoires": self.memoires[-20:]  # Derniers 20 souvenirs
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cellule':
        """Crée une cellule à partir d'un dictionnaire

        Args:
            data: Données sérialisées

        Returns:
            Cellule: Nouvelle instance
        """
        # Reconstruction des états internes
        etats_internes = {}
        for nom, etat_data in data.get("etats_internes", {}).items():
            etats_internes[nom] = EtatInterne(**etat_data)

        # Reconstruction des objets physiques
        objets_physiques = {}
        for nom, obj_data in data.get("objets_physiques", {}).items():
            objets_physiques[nom] = ObjetPhysique(**obj_data)

        # Création de l'instance
        instance = cls(
            nom=data["nom"],
            type_cellule=data["type_cellule"],
            etats_internes=etats_internes,
            objets_physiques=objets_physiques
        )

        # Restauration des attributs
        instance.age = data.get("age", 0)
        instance.date_creation = datetime.fromisoformat(data["date_creation"])
        instance.historique_actions = data.get("historique_actions", [])
        instance.memoires = data.get("memoires", [])

        return instance

    # Méthodes privées d'évolution
    def _calculer_vieillissement(self) -> float:
        """Calcule le vieillissement naturel"""
        # Vieillissement accéléré après 50 ans
        if self.age < 50:
            return 0.5
        else:
            return 1.0 + (self.age - 50) * 0.1

    def _calculer_regeneration(self) -> float:
        """Calcule la capacité de régénération"""
        sante = self.etats_internes["sante"].valeur
        energie = self.etats_internes["energie"].valeur

        # Régénération basée sur la santé et l'énergie
        regeneration_base = 0.3
        bonus_sante = (sante / 100) * 0.4
        bonus_energie = (energie / 100) * 0.3

        return regeneration_base + bonus_sante + bonus_energie

    def _calculer_consommation_energetique(self) -> float:
        """Calcule la consommation énergétique"""
        # Consommation de base
        consommation = 2.0

        # Bonus selon l'activité
        stress = self.etats_internes["stress"].valeur
        consommation += (stress / 100) * 3.0  # Stress augmente la consommation

        return consommation

    def _verifier_survie(self) -> bool:
        """Vérifie si la cellule survit"""
        sante = self.etats_internes["sante"].valeur
        energie = self.etats_internes["energie"].valeur

        # Conditions de survie
        return sante > 0 and energie > 0

    # Méthodes privées d'interaction
    def _calculer_distance(self, autre_cellule: 'Cellule') -> float:
        """Calcule la distance avec une autre cellule"""
        # Distance simplifiée (à étendre avec coordonnées spatiales)
        return 1.0  # Distance unitaire par défaut

    def _distance_interaction_max(self) -> float:
        """Retourne la distance maximale d'interaction"""
        return 5.0

    def _determiner_type_interaction(self, autre_cellule: 'Cellule') -> Optional[str]:
        """Détermine le type d'interaction possible"""
        # Logique simplifiée d'interaction
        interactions_possibles = {
            ("vegetal", "animal"): "pollinisation",
            ("animal", "vegetal"): "consommation",
            ("vegetal", "vegetal"): "competition",
            ("animal", "animal"): "social"
        }

        cle = (self.type_cellule, autre_cellule.type_cellule)
        return interactions_possibles.get(cle)

    def _executer_interaction(self, autre_cellule: 'Cellule', type_interaction: str) -> Dict[str, Any]:
        """Exécute une interaction spécifique"""
        # Logique d'interaction simplifiée
        if type_interaction == "pollinisation":
            # Transfert d'énergie
            energie_transfer = min(10, autre_cellule.etats_internes["energie"].valeur)
            autre_cellule.etats_internes["energie"].modifier(-energie_transfer)
            self.etats_internes["energie"].modifier(energie_transfer)

            return {
                "type": "pollinisation",
                "energie_transfer": energie_transfer,
                "succes": True
            }

        return {"type": type_interaction, "succes": False}

    # Méthodes privées d'urgence
    def _urgence_defenses(self) -> Dict[str, Any]:
        """Active les défenses d'urgence"""
        self.etats_internes["stress"].valeur = min(100, self.etats_internes["stress"].valeur + 20)
        return {"defenses_activées": True, "stress_augmenté": 20}

    def _urgence_energie(self) -> Dict[str, Any]:
        """Économise l'énergie en urgence"""
        economie = self.etats_internes["energie"].valeur * 0.3
        self.etats_internes["energie"].modifier(-economie)
        return {"energie_economisee": economie}

    def _urgence_protection(self) -> Dict[str, Any]:
        """Protège les fonctions vitales"""
        sante_bonus = 10
        self.etats_internes["sante"].modifier(sante_bonus)
        return {"protection_activée": True, "sante_bonus": sante_bonus}

    # Méthodes d'adaptation génériques (peuvent être surchargées)
    def _adapter_resistance_thermique(self, parametres: Dict[str, Any]) -> bool:
        """Adapte la résistance thermique"""
        resistance = parametres.get("resistance", 1.2)
        # Logique d'adaptation simplifiée
        return True

    def _adapter_efficacite_photosynthetique(self, parametres: Dict[str, Any]) -> bool:
        """Adapte l'efficacité photosynthétique"""
        efficacite = parametres.get("efficacite", 1.1)
        # Logique d'adaptation simplifiée
        return True

    def __repr__(self) -> str:
        sante = self.etats_internes["sante"].valeur
        stress = self.etats_internes["stress"].valeur
        return f"Cellule(nom='{self.nom}', type='{self.type_cellule}', sante={sante}%, stress={stress}%, age={self.age})"