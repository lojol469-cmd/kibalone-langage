# 🌍 Écosystème - Gestionnaire d'Environnement Biologique

"""Gestionnaire d'écosystème KIBALI

L'écosystème gère l'ensemble des cellules et leurs interactions
dans un environnement dynamique et évolutif.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .cellule import Cellule, EtatInterne
from .environment import Environnement

@dataclass
class StatistiquesEcosysteme:
    """Statistiques globales de l'écosystème"""
    nombre_cellules: int = 0
    types_cellulaires: Dict[str, int] = field(default_factory=dict)
    stress_moyen: float = 0.0
    sante_moyenne: float = 100.0
    age_moyen: float = 0.0
    biodiversite: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class Ecosysteme:
    """Gestionnaire principal de l'écosystème KIBALI

    L'écosystème coordonne :
    - Les cellules biologiques autonomes
    - Les conditions environnementales
    - Les interactions inter-cellulaires
    - L'évolution globale du système
    """

    def __init__(self, nom: str = "EcosystemeKIBALI", max_cellules: int = 100):
        """Initialise un nouvel écosystème

        Args:
            nom: Nom de l'écosystème
            max_cellules: Nombre maximum de cellules
        """
        self.nom = nom
        self.max_cellules = max_cellules

        # Composants principaux
        self.cellules: List[Cellule] = []
        self.environnement = Environnement()
        self.historique_statistiques: List[StatistiquesEcosysteme] = []

        # Métriques
        self.cycle_courant = 0
        self.total_interactions = 0
        self.evenements_speciaux = []

        # Logger
        self.logger = logging.getLogger(f"Ecosysteme.{nom}")
        self.logger.info(f"Écosystème '{nom}' initialisé (max {max_cellules} cellules)")

    def ajouter_cellule(self, cellule: Cellule) -> bool:
        """Ajoute une cellule à l'écosystème

        Args:
            cellule: La cellule à ajouter

        Returns:
            bool: True si ajout réussi, False sinon
        """
        if len(self.cellules) >= self.max_cellules:
            self.logger.warning(f"Nombre maximum de cellules atteint ({self.max_cellules})")
            return False

        if cellule in self.cellules:
            self.logger.warning(f"Cellule {cellule.nom} déjà présente dans l'écosystème")
            return False

        self.cellules.append(cellule)
        self.logger.info(f"Cellule {cellule.nom} ajoutée à l'écosystème")
        return True

    def retirer_cellule(self, cellule: Cellule) -> bool:
        """Retire une cellule de l'écosystème

        Args:
            cellule: La cellule à retirer

        Returns:
            bool: True si retrait réussi, False sinon
        """
        if cellule not in self.cellules:
            self.logger.warning(f"Cellule {cellule.nom} non trouvée dans l'écosystème")
            return False

        self.cellules.remove(cellule)
        self.logger.info(f"Cellule {cellule.nom} retirée de l'écosystème")
        return True

    def definir_environnement(self, environnement: Environnement) -> None:
        """Définit les conditions environnementales

        Args:
            environnement: Nouvelles conditions environnementales
        """
        self.environnement = environnement
        self.logger.info("Nouvelles conditions environnementales définies")

        # Notification aux cellules du changement
        self._notifier_changement_environnement()

    def cycle_evolution(self) -> Dict[str, Any]:
        """Exécute un cycle d'évolution de l'écosystème

        Returns:
            Dict: Résultats du cycle d'évolution
        """
        self.cycle_courant += 1

        try:
            # Évolution de l'environnement
            evolution_env = self.environnement.evoluer()

            # Évolution des cellules
            evolution_cellules = self._evoluer_cellules()

            # Interactions inter-cellulaires
            interactions = self._gerer_interactions()

            # Mise à jour des statistiques
            statistiques = self._calculer_statistiques()
            self.historique_statistiques.append(statistiques)

            # Nettoyage des cellules mortes
            cellules_mortes = self._nettoyer_cellules_mortes()

            resultats = {
                "cycle": self.cycle_courant,
                "evolution_environnement": evolution_env,
                "evolution_cellules": evolution_cellules,
                "interactions": interactions,
                "statistiques": statistiques,
                "cellules_mortes": cellules_mortes,
                "timestamp": datetime.now()
            }

            self.logger.info(f"Cycle d'évolution {self.cycle_courant} terminé")
            return resultats

        except Exception as e:
            self.logger.error(f"Erreur lors du cycle d'évolution: {e}")
            return {
                "cycle": self.cycle_courant,
                "erreur": str(e),
                "timestamp": datetime.now()
            }

    def obtenir_cellule(self, nom: str) -> Optional[Cellule]:
        """Récupère une cellule par son nom

        Args:
            nom: Nom de la cellule

        Returns:
            Cellule or None: La cellule trouvée ou None
        """
        for cellule in self.cellules:
            if cellule.nom == nom:
                return cellule
        return None

    def lister_cellules(self, type_filtre: Optional[str] = None) -> List[Cellule]:
        """Liste les cellules avec possibilité de filtrage

        Args:
            type_filtre: Type de cellules à filtrer (optionnel)

        Returns:
            List[Cellule]: Liste des cellules filtrées
        """
        if type_filtre is None:
            return self.cellules.copy()

        return [c for c in self.cellules if c.type_cellule == type_filtre]

    def calculer_stress_global(self) -> float:
        """Calcule le niveau de stress global de l'écosystème

        Returns:
            float: Niveau de stress global (0.0 à 1.0)
        """
        if not self.cellules:
            return 0.0

        stress_total = sum(
            cellule.etats_internes.get("stress", EtatInterne(0, "%")).valeur
            for cellule in self.cellules
        )

        return min(1.0, stress_total / len(self.cellules))

    def calculer_biodiversite(self) -> float:
        """Calcule l'indice de biodiversité de l'écosystème

        Returns:
            float: Indice de biodiversité (0.0 à 1.0)
        """
        if not self.cellules:
            return 0.0

        types_presents = len(set(c.type_cellule for c in self.cellules))
        types_possibles = 10  # Nombre théorique de types possibles

        # Formule de biodiversité simplifiée
        biodiversite = types_presents / types_possibles
        return min(1.0, biodiversite)

    def exporter_etat(self) -> Dict[str, Any]:
        """Exporte l'état complet de l'écosystème

        Returns:
            Dict: État sérialisé de l'écosystème
        """
        return {
            "nom": self.nom,
            "cycle_courant": self.cycle_courant,
            "environnement": self.environnement.exporter_etat(),
            "cellules": [cellule.exporter_etat() for cellule in self.cellules],
            "statistiques": self._calculer_statistiques(),
            "timestamp": datetime.now()
        }

    def importer_etat(self, etat: Dict[str, Any]) -> None:
        """Importe un état sauvegardé

        Args:
            etat: État sérialisé à importer
        """
        try:
            self.nom = etat.get("nom", self.nom)
            self.cycle_courant = etat.get("cycle_courant", 0)

            # Import environnement
            if "environnement" in etat:
                self.environnement.importer_etat(etat["environnement"])

            # Import cellules
            self.cellules = []
            for cellule_data in etat.get("cellules", []):
                cellule = Cellule.from_dict(cellule_data)
                self.cellules.append(cellule)

            self.logger.info(f"État importé: {len(self.cellules)} cellules")

        except Exception as e:
            self.logger.error(f"Erreur lors de l'import d'état: {e}")
            raise

    # Méthodes privées
    def _notifier_changement_environnement(self) -> None:
        """Notifie toutes les cellules d'un changement environnemental"""
        for cellule in self.cellules:
            try:
                cellule.percevoir_environnement(self.environnement)
            except Exception as e:
                self.logger.error(f"Erreur notification cellule {cellule.nom}: {e}")

    def _evoluer_cellules(self) -> List[Dict[str, Any]]:
        """Fait évoluer toutes les cellules"""
        resultats = []
        for cellule in self.cellules:
            try:
                evolution = cellule.evoluer()
                resultats.append({
                    "cellule": cellule.nom,
                    "evolution": evolution,
                    "succes": True
                })
            except Exception as e:
                resultats.append({
                    "cellule": cellule.nom,
                    "evolution": None,
                    "succes": False,
                    "erreur": str(e)
                })
        return resultats

    def _gerer_interactions(self) -> List[Dict[str, Any]]:
        """Gère les interactions inter-cellulaires"""
        interactions = []

        # Interactions par paires (simplifié)
        for i, cellule1 in enumerate(self.cellules):
            for cellule2 in self.cellules[i+1:]:
                try:
                    interaction = cellule1.interagir_avec(cellule2)
                    if interaction:
                        interactions.append({
                            "cellules": [cellule1.nom, cellule2.nom],
                            "interaction": interaction,
                            "succes": True
                        })
                        self.total_interactions += 1
                except Exception as e:
                    interactions.append({
                        "cellules": [cellule1.nom, cellule2.nom],
                        "interaction": None,
                        "succes": False,
                        "erreur": str(e)
                    })

        return interactions

    def _calculer_statistiques(self) -> StatistiquesEcosysteme:
        """Calcule les statistiques actuelles"""
        if not self.cellules:
            return StatistiquesEcosysteme()

        # Comptage des types
        types_count = {}
        for cellule in self.cellules:
            types_count[cellule.type_cellule] = types_count.get(cellule.type_cellule, 0) + 1

        # Métriques moyennes
        stress_total = sum(c.etats_internes.get("stress", EtatInterne(0, "%")).valeur for c in self.cellules)
        sante_total = sum(c.etats_internes.get("sante", EtatInterne(100, "%")).valeur for c in self.cellules)
        age_total = sum(c.age for c in self.cellules)

        return StatistiquesEcosysteme(
            nombre_cellules=len(self.cellules),
            types_cellulaires=types_count,
            stress_moyen=stress_total / len(self.cellules),
            sante_moyenne=sante_total / len(self.cellules),
            age_moyen=age_total / len(self.cellules),
            biodiversite=self.calculer_biodiversite()
        )

    def _nettoyer_cellules_mortes(self) -> int:
        """Nettoie les cellules mortes et retourne le nombre supprimé"""
        cellules_avant = len(self.cellules)
        self.cellules = [
            c for c in self.cellules
            if c.etats_internes.get("sante", EtatInterne(100, "%")).valeur > 0
        ]
        cellules_mortes = cellules_avant - len(self.cellules)

        if cellules_mortes > 0:
            self.logger.info(f"{cellules_mortes} cellules mortes supprimées")

        return cellules_mortes

    def __len__(self) -> int:
        """Retourne le nombre de cellules"""
        return len(self.cellules)

    def __iter__(self):
        """Itère sur les cellules"""
        return iter(self.cellules)

    def __repr__(self) -> str:
        return f"Ecosysteme(nom='{self.nom}', cellules={len(self.cellules)}, cycle={self.cycle_courant})"