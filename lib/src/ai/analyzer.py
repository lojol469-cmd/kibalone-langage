# 🧠 Intelligent Analyzer - Analyseur Intelligent

"""Analyseur intelligent pour l'orchestration de l'écosystème

L'analyseur intelligent fournit :
- Analyse de situation globale
- Évaluation des risques et opportunités
- Recommandations d'adaptation
- Prédiction de comportements
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from ..shared.config import Config
from ..shared.logger import get_logger

@dataclass
class AnalyseSituation:
    """Résultat d'une analyse de situation"""
    score_global: float
    facteurs_risques: List[str] = field(default_factory=list)
    opportunites: List[str] = field(default_factory=list)
    recommandations: List[str] = field(default_factory=list)
    predictions: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ProfilCellule:
    """Profil d'analyse d'une cellule"""
    nom: str
    type_cellule: str
    score_sante: float
    score_adaptation: float
    facteurs_stress: List[str] = field(default_factory=list)
    capacites_adaptation: List[str] = field(default_factory=list)

class IntelligentAnalyzer:
    """Analyseur intelligent pour l'écosystème KIBALI"""

    def __init__(self, config: Optional[Config] = None):
        """Initialise l'analyseur intelligent

        Args:
            config: Configuration optionnelle
        """
        self.config = config or Config()
        self.logger = get_logger("IA.IntelligentAnalyzer")

        # Seuils d'analyse
        self.seuils = {
            "sante_critique": 30.0,
            "stress_eleve": 70.0,
            "adaptation_faible": 40.0,
            "risque_global": 60.0
        }

        # Historique des analyses
        self.historique_analyses: List[AnalyseSituation] = []

        self.logger.info("IntelligentAnalyzer initialisé")

    def analyser_situation(self,
                          donnees_cellules: Dict[str, Dict[str, Any]],
                          donnees_environnement: Dict[str, Dict[str, Any]]) -> AnalyseSituation:
        """Analyse la situation globale de l'écosystème

        Args:
            donnees_cellules: Données des cellules {nom: {sante: float, stress: float, ...}}
            donnees_environnement: Données environnementales {nom: {valeur: float, ...}}

        Returns:
            AnalyseSituation: Analyse complète de la situation
        """
        try:
            # Analyse des cellules
            profils_cellules = self._analyser_cellules(donnees_cellules)

            # Analyse environnementale
            analyse_environnement = self._analyser_environnement(donnees_environnement)

            # Calcul du score global
            score_global = self._calculer_score_global(profils_cellules, analyse_environnement)

            # Identification des facteurs de risque
            facteurs_risques = self._identifier_risques(profils_cellules, analyse_environnement)

            # Identification des opportunités
            opportunites = self._identifier_opportunites(profils_cellules, analyse_environnement)

            # Génération de recommandations
            recommandations = self._generer_recommandations(score_global, facteurs_risques, opportunites)

            # Prédictions
            predictions = self._faire_predictions(profils_cellules, analyse_environnement)

            analyse = AnalyseSituation(
                score_global=score_global,
                facteurs_risques=facteurs_risques,
                opportunites=opportunites,
                recommandations=recommandations,
                predictions=predictions
            )

            # Archivage
            self.historique_analyses.append(analyse)

            self.logger.debug(f"Analyse effectuée: score_global={score_global:.2f}")
            return analyse

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse: {e}")
            return AnalyseSituation(score_global=0.5)

    def analyser_cellule(self, donnees_cellule: Dict[str, Any]) -> ProfilCellule:
        """Analyse une cellule spécifique

        Args:
            donnees_cellule: Données de la cellule

        Returns:
            ProfilCellule: Profil d'analyse de la cellule
        """
        try:
            nom = donnees_cellule.get("nom", "inconnu")
            type_cellule = donnees_cellule.get("type", "vegetal")

            # Évaluation de la santé
            sante = donnees_cellule.get("sante", 100)
            stress = donnees_cellule.get("stress", 0)
            age = donnees_cellule.get("age", 0)

            score_sante = self._evaluer_sante(sante, stress, age)

            # Évaluation de l'adaptation
            score_adaptation = self._evaluer_adaptation(donnees_cellule)

            # Facteurs de stress
            facteurs_stress = self._identifier_stress_cellule(donnees_cellule)

            # Capacités d'adaptation
            capacites = self._identifier_capacites_adaptation(type_cellule, donnees_cellule)

            profil = ProfilCellule(
                nom=nom,
                type_cellule=type_cellule,
                score_sante=score_sante,
                score_adaptation=score_adaptation,
                facteurs_stress=facteurs_stress,
                capacites_adaptation=capacites
            )

            return profil

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de cellule: {e}")
            return ProfilCellule(
                nom="erreur",
                type_cellule="inconnu",
                score_sante=0.0,
                score_adaptation=0.0
            )

    def predire_evolution(self, situation_actuelle: AnalyseSituation, horizon: int = 10) -> Dict[str, Any]:
        """Prédit l'évolution future de l'écosystème

        Args:
            situation_actuelle: Situation actuelle
            horizon: Horizon de prédiction en cycles

        Returns:
            Dict: Prédictions d'évolution
        """
        try:
            predictions = {
                "score_global_futur": situation_actuelle.score_global,
                "risques_potentiels": [],
                "opportunites_futures": [],
                "recommandations_preventives": [],
                "horizon": horizon
            }

            # Analyse des tendances
            if len(self.historique_analyses) >= 3:
                tendance = self._analyser_tendance()
                predictions["score_global_futur"] = max(0.0, min(1.0,
                    situation_actuelle.score_global + tendance * horizon * 0.1))

            # Prédictions basées sur les facteurs actuels
            if situation_actuelle.score_global < 0.4:
                predictions["risques_potentiels"].extend([
                    "déclin_generalisé",
                    "perte_biodiversite",
                    "instabilite_ecosysteme"
                ])
            elif situation_actuelle.score_global > 0.8:
                predictions["opportunites_futures"].extend([
                    "expansion_population",
                    "diversification_adaptations",
                    "stabilite_long_terme"
                ])

            return predictions

        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {e}")
            return {"erreur": str(e)}

    def optimiser_strategie(self, strategie_actuelle: str, analyse: AnalyseSituation) -> str:
        """Optimise la stratégie d'orchestration

        Args:
            strategie_actuelle: Stratégie actuelle
            analyse: Analyse de situation

        Returns:
            str: Nouvelle stratégie optimisée
        """
        try:
            strategies = {
                "conservateur": {"seuil_adaptation": 0.8, "priorite": "stabilite"},
                "equilibre": {"seuil_adaptation": 0.6, "priorite": "adaptation"},
                "dynamique": {"seuil_adaptation": 0.4, "priorite": "innovation"}
            }

            # Sélection basée sur le score global
            if analyse.score_global < 0.4:
                return "conservateur"
            elif analyse.score_global > 0.7:
                return "dynamique"
            else:
                return "equilibre"

        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation de stratégie: {e}")
            return strategie_actuelle

    # Méthodes privées d'analyse
    def _analyser_cellules(self, donnees_cellules: Dict[str, Dict[str, Any]]) -> List[ProfilCellule]:
        """Analyse l'ensemble des cellules"""
        profils = []
        for nom, donnees in donnees_cellules.items():
            donnees["nom"] = nom
            profil = self.analyser_cellule(donnees)
            profils.append(profil)
        return profils

    def _analyser_environnement(self, donnees_environnement: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Analyse les conditions environnementales"""
        analyse = {
            "stress_thermique": 0.0,
            "stress_hydrique": 0.0,
            "qualite_air": 1.0,
            "disponibilite_ressources": 1.0
        }

        # Analyse température
        if "temperature" in donnees_environnement:
            temp = donnees_environnement["temperature"]["valeur"]
            if temp < 5 or temp > 35:
                analyse["stress_thermique"] = min(1.0, abs(temp - 20) / 20)

        # Analyse humidité
        if "humidite" in donnees_environnement:
            hum = donnees_environnement["humidite"]["valeur"]
            if hum < 30 or hum > 90:
                analyse["stress_hydrique"] = min(1.0, abs(hum - 60) / 60)

        return analyse

    def _calculer_score_global(self, profils_cellules: List[ProfilCellule],
                              analyse_environnement: Dict[str, float]) -> float:
        """Calcule le score global de l'écosystème"""
        if not profils_cellules:
            return 0.0

        # Score moyen des cellules
        score_cellules = statistics.mean(p.score_sante * 0.6 + p.score_adaptation * 0.4
                                       for p in profils_cellules)

        # Impact environnemental
        stress_env = (analyse_environnement["stress_thermique"] +
                     analyse_environnement["stress_hydrique"]) / 2

        # Score global pondéré
        score_global = score_cellules * 0.7 - stress_env * 0.3
        return max(0.0, min(1.0, score_global))

    def _identifier_risques(self, profils_cellules: List[ProfilCellule],
                           analyse_environnement: Dict[str, float]) -> List[str]:
        """Identifie les facteurs de risque"""
        risques = []

        # Risques cellulaires
        cellules_critiques = [p for p in profils_cellules if p.score_sante < self.seuils["sante_critique"]]
        if len(cellules_critiques) > len(profils_cellules) * 0.3:
            risques.append("population_faible")

        cellules_stressees = [p for p in profils_cellules if any("stress" in f for f in p.facteurs_stress)]
        if len(cellules_stressees) > len(profils_cellules) * 0.5:
            risques.append("stress_generalise")

        # Risques environnementaux
        if analyse_environnement["stress_thermique"] > 0.7:
            risques.append("stress_thermique_eleve")
        if analyse_environnement["stress_hydrique"] > 0.7:
            risques.append("stress_hydrique_eleve")

        return risques

    def _identifier_opportunites(self, profils_cellules: List[ProfilCellule],
                                analyse_environnement: Dict[str, float]) -> List[str]:
        """Identifie les opportunités"""
        opportunites = []

        # Opportunités cellulaires
        cellules_adaptees = [p for p in profils_cellules if p.score_adaptation > 0.7]
        if len(cellules_adaptees) > len(profils_cellules) * 0.4:
            opportunites.append("base_adaptee_solide")

        # Opportunités environnementales
        if analyse_environnement["stress_thermique"] < 0.3 and analyse_environnement["stress_hydrique"] < 0.3:
            opportunites.append("conditions_favorables")

        return opportunites

    def _generer_recommandations(self, score_global: float, risques: List[str],
                                opportunites: List[str]) -> List[str]:
        """Génère des recommandations d'action"""
        recommandations = []

        if score_global < 0.4:
            recommandations.extend([
                "activer_protocoles_urgence",
                "reduire_activite_non_essentielle",
                "renforcer_defenses_collectives"
            ])
        elif score_global > 0.8:
            recommandations.extend([
                "explorer_nouvelles_adaptations",
                "optimiser_ressources",
                "developper_capacites_evolutives"
            ])

        # Recommandations spécifiques aux risques
        for risque in risques:
            if risque == "population_faible":
                recommandations.append("favoriser_reproduction")
            elif risque == "stress_generalise":
                recommandations.append("implementer_strategies_relaxation")

        return recommandations

    def _faire_predictions(self, profils_cellules: List[ProfilCellule],
                          analyse_environnement: Dict[str, float]) -> Dict[str, float]:
        """Fait des prédictions sur l'évolution"""
        predictions = {}

        # Prédiction de santé globale
        sante_moyenne = statistics.mean(p.score_sante for p in profils_cellules)
        predictions["sante_future"] = max(0.0, min(1.0, sante_moyenne - 0.05))  # Déclin naturel

        # Prédiction d'adaptation
        adaptation_moyenne = statistics.mean(p.score_adaptation for p in profils_cellules)
        predictions["adaptation_future"] = min(1.0, adaptation_moyenne + 0.02)  # Amélioration progressive

        return predictions

    # Méthodes privées d'évaluation
    def _evaluer_sante(self, sante: float, stress: float, age: int) -> float:
        """Évalue le score de santé d'une cellule"""
        score_base = sante / 100.0
        penalite_stress = (stress / 100.0) * 0.3
        penalite_age = min(0.2, age / 500.0)  # Vieillissement

        return max(0.0, min(1.0, score_base - penalite_stress - penalite_age))

    def _evaluer_adaptation(self, donnees_cellule: Dict[str, Any]) -> float:
        """Évalue le score d'adaptation d'une cellule"""
        # Évaluation simplifiée basée sur l'âge et l'état
        age = donnees_cellule.get("age", 0)
        sante = donnees_cellule.get("sante", 100)

        # Les cellules plus âgées ont généralement plus d'adaptations
        score_age = min(1.0, age / 100.0)
        score_sante = sante / 100.0

        return (score_age * 0.4) + (score_sante * 0.6)

    def _identifier_stress_cellule(self, donnees_cellule: Dict[str, Any]) -> List[str]:
        """Identifie les facteurs de stress d'une cellule"""
        facteurs = []

        stress = donnees_cellule.get("stress", 0)
        sante = donnees_cellule.get("sante", 100)

        if stress > 70:
            facteurs.append("stress_eleve")
        if sante < 50:
            facteurs.append("sante_faible")
        if donnees_cellule.get("age", 0) > 200:
            facteurs.append("vieillissement")

        return facteurs

    def _identifier_capacites_adaptation(self, type_cellule: str,
                                        donnees_cellule: Dict[str, Any]) -> List[str]:
        """Identifie les capacités d'adaptation d'une cellule"""
        capacites = []

        # Capacités par type
        capacites_base = {
            "vegetal": ["photosynthese", "enracinement", "reproduction_vegetative"],
            "animal": ["mobilite", "chasse", "social"],
            "microbien": ["resistance_extremes", "decomposition", "symbiose"]
        }

        if type_cellule in capacites_base:
            capacites.extend(capacites_base[type_cellule])

        # Capacités basées sur l'âge (expérience)
        if donnees_cellule.get("age", 0) > 50:
            capacites.append("experience")

        return capacites

    def _analyser_tendance(self) -> float:
        """Analyse la tendance des scores globaux"""
        if len(self.historique_analyses) < 2:
            return 0.0

        scores_recents = [a.score_global for a in self.historique_analyses[-5:]]
        if len(scores_recents) < 2:
            return 0.0

        # Tendance linéaire simple
        n = len(scores_recents)
        tendance = (scores_recents[-1] - scores_recents[0]) / (n - 1)

        return tendance