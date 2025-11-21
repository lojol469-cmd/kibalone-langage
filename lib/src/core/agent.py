# 🤖 Agent KIBALI - Orchestrateur Intelligent

"""Agent KIBALI - Le cerveau orchestrateur de l'écosystème

L'Agent KIBALI est l'intelligence centrale qui :
- Analyse l'état global de l'écosystème
- Orchestre les adaptations des cellules
- Gère les situations d'urgence
- Optimise les performances globales
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..shared.config import Config
from ..shared.logger import get_logger
from .ecosystem import Ecosysteme
from ..ai.analyzer import IntelligentAnalyzer

@dataclass
class Situation:
    """Représente l'état actuel de l'écosystème"""
    environnement: Dict[str, float]
    cellules: List[Dict[str, Any]]
    stress_global: float
    problemes_detectes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RapportCycle:
    """Rapport d'un cycle d'orchestration"""
    cycle_id: int
    adaptations: List[Dict[str, Any]] = field(default_factory=list)
    adaptations_urgence: List[Dict[str, Any]] = field(default_factory=list)
    stress_global: float = 0.0
    urgence_declaree: bool = False
    recommandations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class AgentKibali:
    """Agent intelligent orchestrateur de l'écosystème KIBALI

    L'agent analyse en continu l'état de l'écosystème et applique
    des stratégies d'adaptation pour maintenir l'équilibre.
    """

    def __init__(self,
                 strategie: str = "optimisation_adaptative",
                 seuils: Optional[Dict[str, float]] = None,
                 config: Optional[Config] = None):
        """Initialise l'agent KIBALI

        Args:
            strategie: Stratégie d'orchestration
            seuils: Seuils de décision personnalisés
            config: Configuration avancée
        """
        self.config = config or Config()
        self.logger = get_logger("AgentKibali")

        # Configuration des seuils
        self.seuils = {
            "adaptation": 0.7,
            "urgence": 0.8,
            "evolution": 0.9
        }
        if seuils:
            self.seuils.update(seuils)

        # État interne
        self.statut = "actif"
        self.strategie = strategie
        self.cellules_surveillees: List[Any] = []
        self.influences_appliquees = 0
        self.cycle_courant = 0

        # Composants IA
        self.analyseur_ia = IntelligentAnalyzer()

        # Historique
        self.historique_rapports: List[RapportCycle] = []
        self.derniere_situation: Optional[Situation] = None

        self.logger.info(f"Agent KIBALI initialisé avec stratégie: {strategie}")

    def connecter_ecosysteme(self, ecosysteme: Ecosysteme) -> None:
        """Connecte l'agent à un écosystème

        Args:
            ecosysteme: L'écosystème à gérer
        """
        self.ecosysteme = ecosysteme
        self.cellules_surveillees = ecosysteme.cellules.copy()
        self.logger.info(f"Connecté à écosystème avec {len(self.cellules_surveillees)} cellules")

    def analyser_situation(self) -> Situation:
        """Analyse l'état actuel de l'écosystème

        Returns:
            Situation: État détaillé de l'écosystème
        """
        try:
            # Collecte des données environnementales
            env_data = self._collecter_donnees_environnement()

            # Collecte des données cellulaires
            cellules_data = self._collecter_donnees_cellules()

            # Calcul du stress global
            stress_global = self._calculer_stress_global(env_data, cellules_data)

            # Détection des problèmes
            problemes = self._detecter_problemes(env_data, cellules_data)

            # Création de la situation
            situation = Situation(
                environnement=env_data,
                cellules=cellules_data,
                stress_global=stress_global,
                problemes_detectes=problemes
            )

            self.derniere_situation = situation
            self.logger.debug(f"Situation analysée: stress={stress_global:.2f}")

            return situation

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de situation: {e}")
            # Retour d'une situation par défaut en cas d'erreur
            return Situation(
                environnement={},
                cellules=[],
                stress_global=1.0,
                problemes_detectes=["erreur_analyse"]
            )

    def cycle_orchestration(self) -> RapportCycle:
        """Exécute un cycle complet d'orchestration

        Returns:
            RapportCycle: Rapport détaillé du cycle
        """
        self.cycle_courant += 1
        debut_cycle = datetime.now()

        try:
            # Analyse de la situation
            situation = self.analyser_situation()

            # Génération du rapport de base
            rapport = RapportCycle(
                cycle_id=self.cycle_courant,
                stress_global=situation.stress_global,
                timestamp=debut_cycle
            )

            # Vérification d'urgence
            if situation.stress_global >= self.seuils["urgence"]:
                rapport.urgence_declaree = True
                adaptations_urgence = self._appliquer_strategie_urgence(situation)
                rapport.adaptations_urgence = adaptations_urgence
                self.logger.warning(f"Urgence déclarée - {len(adaptations_urgence)} adaptations")

            # Adaptations normales
            elif situation.stress_global >= self.seuils["adaptation"]:
                adaptations = self._appliquer_adaptations_normales(situation)
                rapport.adaptations = adaptations
                self.logger.info(f"Adaptations appliquées: {len(adaptations)}")

            # Optimisation continue
            else:
                optimisations = self._optimiser_performance(situation)
                rapport.recommandations = optimisations
                self.logger.debug(f"Optimisations recommandées: {len(optimisations)}")

            # Mise à jour des métriques
            self.influences_appliquees += len(rapport.adaptations) + len(rapport.adaptations_urgence)

            # Archivage du rapport
            self.historique_rapports.append(rapport)

            duree_cycle = (datetime.now() - debut_cycle).total_seconds()
            self.logger.info(f"Cycle {self.cycle_courant} terminé en {duree_cycle:.2f}s")

            return rapport

        except Exception as e:
            self.logger.error(f"Erreur lors du cycle d'orchestration: {e}")
            # Retour d'un rapport d'erreur
            return RapportCycle(
                cycle_id=self.cycle_courant,
                stress_global=1.0,
                timestamp=debut_cycle
            )

    def influencer_cellule(self, cellule: Any) -> Dict[str, Any]:
        """Influence une cellule spécifique

        Args:
            cellule: La cellule à influencer

        Returns:
            Dict: Résultat de l'influence
        """
        try:
            # Analyse de la situation globale incluant cette cellule
            cellules_data = {cellule.nom: {"sante": {"valeur": cellule.etats_internes.get("sante", 100)},
                                         "stress": {"valeur": cellule.etats_internes.get("stress", 0)}}}
            env_data = {}
            if hasattr(self, 'ecosysteme') and self.ecosysteme.environnement:
                env_data = {
                    "temperature": {"valeur": getattr(self.ecosysteme.environnement, 'temperature', 20)},
                    "humidite": {"valeur": getattr(self.ecosysteme.environnement, 'humidite', 60)}
                }

            analyse = self.analyseur_ia.analyser_situation(cellules_data, env_data)

            # Application de l'adaptation basée sur l'analyse
            if analyse.score_global < 0.6:  # Seuil d'adaptation
                resultat = self._appliquer_adaptation(cellule, {"type_adaptation": "generique", "score_global": analyse.score_global})
                self.logger.debug(f"Adaptation appliquée à {cellule.nom}: score={analyse.score_global:.2f}")
                return resultat
            else:
                return {"action": "aucune_adaptation", "raison": "etat_optimal"}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'influence de {cellule.nom}: {e}")
            return {"action": "erreur", "details": str(e)}

    def influencer_environnement(self) -> Dict[str, Any]:
        """Influence les conditions environnementales

        Returns:
            Dict: Résultat de l'influence environnementale
        """
        try:
            if not hasattr(self, 'ecosysteme'):
                return {"action": "erreur", "raison": "ecosysteme_non_connecte"}

            # Analyse environnementale
            env_data = {
                "temperature": {"valeur": getattr(self.ecosysteme.environnement, 'temperature', 20)},
                "humidite": {"valeur": getattr(self.ecosysteme.environnement, 'humidite', 60)}
            }
            cellules_data = {}  # Vide pour focus environnement

            analyse_env = self.analyseur_ia.analyser_situation(cellules_data, env_data)

            # Application des modifications si nécessaire
            if analyse_env.score_global < 0.7:  # Seuil de modification
                modifications = self._appliquer_modifications_environnement({"score_global": analyse_env.score_global})
                self.logger.info(f"Modifications environnementales: score={analyse_env.score_global:.2f}")
                return {"action": "modifications_appliquees", "modifications": modifications}
            else:
                return {"action": "aucune_modification", "raison": "conditions_optimales"}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'influence environnementale: {e}")
            return {"action": "erreur", "details": str(e)}

    def reagir_urgence(self, type_urgence: str) -> Dict[str, Any]:
        """Gère une situation d'urgence

        Args:
            type_urgence: Type d'urgence détectée

        Returns:
            Dict: Actions d'urgence appliquées
        """
        self.logger.warning(f"Réaction d'urgence: {type_urgence}")

        actions_urgence = {
            "stress_thermique": ["activer_protection_thermique", "reduire_activite"],
            "secheresse": ["economiser_eau", "renforcer_racines"],
            "inondation": ["activer_flottation", "stocker_energie"],
            "tempete": ["ancrer_solide", "reduire_surface"],
            "maladie": ["activer_defenses", "isoler_infectes"]
        }

        actions = actions_urgence.get(type_urgence, ["protocole_standard"])

        # Application des actions d'urgence
        resultats = []
        for cellule in self.cellules_surveillees:
            for action in actions:
                try:
                    resultat = cellule.appliquer_action_urgence(action)
                    resultats.append({
                        "cellule": cellule.nom,
                        "action": action,
                        "succes": resultat.get("succes", False)
                    })
                except Exception as e:
                    resultats.append({
                        "cellule": cellule.nom,
                        "action": action,
                        "succes": False,
                        "erreur": str(e)
                    })

        return {
            "type_urgence": type_urgence,
            "actions_appliquees": actions,
            "resultats": resultats,
            "timestamp": datetime.now()
        }

    def apprendre_experience(self) -> None:
        """Apprend des expériences passées pour améliorer les futures décisions"""
        try:
            if len(self.historique_rapports) < 10:
                return  # Pas assez de données

            # Analyse des patterns de succès
            patterns_reussis = self._analyser_patterns_succes()

            # Mise à jour des stratégies
            self._mettre_a_jour_strategies(patterns_reussis)

            # Optimisation des seuils
            self._optimiser_seuils()

            self.logger.info("Apprentissage terminé - stratégies mises à jour")

        except Exception as e:
            self.logger.error(f"Erreur lors de l'apprentissage: {e}")

    def evoluer(self) -> List[str]:
        """Fait évoluer l'agent basé sur l'expérience accumulée

        Returns:
            List[str]: Nouvelles compétences acquises
        """
        competences_acquises = []

        try:
            # Évaluation de la maturité
            maturite = self._evaluer_maturite()

            if maturite > 0.8:
                # Évolution avancée
                if "gestion_predictive" not in self._competences:
                    self._competences.append("gestion_predictive")
                    competences_acquises.append("gestion_predictive")

                if "optimisation_quantique" not in self._competences:
                    self._competences.append("optimisation_quantique")
                    competences_acquises.append("optimisation_quantique")

            elif maturite > 0.6:
                # Évolution intermédiaire
                if "adaptation_dynamique" not in self._competences:
                    self._competences.append("adaptation_dynamique")
                    competences_acquises.append("adaptation_dynamique")

            self.logger.info(f"Évolution terminée: {len(competences_acquises)} nouvelles compétences")

        except Exception as e:
            self.logger.error(f"Erreur lors de l'évolution: {e}")

        return competences_acquises

    # Méthodes privées d'analyse
    def _collecter_donnees_environnement(self) -> Dict[str, float]:
        """Collecte les données environnementales"""
        if not hasattr(self, 'ecosysteme'):
            return {}

        env = self.ecosysteme.environnement
        return {
            "temperature": getattr(env, 'temperature', 20.0),
            "humidite": getattr(env, 'humidite', 60.0),
            "luminosite": getattr(env, 'luminosite', 70.0),
            "vent": getattr(env, 'vent', 5.0),
            "pression": getattr(env, 'pression', 1013.0)
        }

    def _collecter_donnees_cellules(self) -> List[Dict[str, Any]]:
        """Collecte les données des cellules"""
        donnees = []
        for cellule in self.cellules_surveillees:
            donnees.append({
                "nom": cellule.nom,
                "type": cellule.type_cellule,
                "sante": cellule.etats_internes.get("sante", 100),
                "stress": cellule.etats_internes.get("stress", 0),
                "age": cellule.age
            })
        return donnees

    def _calculer_stress_global(self, env_data: Dict[str, float], cellules_data: List[Dict]) -> float:
        """Calcule le niveau de stress global"""
        if not cellules_data:
            return 0.0

        # Stress environnemental
        stress_env = 0.0
        if env_data.get("temperature", 20) > 30 or env_data.get("temperature", 20) < 5:
            stress_env += 0.3
        if env_data.get("humidite", 60) < 30 or env_data.get("humidite", 60) > 90:
            stress_env += 0.2

        # Stress cellulaire moyen
        stress_cellules = sum(c.get("stress", 0) for c in cellules_data) / len(cellules_data)

        # Stress global pondéré
        return min(1.0, (stress_env * 0.4) + (stress_cellules * 0.6))

    def _detecter_problemes(self, env_data: Dict[str, float], cellules_data: List[Dict]) -> List[str]:
        """Détecte les problèmes dans l'écosystème"""
        problemes = []

        # Problèmes environnementaux
        if env_data.get("temperature", 20) > 35:
            problemes.append("canicule")
        if env_data.get("humidite", 60) < 20:
            problemes.append("secheresse")

        # Problèmes cellulaires
        cellules_stressees = [c for c in cellules_data if c.get("stress", 0) > 70]
        if len(cellules_stressees) > len(cellules_data) * 0.5:
            problemes.append("stress_cellulaire_generalise")

        return problemes

    # Méthodes privées d'action
    def _appliquer_strategie_urgence(self, situation: Situation) -> List[Dict[str, Any]]:
        """Applique la stratégie d'urgence"""
        adaptations = []
        for cellule_data in situation.cellules:
            if cellule_data["stress"] > 80:
                adaptations.append({
                    "cellule": cellule_data["nom"],
                    "action": "urgence_immediate",
                    "priorite": "critique"
                })
        return adaptations

    def _appliquer_adaptations_normales(self, situation: Situation) -> List[Dict[str, Any]]:
        """Applique les adaptations normales"""
        adaptations = []
        for cellule_data in situation.cellules:
            if cellule_data["stress"] > 60:
                adaptations.append({
                    "cellule": cellule_data["nom"],
                    "action": "adaptation_progressive",
                    "priorite": "moyenne"
                })
        return adaptations

    def _optimiser_performance(self, situation: Situation) -> List[str]:
        """Optimise les performances"""
        recommandations = []
        if situation.stress_global < 0.3:
            recommandations.append("explorer_nouvelles_adaptations")
        if all(c["sante"] > 80 for c in situation.cellules):
            recommandations.append("augmenter_efficacite")
        return recommandations

    def _appliquer_adaptation(self, cellule: Any, analyse: Dict) -> Dict[str, Any]:
        """Applique une adaptation à une cellule"""
        # Simulation d'adaptation
        return {
            "cellule": cellule.nom,
            "adaptation": analyse.get("type_adaptation", "generique"),
            "succes": True,
            "timestamp": datetime.now()
        }

    def _appliquer_modifications_environnement(self, analyse: Dict) -> List[Dict]:
        """Applique des modifications environnementales"""
        # Simulation de modifications
        return [{
            "type": "regulation_temperature",
            "valeur": -2.0,
            "duree": 3600
        }]

    # Méthodes privées d'apprentissage
    def _analyser_patterns_succes(self) -> Dict[str, Any]:
        """Analyse les patterns de succès"""
        return {"pattern_principal": "adaptation_thermique"}

    def _mettre_a_jour_strategies(self, patterns: Dict[str, Any]) -> None:
        """Met à jour les stratégies"""
        pass

    def _optimiser_seuils(self) -> None:
        """Optimise les seuils de décision"""
        pass

    def _evaluer_maturite(self) -> float:
        """Évalue la maturité de l'agent"""
        return min(1.0, len(self.historique_rapports) / 1000)

    # Attributs d'évolution
    _competences: List[str] = []

    def __repr__(self) -> str:
        return f"AgentKibali(statut={self.statut}, strategie={self.strategie}, cycles={self.cycle_courant})"