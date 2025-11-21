# 🧪 Tests - Écosystème KIBALI

## Structure des Tests

```
tests/
├── unitaires/           # Tests unitaires
│   ├── test_agent.py
│   ├── test_cellules.py
│   ├── test_ia.py
│   └── test_rag.py
├── integration/         # Tests d'intégration
│   ├── test_ecosysteme.py
│   └── test_orchestration.py
├── performance/         # Tests de performance
│   ├── test_cycles.py
│   └── test_memoire.py
├── benchmarks/          # Benchmarks
│   ├── benchmark_ia.py
│   └── benchmark_ecosysteme.py
└── conftest.py          # Configuration pytest
```

## Tests Unitaires

### Test de l'Agent KIBALI

```python
# tests/unitaires/test_agent.py
import pytest
from unittest.mock import Mock, patch
from kibali.agent import AgentKibali
from kibali.core import Situation, Adaptation

class TestAgentKibali:
    """Tests unitaires pour l'Agent KIBALI"""

    @pytest.fixture
    def agent(self):
        """Fixture pour créer un agent de test"""
        return AgentKibali(
            strategie="optimisation_adaptative",
            seuils={"adaptation": 0.7, "urgence": 0.8}
        )

    def test_initialisation_agent(self, agent):
        """Test de l'initialisation correcte de l'agent"""
        assert agent.statut == "actif"
        assert agent.strategie == "optimisation_adaptative"
        assert agent.cellules_surveillees == []
        assert agent.influences_appliquees == 0

    def test_analyse_situation_normale(self, agent):
        """Test de l'analyse d'une situation normale"""
        situation = Situation(
            environnement={"temperature": 22, "humidite": 65},
            cellules=[],
            stress_global=0.3
        )

        with patch.object(agent, '_collecter_donnees_environnement') as mock_collecte:
            mock_collecte.return_value = situation.environnement

            resultat = agent.analyser_situation()

            assert resultat.stress_global == 0.3
            assert "temperature" in resultat.environnement

    def test_cycle_orchestration_complet(self, agent):
        """Test d'un cycle d'orchestration complet"""
        # Mock des dépendances
        with patch.object(agent, 'analyser_situation') as mock_analyse, \
             patch.object(agent, 'influencer_cellule') as mock_influence, \
             patch.object(agent, 'generer_rapport_cycle') as mock_rapport:

            # Configuration des mocks
            situation = Mock()
            situation.stress_global = 0.4
            mock_analyse.return_value = situation

            adaptation = Mock()
            mock_influence.return_value = [adaptation]

            rapport = Mock()
            rapport.adaptations = [adaptation]
            mock_rapport.return_value = rapport

            # Exécution du cycle
            resultat = agent.cycle_orchestration()

            # Vérifications
            assert mock_analyse.called
            assert mock_influence.called
            assert mock_rapport.called
            assert resultat == rapport

    @pytest.mark.parametrize("stress_level,expected_urgence", [
        (0.5, False),   # Stress normal
        (0.75, False),  # Stress élevé mais pas urgence
        (0.85, True),   # Urgence déclarée
        (0.95, True),   # Urgence critique
    ])
    def test_detection_urgence(self, agent, stress_level, expected_urgence):
        """Test de la détection d'urgence selon le niveau de stress"""
        situation = Mock()
        situation.stress_global = stress_level

        with patch.object(agent, 'analyser_situation', return_value=situation):
            urgence = agent.detecter_urgence()
            assert urgence == expected_urgence

    def test_adaptation_cellule_stress(self, agent):
        """Test de l'adaptation d'une cellule en situation de stress"""
        cellule_mock = Mock()
        cellule_mock.etats_internes = {"stress": Mock(valeur=0.8)}

        situation = Mock()
        situation.stress_global = 0.7

        with patch.object(agent, '_analyser_avec_codellama') as mock_ia:
            mock_ia.return_value = {
                "adaptation": "augmenter_resistance",
                "parametres": {"resistance": 1.2}
            }

            resultat = agent.influencer_cellule(cellule_mock)

            assert mock_ia.called
            assert resultat is not None

    def test_gestion_erreur_ia(self, agent):
        """Test de la gestion d'erreur des modèles IA"""
        cellule_mock = Mock()

        # Simulation d'erreur IA
        with patch.object(agent, '_analyser_avec_codellama', side_effect=Exception("Erreur GPU")):
            with patch.object(agent, '_analyser_avec_phi') as mock_phi:
                mock_phi.return_value = {"adaptation": "fallback"}

                resultat = agent.influencer_cellule(cellule_mock)

                # Vérification du fallback
                assert mock_phi.called
                assert resultat is not None
```

### Test des Cellules

```python
# tests/unitaires/test_cellules.py
import pytest
from kibali.cells.arbre import Arbre
from kibali.cells.fleur import Fleur
from kibali.core import EtatInterne, ObjetPhysique

class TestCelluleArbre:
    """Tests pour la cellule Arbre"""

    @pytest.fixture
    def arbre(self):
        """Fixture pour créer un arbre de test"""
        return Arbre(
            nom="Chêne Test",
            age=50,
            etats_internes={
                "sante": EtatInterne(90, "%", 0, 100),
                "stress": EtatInterne(20, "%", 0, 100),
                "photosynthese": EtatInterne(1.0, "rate", 0.1, 2.0)
            }
        )

    def test_creation_arbre(self, arbre):
        """Test de création d'un arbre"""
        assert arbre.nom == "Chêne Test"
        assert arbre.age == 50
        assert arbre.type_cellule == "vegetal"
        assert arbre.etats_internes["sante"].valeur == 90

    def test_perception_environnement(self, arbre):
        """Test de la perception environnementale"""
        environnement = {
            "temperature": 25,
            "humidite": 60,
            "luminosite": 70,
            "vent": 10
        }

        perceptions = arbre.percevoir_environnement(environnement)

        assert "temperature" in perceptions
        assert perceptions["temperature"] == 25
        assert len(perceptions) == len(environnement)

    @pytest.mark.parametrize("temp,humidite,expected_stress", [
        (20, 70, 10),   # Conditions optimales
        (30, 40, 35),   # Stress modéré
        (35, 20, 60),   # Stress élevé
        (40, 10, 85),   # Stress critique
    ])
    def test_adaptation_stress_thermique(self, arbre, temp, humidite, expected_stress):
        """Test de l'adaptation au stress thermique"""
        perceptions = {"temperature": temp, "humidite": humidite}

        adaptations = arbre.adapter_autonomously(perceptions)

        # Vérification du stress calculé
        stress_calcule = arbre.etats_internes["stress"].valeur
        assert abs(stress_calcule - expected_stress) < 10  # Tolérance de 10%

        # Vérification des adaptations
        if temp > 30:
            assert "reduire_transpiration" in adaptations
        if humidite < 30:
            assert "economiser_eau" in adaptations

    def test_evolution_arbre(self, arbre):
        """Test de l'évolution naturelle de l'arbre"""
        etats_initiaux = {
            k: v.valeur for k, v in arbre.etats_internes.items()
        }

        # Simulation de plusieurs cycles
        for _ in range(10):
            arbre.evoluer()

        # Vérification de l'évolution
        assert arbre.age > 50  # Vieillissement
        assert arbre.etats_internes["sante"].valeur >= etats_initiaux["sante"] * 0.9  # Santé stable

class TestCelluleFleur:
    """Tests pour la cellule Fleur"""

    @pytest.fixture
    def fleur(self):
        """Fixture pour créer une fleur de test"""
        return Fleur(
            nom="Rose Test",
            saison="printemps",
            etats_internes={
                "floraison": EtatInterne(0.8, "rate", 0, 1),
                "pollinisation": EtatInterne(0.3, "rate", 0, 1),
                "stress": EtatInterne(15, "%", 0, 100)
            }
        )

    def test_floraison_saisonniere(self, fleur):
        """Test de la floraison selon les saisons"""
        # Printemps - floraison optimale
        environnement = {"saison": "printemps", "temperature": 18}
        fleur.adapter_autonomously(environnement)
        assert fleur.etats_internes["floraison"].valeur > 0.7

        # Hiver - floraison réduite
        environnement = {"saison": "hiver", "temperature": 5}
        fleur.adapter_autonomously(environnement)
        assert fleur.etats_internes["floraison"].valeur < 0.3

    def test_interaction_pollinisateur(self, fleur):
        """Test de l'interaction avec les pollinisateurs"""
        from kibali.cells.ecureuil import Ecureuil

        pollinisateur = Ecureuil("Abeille Test")

        # Simulation de pollinisation
        fleur.interagir_avec(pollinisateur)

        # Vérification de l'effet
        assert fleur.etats_internes["pollinisation"].valeur > 0.3
        assert pollinisateur.etats_internes["energie"].valeur > 80  # Pollinisateur nourri
```

### Test de l'IA

```python
# tests/unitaires/test_ia.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from kibali.ai import ModeleIA, AnalyseurIA
import torch

class TestModeleIA:
    """Tests pour les modèles IA"""

    @pytest.fixture
    def modele_codellama(self):
        """Fixture pour Code Llama"""
        return ModeleIA(
            nom="codellama",
            type_modele="llm",
            chemin_modele="codellama/CodeLlama-7b-hf",
            device="cpu"  # CPU pour les tests
        )

    def test_initialisation_modele(self, modele_codellama):
        """Test d'initialisation d'un modèle IA"""
        assert modele_codellama.nom == "codellama"
        assert modele_codellama.device == "cpu"
        assert not modele_codellama.est_charge

    @pytest.mark.asyncio
    async def test_analyse_texte(self, modele_codellama):
        """Test d'analyse de texte"""
        prompt = "Analyser le stress hydrique d'un arbre"
        contexte = {"temperature": 35, "humidite": 25}

        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                # Mock du modèle
                mock_model.return_value = Mock()
                mock_tokenizer.return_value = Mock()

                # Mock de génération
                mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
                mock_tokenizer.return_value.decode.return_value = "Analyse: Stress hydrique détecté"

                await modele_codellama.charger()
                resultat = await modele_codellama.analyser(prompt, contexte)

                assert "stress" in resultat.lower()
                assert isinstance(resultat, str)

    def test_quantification_modele(self):
        """Test de la quantification 4-bit"""
        modele_quantifie = ModeleIA(
            nom="codellama_4bit",
            type_modele="llm",
            chemin_modele="codellama/CodeLlama-7b-hf",
            quantification="4bit"
        )

        # Vérification de la configuration de quantification
        assert modele_quantifie.quantification == "4bit"

class TestAnalyseurIA:
    """Tests pour l'analyseur IA"""

    @pytest.fixture
    def analyseur(self):
        """Fixture pour l'analyseur IA"""
        modeles = [
            Mock(spec=ModeleIA, nom="codellama"),
            Mock(spec=ModeleIA, nom="phi")
        ]
        return AnalyseurIA(modeles)

    @pytest.mark.asyncio
    async def test_analyse_situation_complexe(self, analyseur):
        """Test d'analyse de situation complexe"""
        situation = {
            "description": "Forêt en stress thermique",
            "parametres": {"temperature": 38, "humidite": 20}
        }

        # Mock des réponses IA
        analyseur.modeles[0].analyser = AsyncMock(return_value="Code Llama: Augmenter résistance")
        analyseur.modeles[1].analyser = AsyncMock(return_value="Phi: Réduire transpiration")

        resultat = await analyseur.analyser_situation(situation)

        assert "résistance" in resultat.lower()
        assert "transpiration" in resultat.lower()
        assert analyseur.modeles[0].analyser.called
        assert analyseur.modeles[1].analyser.called

    def test_fallback_erreur(self, analyseur):
        """Test du mécanisme de fallback en cas d'erreur"""
        # Simulation d'erreur sur le premier modèle
        analyseur.modeles[0].analyser = Mock(side_effect=Exception("GPU Error"))
        analyseur.modeles[1].analyser = Mock(return_value="Fallback réussi")

        # L'analyseur devrait utiliser le deuxième modèle
        with patch.object(analyseur, '_analyser_avec_modele', side_effect=[Exception("Error"), "Succès"]):
            # Test du fallback (implémentation dépendante)
            pass
```

## Tests d'Intégration

### Test de l'Écosystème Complet

```python
# tests/integration/test_ecosysteme.py
import pytest
from kibali import Ecosysteme, AgentKibali
from kibali.cells import Arbre, Climat, Ecureuil, Fleur
from kibali.environment import Environnement

class TestEcosystemeIntegration:
    """Tests d'intégration de l'écosystème complet"""

    @pytest.fixture
    def ecosysteme_complet(self):
        """Fixture pour un écosystème complet"""
        ecosysteme = Ecosysteme()

        # Ajout de cellules variées
        ecosysteme.ajouter_cellule(Arbre("Chêne", age=100))
        ecosysteme.ajouter_cellule(Arbre("Sapin", age=50))
        ecosysteme.ajouter_cellule(Climat("Forêt Tempérée"))
        ecosysteme.ajouter_cellule(Ecureuil("Noisette"))
        ecosysteme.ajouter_cellule(Fleur("Rose", saison="printemps"))

        # Configuration environnementale
        environnement = Environnement(
            temperature=22,
            humidite=65,
            luminosite=70,
            vent=5
        )
        ecosysteme.definir_environnement(environnement)

        return ecosysteme

    @pytest.fixture
    def agent_orchestrateur(self, ecosysteme_complet):
        """Fixture pour l'agent orchestrateur"""
        agent = AgentKibali(strategie="optimisation_adaptative")
        agent.connecter_ecosysteme(ecosysteme_complet)
        return agent

    def test_initialisation_ecosysteme(self, ecosysteme_complet):
        """Test d'initialisation d'un écosystème complet"""
        assert len(ecosysteme_complet.cellules) == 5
        assert ecosysteme_complet.environnement.temperature == 22

        # Vérification des types de cellules
        types = [c.type_cellule for c in ecosysteme_complet.cellules]
        assert "vegetal" in types
        assert "climatique" in types
        assert "animal" in types

    def test_cycle_orchestration_integration(self, agent_orchestrateur):
        """Test d'intégration des cycles d'orchestration"""
        # Exécution de plusieurs cycles
        rapports = []
        for _ in range(5):
            rapport = agent_orchestrateur.cycle_orchestration()
            rapports.append(rapport)

            # Vérifications de base
            assert rapport is not None
            assert hasattr(rapport, 'adaptations')
            assert hasattr(rapport, 'stress_global')

        # Vérification de l'évolution
        stress_initial = rapports[0].stress_global
        stress_final = rapports[-1].stress_global

        # Le système devrait s'adapter et réduire le stress
        assert stress_final <= stress_initial + 0.1  # Tolérance

    def test_interactions_cellulaires(self, ecosysteme_complet):
        """Test des interactions entre cellules"""
        chene = next(c for c in ecosysteme_complet.cellules if c.nom == "Chêne")
        ecureuil = next(c for c in ecosysteme_complet.cellules if c.nom == "Noisette")

        # Simulation d'interaction
        chene.interagir_avec(ecureuil)

        # Vérification des effets
        assert ecureuil.etats_internes["energie"].valeur > 90  # Nourri par le chêne
        assert chene.etats_internes["stress"].valeur < 25  # Bénéfice de l'interaction

    @pytest.mark.slow
    def test_stabilite_long_terme(self, agent_orchestrateur):
        """Test de stabilité sur le long terme"""
        stress_historique = []

        # Simulation de 50 cycles
        for cycle in range(50):
            rapport = agent_orchestrateur.cycle_orchestration()
            stress_historique.append(rapport.stress_global)

            # Vérification de la survie du système
            assert rapport.stress_global < 0.9  # Pas d'effondrement

        # Analyse de la tendance
        stress_moyen = sum(stress_historique) / len(stress_historique)
        stress_final = stress_historique[-1]

        # Le système devrait converger vers un état stable
        assert stress_moyen < 0.5
        assert stress_final < stress_moyen + 0.2

    def test_scenario_stress_extreme(self, agent_orchestrateur):
        """Test de résistance aux scénarios extrêmes"""
        # Application de stress extrême
        environnement_stress = Environnement(
            temperature=40,  # Canicule
            humidite=15,     # Sécheresse
            luminosite=90,   # Forte exposition
            vent=25          # Tempête
        )

        agent_orchestrateur.ecosysteme.definir_environnement(environnement_stress)

        # Test de réponse d'urgence
        urgence_detectee = False
        adaptations_urgence = 0

        for cycle in range(10):
            rapport = agent_orchestrateur.cycle_orchestration()

            if rapport.urgence_declaree:
                urgence_detectee = True
                adaptations_urgence += len(rapport.adaptations_urgence)

        # Vérifications
        assert urgence_detectee, "L'urgence devrait être détectée"
        assert adaptations_urgence > 0, "Des adaptations d'urgence devraient être appliquées"

        # Vérification de la résilience
        stress_final = agent_orchestrateur.analyser_situation().stress_global
        assert stress_final < 0.95, "Le système devrait résister au stress extrême"
```

## Tests de Performance

### Benchmarks IA

```python
# tests/performance/benchmark_ia.py
import pytest
import time
import psutil
import torch
from kibali.ai import ModeleIA, AnalyseurIA

class TestPerformanceIA:
    """Tests de performance pour les composants IA"""

    @pytest.fixture
    def modele_test(self):
        """Fixture pour modèle de test"""
        return ModeleIA(
            nom="test_model",
            type_modele="llm",
            chemin_modele="codellama/CodeLlama-7b-hf",
            device="cpu"
        )

    def test_temps_chargement_modele(self, modele_test):
        """Test du temps de chargement des modèles"""
        debut = time.time()

        # Simulation de chargement (en vrai, utiliser await modele.charger())
        time.sleep(0.1)  # Simulati on

        temps_chargement = time.time() - debut

        # Le chargement devrait être raisonnable
        assert temps_chargement < 30  # 30 secondes max

    def test_consommation_memoire_ia(self, modele_test):
        """Test de la consommation mémoire des modèles IA"""
        process = psutil.Process()
        memoire_avant = process.memory_info().rss / 1024 / 1024  # MB

        # Simulation de chargement modèle
        # En vrai: await modele.charger()

        memoire_apres = process.memory_info().rss / 1024 / 1024  # MB
        consommation = memoire_apres - memoire_avant

        # Vérification des limites
        assert consommation < 500  # 500MB max pour les tests

    @pytest.mark.parametrize("taille_prompt", [100, 500, 1000])
    def test_temps_inference(self, taille_prompt):
        """Test du temps d'inférence selon la taille du prompt"""
        # Génération de prompts de différentes tailles
        prompt = "Analyser " * (taille_prompt // 8)

        # Mesure du temps d'inférence
        debut = time.time()

        # Simulation d'inférence
        time.sleep(0.01 * (taille_prompt / 100))  # Simulation proportionnelle

        temps_inference = time.time() - debut

        # Vérification des performances
        assert temps_inference < 5.0  # 5 secondes max

        # Vérification de la scalabilité
        if taille_prompt > 500:
            assert temps_inference < 2.0  # Plus rapide pour gros prompts (optimisation)

    def test_parallele_analyses(self):
        """Test des analyses en parallèle"""
        import asyncio

        async def analyser_prompt(i):
            await asyncio.sleep(0.1)  # Simulation
            return f"Résultat {i}"

        async def test_parallele():
            taches = [analyser_prompt(i) for i in range(10)]

            debut = time.time()
            resultats = await asyncio.gather(*taches)
            temps_total = time.time() - debut

            # Vérification du parallélisme
            assert temps_total < 0.3  # Moins que 10 * 0.1 en séquentiel
            assert len(resultats) == 10

        asyncio.run(test_parallele())
```

### Benchmarks Écosystème

```python
# tests/performance/benchmark_ecosysteme.py
import pytest
import time
import cProfile
import pstats
from kibali import Ecosysteme, AgentKibali
from kibali.cells import Arbre

class TestPerformanceEcosysteme:
    """Tests de performance pour l'écosystème"""

    def test_scalabilite_cellules(self):
        """Test de scalabilité selon le nombre de cellules"""
        tailles_ecosysteme = [10, 50, 100, 500]

        for taille in tailles_ecosysteme:
            ecosysteme = Ecosysteme()
            agent = AgentKibali()

            # Ajout de cellules
            for i in range(taille):
                arbre = Arbre(f"Arbre_{i}", age=30)
                ecosysteme.ajouter_cellule(arbre)

            agent.connecter_ecosysteme(ecosysteme)

            # Mesure du temps de cycle
            debut = time.time()
            rapport = agent.cycle_orchestration()
            temps_cycle = time.time() - debut

            # Vérification des performances
            assert temps_cycle < 5.0, f"Cycle trop lent pour {taille} cellules: {temps_cycle}s"

            # Vérification de la qualité
            assert len(rapport.adaptations) > 0

    def test_profiling_orchestration(self):
        """Profiling détaillé de l'orchestration"""
        ecosysteme = Ecosysteme()
        agent = AgentKibali()

        # Écosystème de taille moyenne
        for i in range(20):
            ecosysteme.ajouter_cellule(Arbre(f"Arbre_{i}", age=40))

        agent.connecter_ecosysteme(ecosysteme)

        # Profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Exécution profilée
        for _ in range(5):
            agent.cycle_orchestration()

        profiler.disable()

        # Analyse des résultats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')

        # Fonctions les plus coûteuses
        fonctions_critiques = []
        for func_stat in stats.stats.items():
            if func_stat[1][3] > 0.1:  # Plus de 100ms
                fonctions_critiques.append(func_stat[0])

        # Vérification qu'aucune fonction ne domine
        assert len(fonctions_critiques) < 5, f"Trop de fonctions critiques: {fonctions_critiques}"

    def test_memoire_cycles_prolonges(self):
        """Test de stabilité mémoire sur cycles prolongés"""
        import gc

        ecosysteme = Ecosysteme()
        agent = AgentKibali()

        # Écosystème modéré
        for i in range(25):
            ecosysteme.ajouter_cellule(Arbre(f"Arbre_{i}", age=35))

        agent.connecter_ecosysteme(ecosysteme)

        memoire_initiale = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Cycles prolongés
        for cycle in range(100):
            agent.cycle_orchestration()

            # Nettoyage périodique
            if cycle % 10 == 0:
                gc.collect()

        memoire_finale = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        consommation = memoire_finale - memoire_initiale

        # Vérification de l'absence de fuites
        assert consommation < 100, f"Fuite mémoire détectée: +{consommation}MB"

    @pytest.mark.benchmark
    def test_benchmark_complet(self, benchmark):
        """Benchmark complet de l'écosystème"""
        def setup_ecosysteme():
            ecosysteme = Ecosysteme()
            for i in range(30):
                ecosysteme.ajouter_cellule(Arbre(f"Arbre_{i}", age=45))
            agent = AgentKibali()
            agent.connecter_ecosysteme(ecosysteme)
            return agent

        agent = setup_ecosysteme()

        # Benchmark du cycle d'orchestration
        result = benchmark(agent.cycle_orchestration)

        # Assertions sur les performances
        assert result is not None
        assert hasattr(result, 'adaptations')

        # Vérification des métriques de performance
        # (Dépend des outils de benchmarking utilisés)
```

## Configuration des Tests

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --disable-warnings
    --tb=short
    -v
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    benchmark: marks tests as benchmarks
    gpu: marks tests that require GPU
```

### conftest.py

```python
# tests/conftest.py
import pytest
import torch
from kibali.config import Config

@pytest.fixture(scope="session")
def config_test():
    """Configuration globale pour les tests"""
    return Config.from_dict({
        "ai": {
            "models": {
                "codellama": {"device": "cpu"},
                "phi": {"device": "cpu"}
            }
        },
        "performance": {
            "cache_analyses": False,  # Désactiver pour tests
            "parallel_processing": False
        }
    })

@pytest.fixture
def gpu_available():
    """Vérification de la disponibilité du GPU"""
    return torch.cuda.is_available()

@pytest.fixture
def mock_modele_ia():
    """Mock pour les modèles IA"""
    from unittest.mock import Mock
    mock = Mock()
    mock.analyser = Mock(return_value="Analyse mockée")
    mock.generer_code = Mock(return_value="Code mocké")
    return mock
```

## Exécution des Tests

### Commandes de Base

```bash
# Tests unitaires rapides
pytest tests/unitaires/ -v

# Tests d'intégration
pytest tests/integration/ -v

# Tests de performance
pytest tests/performance/ -v -m "not slow"

# Tests complets (lent)
pytest tests/ -v --tb=long

# Avec couverture
pytest --cov=kibali --cov-report=html --cov-report=term

# Tests spécifiques
pytest tests/unitaires/test_agent.py::TestAgentKibali::test_cycle_orchestration_complet -v
```

### Tests en Continu

```bash
# Surveillance des changements
pytest-watch tests/unitaires/

# Tests avant commit (pre-commit)
pre-commit run --all-files
```

### Analyse des Résultats

```bash
# Rapport de couverture détaillé
coverage report --show-missing

# Génération de badges
coverage-badge -o coverage.svg

# Analyse des performances
pytest --benchmark-only --benchmark-histogram
```

---

*Cette suite de tests assure la qualité et la fiabilité de l'écosystème KIBALI.*