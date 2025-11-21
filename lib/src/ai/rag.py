# 🔍 RAG - Retrieval Augmented Generation

"""Système RAG pour la recherche intelligente dans la base de connaissances

Le système RAG permet :
- Indexation sémantique des documents
- Recherche contextuelle intelligente
- Génération augmentée par récupération
- Gestion de connaissances biologiques
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import re
import math
from collections import defaultdict

from ..shared.config import Config
from ..shared.logger import get_logger

@dataclass
class DocumentConnaissance:
    """Représente un document dans la base de connaissances"""
    id: str
    titre: str
    contenu: str
    categorie: str = "general"
    tags: List[str] = field(default_factory=list)
    source: str = ""
    date_creation: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour la sérialisation"""
        return {
            "id": self.id,
            "titre": self.titre,
            "contenu": self.contenu,
            "categorie": self.categorie,
            "tags": self.tags,
            "source": self.source,
            "date_creation": self.date_creation.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentConnaissance':
        """Crée un document à partir d'un dictionnaire"""
        return cls(
            id=data["id"],
            titre=data["titre"],
            contenu=data["contenu"],
            categorie=data.get("categorie", "general"),
            tags=data.get("tags", []),
            source=data.get("source", ""),
            date_creation=datetime.fromisoformat(data["date_creation"]),
            metadata=data.get("metadata", {})
        )

@dataclass
class ResultatRecherche:
    """Résultat d'une recherche dans la base de connaissances"""
    document: DocumentConnaissance
    score_pertinence: float
    extraits_pertinents: List[str] = field(default_factory=list)
    termes_trouves: List[str] = field(default_factory=list)

@dataclass
class ContexteRAG:
    """Contexte RAG pour la génération augmentée"""
    question: str
    documents_pertinents: List[ResultatRecherche] = field(default_factory=list)
    contexte_general: str = ""
    historique_conversation: List[Dict[str, Any]] = field(default_factory=list)

class SystemeRAG:
    """Système RAG principal pour la gestion des connaissances"""

    def __init__(self, chemin_base: Union[str, Path] = "data/rag/knowledge_base"):
        """Initialise le système RAG

        Args:
            chemin_base: Chemin vers la base de connaissances
        """
        self.chemin_base = Path(chemin_base)
        self.chemin_base.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("RAG.SystemeRAG")

        # Structures de données
        self.documents: Dict[str, DocumentConnaissance] = {}
        self.index_semantique: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.cache_recherches: Dict[str, List[ResultatRecherche]] = {}

        # Chargement de la base existante
        self._charger_base_connaissances()

        self.logger.info(f"Système RAG initialisé avec {len(self.documents)} documents")

    def ajouter_document(self, document: DocumentConnaissance) -> bool:
        """Ajoute un document à la base de connaissances

        Args:
            document: Document à ajouter

        Returns:
            bool: True si ajout réussi
        """
        try:
            if document.id in self.documents:
                self.logger.warning(f"Document {document.id} existe déjà, mise à jour")
                self._reindexer_document(document)
            else:
                self._indexer_document(document)

            self.documents[document.id] = document
            self._sauvegarder_document(document)

            self.logger.info(f"Document ajouté: {document.titre}")
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout du document {document.id}: {e}")
            return False

    def rechercher(self,
                  query: str,
                  categorie: Optional[str] = None,
                  top_k: int = 5,
                  seuil_pertinence: float = 0.1) -> List[ResultatRecherche]:
        """Effectue une recherche dans la base de connaissances

        Args:
            query: Requête de recherche
            categorie: Filtre par catégorie
            top_k: Nombre maximum de résultats
            seuil_pertinence: Seuil minimum de pertinence

        Returns:
            List[ResultatRecherche]: Résultats de recherche
        """
        try:
            # Vérification du cache
            cache_key = f"{query}_{categorie}_{top_k}"
            if cache_key in self.cache_recherches:
                self.logger.debug(f"Résultats trouvés en cache pour: {query}")
                return self.cache_recherches[cache_key]

            # Prétraitement de la requête
            termes_query = self._pretraiter_texte(query)

            # Recherche par termes
            resultats_bruts = self._rechercher_par_termes(termes_query, categorie)

            # Calcul des scores de pertinence
            resultats = []
            for doc_id, score_brut in resultats_bruts:
                document = self.documents[doc_id]

                # Calcul du score final
                score_final = self._calculer_score_pertinence(query, document, score_brut)

                if score_final >= seuil_pertinence:
                    # Extraction d'extraits pertinents
                    extraits = self._extraire_extraits_pertinents(document.contenu, query)

                    resultat = ResultatRecherche(
                        document=document,
                        score_pertinence=score_final,
                        extraits_pertinents=extraits,
                        termes_trouves=[t for t in termes_query if t in document.contenu.lower()]
                    )
                    resultats.append(resultat)

            # Tri par score décroissant
            resultats.sort(key=lambda x: x.score_pertinence, reverse=True)
            resultats = resultats[:top_k]

            # Mise en cache
            self.cache_recherches[cache_key] = resultats

            self.logger.debug(f"Recherche effectuée: {len(resultats)} résultats pour '{query}'")
            return resultats

        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche: {e}")
            return []

    def generer_contexte_rag(self, question: str, categorie: Optional[str] = None) -> ContexteRAG:
        """Génère un contexte RAG pour une question donnée

        Args:
            question: Question posée
            categorie: Catégorie de recherche

        Returns:
            ContexteRAG: Contexte enrichi pour la génération
        """
        try:
            # Recherche de documents pertinents
            resultats = self.rechercher(question, categorie=categorie, top_k=3)

            # Génération du contexte général
            contexte_general = self._generer_contexte_general(question)

            contexte = ContexteRAG(
                question=question,
                documents_pertinents=resultats,
                contexte_general=contexte_general
            )

            self.logger.debug(f"Contexte RAG généré pour: {question}")
            return contexte

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du contexte RAG: {e}")
            return ContexteRAG(question=question)

    def obtenir_statistiques(self) -> Dict[str, Any]:
        """Retourne les statistiques du système RAG

        Returns:
            Dict: Statistiques détaillées
        """
        stats = {
            "documents_totaux": len(self.documents),
            "termes_indexes": len(self.index_semantique),
            "cache_recherches": len(self.cache_recherches),
            "categories": {},
            "tailles_moyennes": {
                "documents": 0,
                "extraits": 0
            }
        }

        # Statistiques par catégorie
        categories = defaultdict(int)
        tailles_docs = []
        tailles_extraits = []

        for doc in self.documents.values():
            categories[doc.categorie] += 1
            tailles_docs.append(len(doc.contenu))

            # Comptage des extraits (simplifié)
            extraits = doc.contenu.split('.')
            tailles_extraits.extend([len(e.strip()) for e in extraits if e.strip()])

        stats["categories"] = dict(categories)
        stats["tailles_moyennes"]["documents"] = sum(tailles_docs) / len(tailles_docs) if tailles_docs else 0
        stats["tailles_moyennes"]["extraits"] = sum(tailles_extraits) / len(tailles_extraits) if tailles_extraits else 0

        return stats

    def exporter_base(self, chemin_export: Union[str, Path]) -> None:
        """Exporte la base de connaissances

        Args:
            chemin_export: Chemin d'export
        """
        chemin_export = Path(chemin_export)
        chemin_export.mkdir(parents=True, exist_ok=True)

        # Export des documents
        for doc_id, document in self.documents.items():
            chemin_doc = chemin_export / f"{doc_id}.json"
            with open(chemin_doc, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)

        # Export des statistiques
        stats = self.obtenir_statistiques()
        chemin_stats = chemin_export / "statistiques.json"
        with open(chemin_stats, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Base exportée vers {chemin_export}")

    # Méthodes privées
    def _charger_base_connaissances(self) -> None:
        """Charge la base de connaissances depuis les fichiers"""
        if not self.chemin_base.exists():
            return

        for fichier in self.chemin_base.glob("*.json"):
            if fichier.name == "statistiques.json":
                continue

            try:
                with open(fichier, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    document = DocumentConnaissance.from_dict(data)
                    self.documents[document.id] = document
                    self._indexer_document(document)

            except Exception as e:
                self.logger.error(f"Erreur lors du chargement de {fichier}: {e}")

    def _sauvegarder_document(self, document: DocumentConnaissance) -> None:
        """Sauvegarde un document dans un fichier"""
        chemin_doc = self.chemin_base / f"{document.id}.json"
        with open(chemin_doc, 'w', encoding='utf-8') as f:
            json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)

    def _indexer_document(self, document: DocumentConnaissance) -> None:
        """Indexe un document pour la recherche"""
        termes = self._pretraiter_texte(document.contenu + " " + document.titre)

        # Comptage des termes
        freq_termes = {}
        for terme in termes:
            freq_termes[terme] = freq_termes.get(terme, 0) + 1

        # Calcul des scores TF (simplifié)
        max_freq = max(freq_termes.values()) if freq_termes else 1
        score_base = len(termes) / 1000.0  # Normalisation par longueur

        for terme, freq in freq_termes.items():
            score = (freq / max_freq) * score_base
            self.index_semantique[terme].append((document.id, score))

    def _reindexer_document(self, document: DocumentConnaissance) -> None:
        """Réindexe un document modifié"""
        # Suppression de l'ancien index
        termes_a_nettoyer = []
        for terme, docs in self.index_semantique.items():
            self.index_semantique[terme] = [(d, s) for d, s in docs if d != document.id]
            if not self.index_semantique[terme]:
                termes_a_nettoyer.append(terme)

        for terme in termes_a_nettoyer:
            del self.index_semantique[terme]

        # Réindexation
        self._indexer_document(document)

    def _pretraiter_texte(self, texte: str) -> List[str]:
        """Prétraitement basique du texte pour l'indexation"""
        # Minuscules
        texte = texte.lower()

        # Suppression de la ponctuation basique
        texte = re.sub(r'[^\w\s]', ' ', texte)

        # Split en mots
        termes = texte.split()

        # Filtrage des mots courts et communs
        mots_communs = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'à', 'un', 'une', 'dans', 'sur', 'avec', 'pour', 'par', 'qui', 'que', 'dont', 'où'}
        termes = [terme for terme in termes if len(terme) > 2 and terme not in mots_communs]

        return termes

    def _rechercher_par_termes(self, termes: List[str], categorie: Optional[str] = None) -> List[Tuple[str, float]]:
        """Recherche par termes dans l'index sémantique"""
        scores_docs = defaultdict(float)

        for terme in termes:
            if terme in self.index_semantique:
                for doc_id, score in self.index_semantique[terme]:
                    if categorie and self.documents[doc_id].categorie != categorie:
                        continue
                    scores_docs[doc_id] += score

        # Normalisation par nombre de termes trouvés
        resultats = [(doc_id, score / len(termes)) for doc_id, score in scores_docs.items()]
        resultats.sort(key=lambda x: x[1], reverse=True)

        return resultats

    def _calculer_score_pertinence(self, query: str, document: DocumentConnaissance, score_brut: float) -> float:
        """Calcule le score de pertinence final"""
        # Score basé sur la similarité de contenu
        query_lower = query.lower()
        contenu_lower = document.contenu.lower()

        # Similarité simple (pourcentage de termes de la query présents)
        termes_query = set(self._pretraiter_texte(query))
        termes_doc = set(self._pretraiter_texte(document.contenu))

        similarite = len(termes_query.intersection(termes_doc)) / len(termes_query) if termes_query else 0

        # Score pondéré
        return (score_brut * 0.6) + (similarite * 0.4)

    def _extraire_extraits_pertinents(self, contenu: str, query: str, max_extraits: int = 3) -> List[str]:
        """Extrait des passages pertinents du contenu"""
        termes_query = set(self._pretraiter_texte(query))

        # Découpage en phrases (simplifié)
        phrases = contenu.split('.')

        # Score des phrases
        scores_phrases = []
        for phrase in phrases:
            termes_phrase = set(self._pretraiter_texte(phrase))
            score = len(termes_query.intersection(termes_phrase))
            if score > 0:
                scores_phrases.append((phrase.strip(), score))

        # Tri et sélection
        scores_phrases.sort(key=lambda x: x[1], reverse=True)
        extraits = [phrase for phrase, score in scores_phrases[:max_extraits]]

        return extraits

    def _generer_contexte_general(self, question: str) -> str:
        """Génère un contexte général pour la question"""
        # Recherche de documents généraux sur le sujet
        resultats = self.rechercher(question, categorie="general", top_k=2)

        if not resultats:
            return "Aucun contexte général disponible."

        contexte = "Contexte général:\n"
        for resultat in resultats:
            contexte += f"- {resultat.document.titre}: {resultat.extraits_pertinents[0] if resultat.extraits_pertinents else '...'}\n"

        return contexte

    def _construire_prompt_rag(self, contexte_rag: ContexteRAG) -> str:
        """Construit un prompt enrichi pour la génération RAG"""
        prompt = f"Question: {contexte_rag.question}\n\n"

        # Ajout du contexte général
        if contexte_rag.contexte_general:
            prompt += f"Contexte général:\n{contexte_rag.contexte_general}\n\n"

        # Ajout des documents pertinents
        if contexte_rag.documents_pertinents:
            prompt += "Informations pertinentes:\n"
            for i, resultat in enumerate(contexte_rag.documents_pertinents, 1):
                prompt += f"\n--- Document {i}: {resultat.document.titre} ---\n"
                prompt += f"Score de pertinence: {resultat.score_pertinence:.2f}\n"
                if resultat.extraits_pertinents:
                    prompt += "Extraits clés:\n"
                    for extrait in resultat.extraits_pertinents[:2]:  # Max 2 extraits par doc
                        prompt += f"- {extrait}\n"
                prompt += "\n"

        # Ajout de l'historique de conversation si disponible
        if contexte_rag.historique_conversation:
            prompt += "Historique de la conversation:\n"
            for msg in contexte_rag.historique_conversation[-3:]:  # Derniers 3 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "\n"

        prompt += "Réponse:"

        return prompt

    def _nettoyer_cache(self) -> None:
        """Nettoie le cache des recherches"""
        # Simplification: vidage complet du cache
        self.cache_recherches.clear()
        self.logger.debug("Cache des recherches nettoyé")