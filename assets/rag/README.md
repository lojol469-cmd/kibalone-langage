# Kibali RAG System

Système de Retrieval-Augmented Generation pour les cellules nano-IA Kibali.

## Structure

```
rag/
├── config.json          # Configuration du système RAG
├── indexes/            # Index FAISS pour la recherche vectorielle
├── embeddings/         # Embeddings pré-calculés
├── metadata/           # Métadonnées des documents indexés
└── cache/              # Cache temporaire pour les calculs
```

## Fonctionnalités

- **Indexation intelligente** : Indexation des connaissances des cellules
- **Recherche sémantique** : Recherche par similarité dans les connaissances
- **Intégration cellulaire** : Connexion directe avec les cellules mémoire
- **Évolution tracking** : Suivi des changements dans les connaissances

## Utilisation

Le système RAG est automatiquement utilisé par les cellules Kibali pour :
- Enrichir les réponses du cerveau avec des connaissances contextuelles
- Maintenir la mémoire à long terme des cellules
- Faciliter l'évolution et l'adaptation des nano-IA

## Configuration

Voir `config.json` pour les paramètres d'embedding, chunking, et seuils de similarité.