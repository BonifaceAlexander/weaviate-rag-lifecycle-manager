# RFC: RAG Lifecycle Management Standard for Weaviate

| Status        | Draft |
| :---          | :--- |
| **Authors**   | [User Name] |
| **Created**   | 2026-01-12 |
| **Target**    | Weaviate Ecosystem / Python Client |

## 1. Abstract
This proposal introduces a standardized reference architecture for managing the lifecycle of Retrieval-Augmented Generation (RAG) systems using Weaviate. It defines a schema and state machine for versioning datasets (`RAGDataset`), embedding configurations (`EmbeddingConfig`), and physical indices (`IndexGeneration`). This allows teams to implement safe promotion workflows (Draft → Staging → Production), robust rollbacks, and zero-downtime upgrades of embedding models.

## 2. Motivation
As RAG systems move to production, teams face critical lifecycle challenges that vectors databases do not natively solve today:
1.  **Embedding Model Upgrades**: Changing an embedding model (e.g., from `text-embedding-3-small` to `large`) requires full re-indexing and often results in downtime or risky "hot swaps".
2.  **Dataset Versioning**: There is no standard way to track which version of a source dataset is currently searchable.
3.  **Governance**: There is no "System of Record" for which index is vital for production traffic vs. experimental.

Currently, these concerns are handled by ad-hoc scripts outside the database. We propose making Weaviate itself the system of record for this lifecycle state.

## 3. Operations & Terminology
We introduce three core entities that model the RAG domain:

### 3.1 Entities
*   **RAGDataset**: Represents a logical corpus of data (e.g., "Company Wiki", "Legal Contracts"). It tracks the content source version.
*   **EmbeddingConfig**: Represents the strategy used to turn text into vectors. Uniquely defined by the generic tuple `(Model Name, Chunk Size, Chunk Overlap, Vectorizer Parameters)`.
*   **IndexGeneration**: Represents a concrete, physical set of vectors in Weaviate. An Index Generation is the result of applying an *EmbeddingConfig* to a *RAGDataset*. It has a queryable lifecycle status.

### 3.2 Lifecycle States
An `IndexGeneration` flows through a strict state machine:
*   `DRAFT`: The index is being created and populated. Not ready for queries.
*   `INDEXING`: Data ingestion is in progress.
*   `STAGING`: Index is complete and frozen. Available for internal QA/evaluation.
*   `PRODUCTION`: The single authoritative index for live traffic.
*   `DEPRECATED`: A former production index that is kept for instant rollback/fallback.
*   `ARCHIVED`: Cold storage or deleted.

## 4. Detailed Design

### 4.1 Schema Definition
The system utilizes Weaviate's own schema to track metadata. We propose reserved classes (collections) for this purpose.

**Class: `RAGDataset`**
```json
{
  "class": "RAGDataset",
  "properties": [
    {"name": "dataset_id", "dataType": ["text"]},
    {"name": "name", "dataType": ["text"]},
    {"name": "version", "dataType": ["text"]}
  ]
}
```

**Class: `EmbeddingConfig`**
```json
{
  "class": "EmbeddingConfig",
  "properties": [
    {"name": "config_id", "dataType": ["text"]},
    {"name": "model_name", "dataType": ["text"]},
    {"name": "chunk_size", "dataType": ["int"]},
    {"name": "chunk_overlap", "dataType": ["int"]}
  ]
}
```

**Class: `IndexGeneration`**
```json
{
  "class": "IndexGeneration",
  "properties": [
    {"name": "generation_id", "dataType": ["text"]},
    {"name": "status", "dataType": ["text"]}, // Enum: DRAFT, PRODUCTION, etc.
    {"name": "weaviate_collection_name", "dataType": ["text"]}, // Pointer to actual vector index
    {"name": "created_at", "dataType": ["date"]}
  ],
  "references": [
    {"name": "hasDataset", "dataType": ["RAGDataset"]},
    {"name": "hasConfig", "dataType": ["EmbeddingConfig"]}
  ]
}
```

### 4.2 Application Logic (The Manager)
A lightweight client-side manager handles the logic. This logic does not need to live inside Weaviate core, keeping the database lean.

**Promotion Logic:**
When an index is promoted to `PRODUCTION`:
1.  Query Weaviate for any existing `IndexGeneration` for the same `RAGDataset` that is currently `PRODUCTION`.
2.  Atomically (or eventually consistently) downgrade those to `DEPRECATED`.
3.  Update the target `IndexGeneration` status to `PRODUCTION`.

**Retrieval Logic:**
Retrievers (e.g., LangChain, LlamaIndex) do not query a static collection name. Instead:
1.  App requests: `get_retriever(dataset="WikiDocs")`.
2.  Manager queries Weaviate: `Select weaviate_collection_name from IndexGeneration where dataset.name="WikiDocs" AND status="PRODUCTION" sort by updated_at desc limit 1`.
3.  Manager returns a retriever pointing to the resolved collection (e.g., `Index_7f8a9...`).

## 5. User Experience Example

```python
# 1. Register a new configuration
config = manager.register_config(model="text-embedding-3", chunk_size=512)

# 2. Create a new index generation (e.g. for a nightly build)
gen = manager.create_generation(dataset="Wiki", config=config)

# 3. Populate 'gen.weaviate_collection_name' with data...
# ... ingestion process ...

# 4. Verify in Staging
manager.promote(gen.id, "STAGING")
evaluate_rag(gen) # Run Ragas/evals

# 5. Flip to Production (Instant)
manager.promote(gen.id, "PRODUCTION")
```

## 6. Compatibility
This design is fully compatible with Weaviate v4 client and existing orchestration frameworks. It behaves as a "Meta-Retriever" layer.

## 7. Future Work
*   **A/B Testing**: Allow `PRODUCTION` state to have a `traffic_weight` property to split traffic between two generations.
*   **Garbage Collection**: Automated policies to delete `ARCHIVED` physical indices to free up disk space.
*   **UI Integration**: A simple Weaviate Console plugin to visualize the timeline of Index Generations.
