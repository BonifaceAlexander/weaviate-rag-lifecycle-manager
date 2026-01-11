import weaviate
from weaviate.classes.query import Filter
import uuid
from datetime import datetime
from typing import Optional, List

from .states import LifecycleState
from .models import RAGDataset, EmbeddingConfig, IndexGeneration
from .schema import init_schemas, DATASET_CLASS, CONFIG_CLASS, GENERATION_CLASS

class WeaviateRAGLifecycleManager:
    def __init__(self, client: weaviate.WeaviateClient):
        self.client = client

    def initialize(self):
        """Initializes the Weaviate schema for lifecycle management."""
        init_schemas(self.client)

    def create_dataset(self, name: str, version: str) -> RAGDataset:
        """Creates a new RAG Dataset version."""
        dataset = RAGDataset(
            id=str(uuid.uuid4()),
            name=name,
            version=version
        )
        
        self.client.collections.get(DATASET_CLASS).data.insert(
            properties={
                "dataset_id": dataset.id,
                "name": dataset.name,
                "version": dataset.version,
                "created_at": dataset.created_at
            }
        )
        return dataset

    def register_embedding_config(self, model_name: str, chunk_size: int, chunk_overlap: int) -> EmbeddingConfig:
        """Registers a new Embedding Configuration."""
        # Create a deterministic ID based on content to avoid duplicates for same config
        config_str = f"{model_name}-{chunk_size}-{chunk_overlap}"
        config_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, config_str))
        
        config = EmbeddingConfig(
            id=config_id,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Check if exists
        collection = self.client.collections.get(CONFIG_CLASS)
        exists = collection.query.fetch_objects(
            filters=Filter.by_property("config_id").equal(config_id),
            limit=1
        )
        
        if not exists.objects:
            collection.data.insert(
                properties={
                    "config_id": config.id,
                    "model_name": config.model_name,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "created_at": config.created_at
                }
            )
            
        return config

    def create_index_generation(self, dataset_id: str, config_id: str) -> IndexGeneration:
        """
        Creates a new Index Generation entry and provisionally creates the target collection.
        Does NOT insert data.
        """
        gen_id = str(uuid.uuid4())
        # Naming convention for the physical index: "Index_<GenID>" 
        # (Weaviate requires class names to be capitalized)
        physical_index_name = f"Index_{gen_id.replace('-', '')}"

        generation = IndexGeneration(
            id=gen_id,
            dataset_id=dataset_id,
            config_id=config_id,
            status=LifecycleState.DRAFT,
            weaviate_collection_name=physical_index_name
        )

        # 1. Record metadata
        self.client.collections.get(GENERATION_CLASS).data.insert(
            properties={
                "generation_id": generation.id,
                "dataset_id": generation.dataset_id,
                "config_id": generation.config_id,
                "status": generation.status.value,
                "weaviate_collection_name": generation.weaviate_collection_name,
                "created_at": generation.created_at,
                "updated_at": generation.updated_at
            }
        )

        return generation

    def get_index_generation(self, generation_id: str) -> Optional[IndexGeneration]:
        collection = self.client.collections.get(GENERATION_CLASS)
        result = collection.query.fetch_objects(
            filters=Filter.by_property("generation_id").equal(generation_id),
            limit=1
        )
        if not result.objects:
            return None
        
        props = result.objects[0].properties
        return IndexGeneration(
            id=props["generation_id"],
            dataset_id=props["dataset_id"],
            config_id=props["config_id"],
            status=LifecycleState(props["status"]),
            weaviate_collection_name=props["weaviate_collection_name"],
            created_at=props["created_at"],
            updated_at=props["updated_at"]
        )

    def promote_index(self, generation_id: str, target_state: LifecycleState) -> IndexGeneration:
        """Promotes an index to a new state (e.g., STAGING -> PRODUCTION)."""
        gen = self.get_index_generation(generation_id)
        if not gen:
            raise ValueError(f"Generation {generation_id} not found")

        # If promoting to PRODUCTION, we might want to archive the old one for this dataset
        # For simplicity, we just allow multiple for now or rely on the user to use 
        # get_production_index which returns the *latest* production index.
        # But ideally, we should mark previous production as deprecated or archived.
        
        if target_state == LifecycleState.PRODUCTION:
             self._archive_previous_production(gen.dataset_id, exclude_gen_id=generation_id)

        # Update status
        collection = self.client.collections.get(GENERATION_CLASS)
        # We need the UUID of the object to update it. 
        # Since we use a custom 'generation_id' property, we first find the object UUID.
        result = collection.query.fetch_objects(
            filters=Filter.by_property("generation_id").equal(generation_id),
            limit=1
        )
        if not result.objects:
            raise ValueError("System error: generation found earlier but not now")
        
        obj_uuid = result.objects[0].uuid
        
        collection.data.update(
            uuid=obj_uuid,
            properties={
                "status": target_state.value,
                "updated_at": datetime.utcnow()
            }
        )
        
        gen.status = target_state
        return gen

    def _archive_previous_production(self, dataset_id: str, exclude_gen_id: str):
        """Marks previous PRODUCTION indices for this dataset as DEPRECATED."""
        collection = self.client.collections.get(GENERATION_CLASS)
        
        # Find all PRODUCTION indices for this dataset
        results = collection.query.fetch_objects(
            filters=(
                Filter.by_property("dataset_id").equal(dataset_id) & 
                Filter.by_property("status").equal(LifecycleState.PRODUCTION.value)
            )
        )
        
        for obj in results.objects:
            if obj.properties["generation_id"] == exclude_gen_id:
                continue
            
            # Update to DEPRECATED
            collection.data.update(
                uuid=obj.uuid,
                properties={
                    "status": LifecycleState.DEPRECATED.value,
                    "updated_at": datetime.utcnow()
                }
            )

    def get_production_index(self, dataset_name: str) -> Optional[IndexGeneration]:
        """
        Finds the active PRODUCTION index for a given dataset name.
        Do this by joining Dataset and Generation (simulated join).
        """
        # 1. Find dataset ID by name
        ds_coll = self.client.collections.get(DATASET_CLASS)
        ds_res = ds_coll.query.fetch_objects(
            filters=Filter.by_property("name").equal(dataset_name),
            limit=1
            # In a real system we'd pick a specific version, but here we assume dataset_name implies a logical group
            # Actually, the user might want "latest production index for dataset 'Wiki'".
        )
        
        if not ds_res.objects:
            return None
            
        # If there are multiple versions of the dataset, we might have multiple IDs.
        # But generally, we want the *index* that is in PRODUCTION for *any* version of this dataset?
        # Or usually, a dataset is "Wiki v1" and "Wiki v2".
        # Let's assume we search for all datasets with that name.
        
        # Simpler approach: Iterate all datasets with that name, find their ID, search for production index.
        # Or, we query Generation where status=PRODUCTION and dataset_id IN (ids).
        
        # For this reference implementation, let's assume we want the *latest* updated PRODUCTION index 
        # associated with any version of the dataset named `dataset_name`.
        
        target_dataset_ids = []
        ds_res = ds_coll.query.fetch_objects(
            filters=Filter.by_property("name").equal(dataset_name),
            limit=100
        )
        for obj in ds_res.objects:
            target_dataset_ids.append(obj.properties["dataset_id"])
            
        if not target_dataset_ids:
            return None

        # 2. Find Generation with status=PRODUCTION and dataset_id in target_dataset_ids
        gen_coll = self.client.collections.get(GENERATION_CLASS)
        
        # specific filter construction
        dataset_filter = Filter.by_property("dataset_id").equal(target_dataset_ids[0])
        for ds_id in target_dataset_ids[1:]:
            dataset_filter = dataset_filter | Filter.by_property("dataset_id").equal(ds_id)
            
        final_filter = (
            Filter.by_property("status").equal(LifecycleState.PRODUCTION.value) &
            dataset_filter
        )
        
        gen_res = gen_coll.query.fetch_objects(
            filters=final_filter,
            sort=weaviate.classes.query.Sort.by_property("updated_at", ascending=False),
            limit=1
        )
        
        if not gen_res.objects:
            return None
            
        props = gen_res.objects[0].properties
        return IndexGeneration(
            id=props["generation_id"],
            dataset_id=props["dataset_id"],
            config_id=props["config_id"],
            status=LifecycleState(props["status"]),
            weaviate_collection_name=props["weaviate_collection_name"],
            created_at=props["created_at"],
            updated_at=props["updated_at"]
        )
