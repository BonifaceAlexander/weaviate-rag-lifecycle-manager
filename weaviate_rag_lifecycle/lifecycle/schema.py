import weaviate
import weaviate.classes.config as wvc

DATASET_CLASS = "RAGDataset"
CONFIG_CLASS = "EmbeddingConfig"
GENERATION_CLASS = "IndexGeneration"

def init_schemas(client: weaviate.WeaviateClient):
    """
    Initializes the schema for the RAG Lifecycle Manager.
    """
    collections = client.collections.list_all()
    
    # RAGDataset
    if DATASET_CLASS not in collections:
        client.collections.create(
            name=DATASET_CLASS,
            properties=[
                wvc.Property(name="dataset_id", data_type=wvc.DataType.TEXT),
                wvc.Property(name="name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="version", data_type=wvc.DataType.TEXT),
                wvc.Property(name="created_at", data_type=wvc.DataType.DATE),
            ]
        )

    # EmbeddingConfig
    if CONFIG_CLASS not in collections:
        client.collections.create(
            name=CONFIG_CLASS,
            properties=[
                wvc.Property(name="config_id", data_type=wvc.DataType.TEXT),
                wvc.Property(name="model_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="chunk_size", data_type=wvc.DataType.INT),
                wvc.Property(name="chunk_overlap", data_type=wvc.DataType.INT),
                wvc.Property(name="created_at", data_type=wvc.DataType.DATE),
            ]
        )

    # IndexGeneration
    if GENERATION_CLASS not in collections:
        client.collections.create(
            name=GENERATION_CLASS,
            properties=[
                wvc.Property(name="generation_id", data_type=wvc.DataType.TEXT),
                wvc.Property(name="dataset_id", data_type=wvc.DataType.TEXT),
                wvc.Property(name="config_id", data_type=wvc.DataType.TEXT),
                wvc.Property(name="status", data_type=wvc.DataType.TEXT),
                wvc.Property(name="weaviate_collection_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="created_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="updated_at", data_type=wvc.DataType.DATE),
            ],
            references=[
               wvc.ReferenceProperty(name="hasDataset", target_collection=DATASET_CLASS),
               wvc.ReferenceProperty(name="hasConfig", target_collection=CONFIG_CLASS),
            ]
        )
