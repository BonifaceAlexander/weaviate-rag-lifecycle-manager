import weaviate
import os
import json
from datetime import datetime

from weaviate_rag_lifecycle import WeaviateRAGLifecycleManager, LifecycleState, LifecycleAwareRetriever

def main():
    print("🚀 Starting RAG Lifecycle Demo...")

    # 1. Connect to Weaviate (Using Embedded for easy demo)
    # Ensure you do not have another Weaviate instance running on port 8080 or change ports
    print("Connecting to Weaviate (Embedded)...")
    client = weaviate.connect_to_embedded()
    
    try:
        # 2. Initialize Lifecycle Manager
        manager = WeaviateRAGLifecycleManager(client)
        manager.initialize()
        print("✅ Schema initialized.")

        # 3. Create Dataset
        dataset_name = "WikiDocs"
        version = "v1.0"
        dataset = manager.create_dataset(dataset_name, version)
        print(f"✅ Created Dataset: {dataset.name} ({dataset.version}) - ID: {dataset.id}")

        # 4. Register Embedding Config
        config = manager.register_embedding_config("openai/text-embedding-3-small", 512, 50)
        print(f"✅ Registered Config: {config.model_name} - ID: {config.id}")

        # 5. Create Index Generation (DRAFT)
        gen1 = manager.create_index_generation(dataset.id, config.id)
        print(f"✅ Created Index Generation 1 (DRAFT): {gen1.id}")
        print(f"   Physical Index: {gen1.weaviate_collection_name}")

        # 6. Simulate Indexing (we just create the collection essentially)
        # We assume the manager.create_index_generation ONLY creates the metadata record.
        # It assumes the physical index creation/feeding happens separately or we add a helper.
        # For this demo, let's manually create the physical collection to simulate reality.
        # To avoid vectorizer errors in Embedded mode without keys, we set vectorizer to none.
        import weaviate.classes.config as wvc

        if not client.collections.exists(gen1.weaviate_collection_name):
            client.collections.create(
                name=gen1.weaviate_collection_name,
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                properties=[
                    wvc.Property(name="text", data_type=wvc.DataType.TEXT)
                ]
            )
            # Add some dummy data
            coll = client.collections.get(gen1.weaviate_collection_name)
            coll.data.insert({"text": "The capital of France is Paris."})
            coll.data.insert({"text": "LangChain is great for RAG."})
            print("   (Populated Index 1 with dummy data)")

        # 7. Promote to STAGING
        manager.promote_index(gen1.id, LifecycleState.STAGING)
        print("✅ Promoted Index 1 to STAGING")

        # 8. Promote to PRODUCTION
        manager.promote_index(gen1.id, LifecycleState.PRODUCTION)
        print("✅ Promoted Index 1 to PRODUCTION")

        # 9. Verify Retriever finds it (using BM25 to avoid vectorizer need)
        retriever = LifecycleAwareRetriever(
            lifecycle_manager=manager,
            dataset_name=dataset_name,
            search_type="bm25"
        )
        
        docs = retriever.invoke("France") # BM25 works best with keyword-like queries

        print(f"\n🔍 Retriever Query Result (Index 1):")
        for d in docs:
            print(f"   - {d.page_content}")
            
        assert len(docs) > 0, "Retriever failed to find documents from Index 1"

        # 10. Create a NEW Generation (e.g. new embedding model)
        print("\n🔄 Creating Index Generation 2 (Upgrade)...")
        config2 = manager.register_embedding_config("openai/text-embedding-3-large", 512, 50)
        gen2 = manager.create_index_generation(dataset.id, config2.id)
        
        # Populate Index 2
        if not client.collections.exists(gen2.weaviate_collection_name):
            client.collections.create(
                name=gen2.weaviate_collection_name,
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                properties=[
                    wvc.Property(name="text", data_type=wvc.DataType.TEXT)
                ]
            )
            coll2 = client.collections.get(gen2.weaviate_collection_name)
            coll2.data.insert({"text": "The capital of Germany is Berlin."}) # Distinct data
            coll2.data.insert({"text": "Weaviate Lifecycle Manager is cool."})
        
        # Promote New Index to PRODUCTION (should replace old one)
        manager.promote_index(gen2.id, LifecycleState.PRODUCTION)
        print("✅ Promoted Index 2 to PRODUCTION")
        
        # 11. Verify Retriever now queries Index 2
        # Note: We might need to re-instantiate retriever or it should dynamic lookup on every call.
        # Our implementation does dynamic lookup in _get_relevant_documents, so it should work immediately.
        
        docs2 = retriever.invoke("Germany")
        print(f"\n🔍 Retriever Query Result (Index 2):")
        for d in docs2:
            print(f"   - {d.page_content}")

        assert any("Berlin" in d.page_content for d in docs2), "Retriever should now find documents from Index 2"
        
        # Verify old index is deprecated (or at least check it's not production)
        old_gen = manager.get_index_generation(gen1.id)
        print(f"\nstatus of Index 1: {old_gen.status}")
        assert old_gen.status == LifecycleState.DEPRECATED, "Old index should be DEPRECATED"

        print("\n🎉 Demo Completed Successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    main()
