from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from .lifecycle.manager import WeaviateRAGLifecycleManager
from .lifecycle.states import LifecycleState

class LifecycleAwareRetriever(BaseRetriever):
    """
    A LangChain Retriever that automatically queries the current PRODUCTION index
    managed by the WeaviateRAGLifecycleManager.
    """
    lifecycle_manager: Any  # Typed as Any to avoid circular imports or complex type hinting with Pydantic v1/v2 mismatches
    dataset_name: str
    search_type: str = "near_text"  # "near_text", "bm25", "hybrid"
    search_kwargs: dict = {}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Dynamically finds the production index and queries it.
        """
        # 1. Resolve Production Index
        prod_index = self.lifecycle_manager.get_production_index(self.dataset_name)
        
        if not prod_index:
            print(f"WARNING: No PRODUCTION index found for dataset '{self.dataset_name}'. Returning empty results.")
            return []
            
        collection_name = prod_index.weaviate_collection_name
        
        # 2. Execute Query against that collection
        client = self.lifecycle_manager.client
        collection = client.collections.get(collection_name)
        
        limit_val = self.search_kwargs.get("k", 4)
        
        try:
            if self.search_type == "bm25":
                response = collection.query.bm25(
                    query=query,
                    limit=limit_val
                )
            elif self.search_type == "near_text":
                response = collection.query.near_text(
                    query=query,
                    limit=limit_val
                )
            else:
                 # Default to near_text or handle other types
                response = collection.query.near_text(
                    query=query,
                    limit=limit_val
                )
        except Exception as e:
            print(f"Error querying collection {collection_name} with {self.search_type}: {e}")
            return []
            
        # 3. Convert to LangChain Documents
            
        # 3. Convert to LangChain Documents
        documents = []
        for obj in response.objects:
            # We assume the stored object has a 'text' property or similar.
            # We blindly copy all properties to metadata and try to find a page_content.
            
            content = obj.properties.get("text", "") or obj.properties.get("content", "")
            if not content:
                # Fallback: use the string representation of properties
                content = str(obj.properties)
                
            doc = Document(
                page_content=content,
                metadata=obj.properties
            )
            documents.append(doc)
            
        return documents
