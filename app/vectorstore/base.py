from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.chunk import DocumentChunk


class BaseVectorStore(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Stores chunks into vector store."""
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 3, filter_dict: Optional[dict] = None) -> List[DocumentChunk]:
        """Performs vector / hybrid search for relevant chunks."""
        pass

    @abstractmethod
    async def delete_collection(self) -> bool:
        """Deletes all items in collection."""
        pass
