import json
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.models.chunk import DocumentChunk, ChunkMetadata, YouTubeInfo
from app.vectorstore.base import BaseVectorStore
from app.core.config import settings
from app.core.logger import logger


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "hdsd_chunks"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        try:
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.text_content for chunk in chunks]
            metadatas = []
            for chunk in chunks:
                meta = {
                    "module": chunk.metadata.module or "",
                    "sub_module": chunk.metadata.sub_module or "",
                    "section_title": chunk.metadata.section_title,
                    "heading_level": chunk.metadata.heading_level,
                    "document_source": chunk.metadata.document_source,
                    "parent_id": chunk.metadata.parent_id or "",
                    "section_breadcrumb": chunk.metadata.section_breadcrumb or "",
                    "is_child": chunk.metadata.is_child,
                    "image_urls": json.dumps(chunk.metadata.image_urls),
                    "youtube_info": json.dumps(chunk.metadata.youtube_info.model_dump() if chunk.metadata.youtube_info else None),
                    "extra_attributes": json.dumps(chunk.metadata.extra_attributes)
                }
                metadatas.append(meta)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully added/updated {len(chunks)} chunks in ChromaDB collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}", exc_info=True)
            return False

    async def search(self, query: str, top_k: int = 3, filter_dict: Optional[dict] = None) -> List[DocumentChunk]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_dict
            )

            chunks = []
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    doc_text = results["documents"][0][i]
                    meta = results["metadatas"][0][i]

                    raw_imgs = meta.get("image_urls", "[]")
                    image_urls = json.loads(raw_imgs) if isinstance(raw_imgs, str) else raw_imgs

                    raw_yt = meta.get("youtube_info")
                    yt_dict = json.loads(raw_yt) if (isinstance(raw_yt, str) and raw_yt) else raw_yt
                    youtube_info = YouTubeInfo(**yt_dict) if yt_dict else None

                    raw_extra = meta.get("extra_attributes", "{}")
                    extra_attrs = json.loads(raw_extra) if (isinstance(raw_extra, str) and raw_extra) else (raw_extra or {})

                    metadata = ChunkMetadata(
                        module=meta.get("module"),
                        sub_module=meta.get("sub_module"),
                        section_title=meta.get("section_title", ""),
                        heading_level=meta.get("heading_level", 1),
                        document_source=meta.get("document_source", ""),
                        parent_id=meta.get("parent_id") or None,
                        section_breadcrumb=meta.get("section_breadcrumb") or None,
                        is_child=bool(meta.get("is_child", False)),
                        image_urls=image_urls,
                        youtube_info=youtube_info,
                        extra_attributes=extra_attrs
                    )

                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        text_content=doc_text,
                        metadata=metadata
                    ))
            return chunks
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
            return []

    async def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error(f"Error resetting ChromaDB collection: {e}", exc_info=True)
            return False
