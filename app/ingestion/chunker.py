from typing import List
from app.models.chunk import DocumentChunk
from app.core.logger import logger


class Chunker:
    def __init__(self, max_chunk_size: int = 4000, chunk_overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def split_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Splits large section chunks if necessary while preserving metadata.
        """
        refined_chunks = []
        for chunk in chunks:
            if len(chunk.text_content) <= self.max_chunk_size:
                refined_chunks.append(chunk)
            else:
                # Split content into smaller pieces
                text = chunk.text_content
                start = 0
                idx = 0
                while start < len(text):
                    end = min(start + self.max_chunk_size, len(text))
                    sub_text = text[start:end]
                    new_chunk = DocumentChunk(
                        id=f"{chunk.id}_sub_{idx}",
                        text_content=sub_text,
                        metadata=chunk.metadata
                    )
                    refined_chunks.append(new_chunk)
                    idx += 1
                    start += self.max_chunk_size - self.chunk_overlap

        logger.info(f"Refined {len(chunks)} original chunks into {len(refined_chunks)} sub-chunks")
        return refined_chunks


chunker = Chunker()
