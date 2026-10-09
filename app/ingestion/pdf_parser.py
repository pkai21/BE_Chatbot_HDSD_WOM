import os
import re
import uuid
from typing import List, Dict, Any
import pymupdf
from app.models.chunk import DocumentChunk, ChunkMetadata
from app.core.logger import logger


class PDFParser:
    def parse_pdf(self, file_path: str, max_chunk_chars: int = 1000) -> List[DocumentChunk]:
        """
        Parses a PDF document, extracts text page by page / section by section,
        and constructs structured DocumentChunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = pymupdf.open(file_path)
        doc_filename = os.path.basename(file_path)
        doc_prefix = re.sub(r"[^\w\-_]", "_", os.path.splitext(doc_filename)[0])[:25]

        chunks: List[DocumentChunk] = []
        current_module = "Quản lý Điều hành Cấp Xã"
        
        full_text_pages = []
        for page_idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                continue
            full_text_pages.append((page_idx + 1, text))

        # Chunk by page / logical sections
        for page_num, page_text in full_text_pages:
            # Detect section title if any on the top lines of the page
            lines = [l.strip() for l in page_text.split("\n") if l.strip()]
            section_title = f"Trang {page_num}"
            if lines:
                for line in lines[:3]:
                    if len(line) < 80 and (line.isupper() or re.match(r"^[I|V|X|\d+\.]", line)):
                        section_title = line
                        break

            # If page text is within limit, add as one chunk
            if len(page_text) <= max_chunk_chars:
                chunk = DocumentChunk(
                    id=f"{doc_prefix}_p{page_num}_{uuid.uuid4().hex[:4]}",
                    text_content=page_text,
                    metadata=ChunkMetadata(
                        module=current_module,
                        sub_module=section_title,
                        section_title=section_title,
                        heading_level=2,
                        document_source=doc_filename,
                        image_urls=[],
                        youtube_info=None
                    )
                )
                chunks.append(chunk)
            else:
                # Split large page text into sub-chunks
                start = 0
                sub_idx = 0
                while start < len(page_text):
                    end = min(start + max_chunk_chars, len(page_text))
                    sub_content = page_text[start:end]
                    chunk = DocumentChunk(
                        id=f"{doc_prefix}_p{page_num}_sub{sub_idx}_{uuid.uuid4().hex[:4]}",
                        text_content=sub_content,
                        metadata=ChunkMetadata(
                            module=current_module,
                            sub_module=section_title,
                            section_title=f"{section_title} (Phần {sub_idx+1})",
                            heading_level=2,
                            document_source=doc_filename,
                            image_urls=[],
                            youtube_info=None
                        )
                    )
                    chunks.append(chunk)
                    sub_idx += 1
                    start += max_chunk_chars - 150

        logger.info(f"Parsed {len(chunks)} chunks from PDF: {doc_filename} ({len(doc)} pages)")
        return chunks


pdf_parser = PDFParser()
