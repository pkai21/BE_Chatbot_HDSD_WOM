import os
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from app.models.chunk import DocumentChunk, ChunkMetadata, YouTubeInfo
from app.ingestion.media_extractor import media_extractor
from app.core.logger import logger


class MarkdownParser:
    """
    Parser phân cấp ngữ nghĩa 2 tầng cho tài liệu Markdown chuẩn AST:
    - Tầng 1 (Parent Chunk): Nhóm toàn bộ nội dung của từng Heading 1 (#) thành 1 Chunk cha hoàn chỉnh (dành cho Mode A).
    - Tầng 2 (Child Chunk): Tách từng Bước (Step), từng tiểu mục con (##) thành các Chunk con (50-150 tokens)
      mang parent_id, is_child=True và section_breadcrumb (dành cho Mode B).
    """

    def parse_markdown(self, file_path: str) -> List[DocumentChunk]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_name = os.path.basename(file_path)
        doc_prefix = re.sub(r"[^\w\-_]", "_", os.path.splitext(file_name)[0])[:30]

        chunks: List[DocumentChunk] = []
        
        # Split into main sections by Heading 1 (# )
        h1_pattern = r"(?m)^#\s+(.+)$"
        h1_matches = list(re.finditer(h1_pattern, content))

        if not h1_matches:
            parent_id = f"{doc_prefix}_p0_{uuid.uuid4().hex[:6]}"
            p_chunk = DocumentChunk(
                id=parent_id,
                text_content=content.strip(),
                metadata=ChunkMetadata(
                    module=file_name,
                    section_title=file_name,
                    heading_level=1,
                    document_source=file_name,
                    is_child=False
                )
            )
            chunks.append(p_chunk)
            return chunks

        for idx, match in enumerate(h1_matches):
            h1_title = match.group(1).strip()
            clean_h1_title = re.sub(r"[\*\_]", "", h1_title).strip()
            
            start_pos = match.end()
            end_pos = h1_matches[idx + 1].start() if idx + 1 < len(h1_matches) else len(content)
            section_raw = content[start_pos:end_pos].strip()

            # 1. Parse Parent Chunk (Full section content)
            parent_id = f"{doc_prefix}_p{idx}_{uuid.uuid4().hex[:6]}"
            
            keywords = []
            ai_notes = []
            for kw_str in re.findall(r"\[Từ khóa tìm kiếm:\s*(.*?)\]", section_raw, flags=re.IGNORECASE):
                keywords.extend([k.strip() for k in re.split(r"[\?;,]\s*", kw_str) if k.strip()])
            for note in re.findall(r"\[(?:Ghi chú AI|Ghi chú dành cho Trợ lý ảo[^\]]*):\s*(.*?)\]", section_raw, flags=re.IGNORECASE):
                ai_notes.append(note.strip())

            image_urls = re.findall(r"!\[.*?\]\((.*?)\)", section_raw)
            youtube_matches = re.findall(r"(https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s\)\>]+)", section_raw)
            youtube_info = None
            if youtube_matches:
                youtube_info = media_extractor.extract_youtube_info(youtube_matches[0])

            parent_header = f"### {clean_h1_title}\n"
            parent_text = parent_header + section_raw

            parent_chunk = DocumentChunk(
                id=parent_id,
                text_content=parent_text,
                metadata=ChunkMetadata(
                    module=clean_h1_title,
                    sub_module=None,
                    section_title=clean_h1_title,
                    heading_level=1,
                    document_source=file_name,
                    image_urls=image_urls,
                    youtube_info=youtube_info,
                    parent_id=None,
                    section_breadcrumb=f"[{clean_h1_title}]",
                    is_child=False,
                    extra_attributes={
                        "keywords": keywords,
                        "ai_notes": ai_notes
                    }
                )
            )
            chunks.append(parent_chunk)

            # 2. Parse Child Chunks for this section
            child_chunks = self._extract_child_chunks(
                section_raw=section_raw,
                parent_id=parent_id,
                parent_module=clean_h1_title,
                doc_filename=file_name,
                doc_prefix=doc_prefix,
                section_idx=idx
            )
            chunks.extend(child_chunks)

        logger.info(f"MarkdownParser parsed {len(chunks)} total chunks ({len([c for c in chunks if not c.metadata.is_child])} parents, {len([c for c in chunks if c.metadata.is_child])} children) from {file_name}")
        return chunks

    def _extract_child_chunks(
        self,
        section_raw: str,
        parent_id: str,
        parent_module: str,
        doc_filename: str,
        doc_prefix: str,
        section_idx: int
    ) -> List[DocumentChunk]:
        """Tách các bước (Steps) hoặc Heading 2 (##) thành các Child Chunk nhỏ gọn."""
        children: List[DocumentChunk] = []

        h2_pattern = r"(?m)^##\s+(.+)$"
        step_pattern = r"(?m)^(?:(?:\*\*Bước\s+\d+[:\.]?\*\*)|(?:Bước\s+\d+[:\.]))\s*(.*)$"
        
        h2_matches = list(re.finditer(h2_pattern, section_raw))
        if h2_matches:
            for c_idx, match in enumerate(h2_matches):
                sub_title = re.sub(r"[\*\_]", "", match.group(1)).strip()
                c_start = match.end()
                c_end = h2_matches[c_idx + 1].start() if c_idx + 1 < len(h2_matches) else len(section_raw)
                child_body = section_raw[c_start:c_end].strip()

                if not child_body:
                    continue

                breadcrumb = f"[Phân hệ: {parent_module} > {sub_title}]"
                child_text = f"{breadcrumb}\nNội dung: {child_body}"

                c_images = re.findall(r"!\[.*?\]\((.*?)\)", child_body)
                c_yt_matches = re.findall(r"(https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s\)\>]+)", child_body)
                c_yt = media_extractor.extract_youtube_info(c_yt_matches[0]) if c_yt_matches else None

                child_chunk = DocumentChunk(
                    id=f"{doc_prefix}_p{section_idx}_c{c_idx}_{uuid.uuid4().hex[:6]}",
                    text_content=child_text,
                    metadata=ChunkMetadata(
                        module=parent_module,
                        sub_module=sub_title,
                        section_title=f"{parent_module} - {sub_title}",
                        heading_level=2,
                        document_source=doc_filename,
                        image_urls=c_images,
                        youtube_info=c_yt,
                        parent_id=parent_id,
                        section_breadcrumb=breadcrumb,
                        is_child=True
                    )
                )
                children.append(child_chunk)
            return children

        step_matches = list(re.finditer(step_pattern, section_raw, re.IGNORECASE))
        if step_matches:
            if step_matches[0].start() > 0:
                intro_text = section_raw[:step_matches[0].start()].strip()
                if len(intro_text) > 30:
                    breadcrumb = f"[Phân hệ: {parent_module} > Tổng quan]"
                    children.append(DocumentChunk(
                        id=f"{doc_prefix}_p{section_idx}_c_intro_{uuid.uuid4().hex[:6]}",
                        text_content=f"{breadcrumb}\nNội dung: {intro_text}",
                        metadata=ChunkMetadata(
                            module=parent_module,
                            sub_module="Tổng quan",
                            section_title=f"{parent_module} - Tổng quan",
                            heading_level=2,
                            document_source=doc_filename,
                            parent_id=parent_id,
                            section_breadcrumb=breadcrumb,
                            is_child=True
                        )
                    ))

            for s_idx, match in enumerate(step_matches):
                step_header = match.group(0).strip()
                clean_step_header = re.sub(r"[\*\_]", "", step_header).strip()
                s_start = match.start()
                s_end = step_matches[s_idx + 1].start() if s_idx + 1 < len(step_matches) else len(section_raw)
                step_body = section_raw[s_start:s_end].strip()

                breadcrumb = f"[Phân hệ: {parent_module} > {clean_step_header}]"
                child_text = f"{breadcrumb}\nNội dung: {step_body}"

                c_images = re.findall(r"!\[.*?\]\((.*?)\)", step_body)
                c_yt_matches = re.findall(r"(https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s\)\>]+)", step_body)
                c_yt = media_extractor.extract_youtube_info(c_yt_matches[0]) if c_yt_matches else None

                child_chunk = DocumentChunk(
                    id=f"{doc_prefix}_p{section_idx}_step{s_idx + 1}_{uuid.uuid4().hex[:6]}",
                    text_content=child_text,
                    metadata=ChunkMetadata(
                        module=parent_module,
                        sub_module=clean_step_header,
                        section_title=f"{parent_module} - {clean_step_header}",
                        heading_level=2,
                        document_source=doc_filename,
                        image_urls=c_images,
                        youtube_info=c_yt,
                        parent_id=parent_id,
                        section_breadcrumb=breadcrumb,
                        is_child=True
                    )
                )
                children.append(child_chunk)
            return children

        paragraphs = [p.strip() for p in section_raw.split("\n\n") if p.strip()]
        for p_idx, para in enumerate(paragraphs):
            if len(para) < 20:
                continue
            breadcrumb = f"[Phân hệ: {parent_module} > Mục {p_idx + 1}]"
            children.append(DocumentChunk(
                id=f"{doc_prefix}_p{section_idx}_p{p_idx}_{uuid.uuid4().hex[:6]}",
                text_content=f"{breadcrumb}\nNội dung: {para}",
                metadata=ChunkMetadata(
                    module=parent_module,
                    sub_module=f"Mục {p_idx + 1}",
                    section_title=f"{parent_module} - Mục {p_idx + 1}",
                    heading_level=2,
                    document_source=doc_filename,
                    parent_id=parent_id,
                    section_breadcrumb=breadcrumb,
                    is_child=True
                )
            ))

        return children


md_parser = MarkdownParser()
