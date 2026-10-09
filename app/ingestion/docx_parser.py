import os
import re
import uuid
from typing import List, Dict, Any, Optional
from docx import Document
from app.models.chunk import DocumentChunk, ChunkMetadata
from app.ingestion.media_extractor import media_extractor
from app.core.logger import logger


def extract_rich_markdown_paragraph(paragraph) -> str:
    """
    Extracts text from a docx paragraph while preserving bold, italic, and bullet formats.
    """
    runs_text = []
    for run in paragraph.runs:
        t = run.text
        if not t:
            continue
        # Preserve whitespace while wrapping formatting tokens
        leading_space = " " if t.startswith(" ") and not t.strip().startswith("*") else ""
        trailing_space = " " if t.endswith(" ") and not t.strip().endswith("*") else ""
        clean_t = t.strip()

        if not clean_t:
            runs_text.append(t)
            continue

        if run.bold and run.italic:
            runs_text.append(f"{leading_space}***{clean_t}***{trailing_space}")
        elif run.bold:
            runs_text.append(f"{leading_space}**{clean_t}**{trailing_space}")
        elif run.italic:
            runs_text.append(f"{leading_space}*{clean_t}*{trailing_space}")
        else:
            runs_text.append(t)

    raw_text = "".join(runs_text).strip()
    if not raw_text:
        return ""

    # Normalize standard headings & steps if not already bolded
    raw_text = re.sub(r"^(Bước\s+\d+:\s*)", r"**\1**", raw_text)
    raw_text = re.sub(r"^\*\*(Bước\s+\d+:\s*)\*\*", r"**\1**", raw_text)  # avoid double bolding
    raw_text = re.sub(r"^(Lưu\s*ý:\s*)", r"***\1***", raw_text, flags=re.IGNORECASE)

    # Bullet point styling
    if paragraph._p.xpath("./w:pPr/w:numPr") or raw_text.startswith("•") or raw_text.startswith("-"):
        clean_bullet = re.sub(r"^[•\-]\s*", "", raw_text)
        return f"- {clean_bullet}"

    return raw_text


def sanitize_ascii_slug(text: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_str = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', ascii_str).lower()
    return re.sub(r'_+', '_', clean).strip('_')


class DocxParser:
    def parse_docx(self, file_path: str) -> List[DocumentChunk]:
        """
        Parses docx document hierarchically by functional business modules.
        Encapsulates each module into a high-fidelity Rich Markdown chunk
        preserving all bold/italic formatting, step structures, and image placeholders.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = Document(file_path)
        doc_filename = os.path.basename(file_path)
        doc_prefix = sanitize_ascii_slug(os.path.splitext(doc_filename)[0])[:30]

        # 1. Extract and map all images by relationship ID (rId)
        rid_to_url = media_extractor.extract_and_map_images(doc, doc_prefix=doc_prefix)

        SUBHEADING_IGNORE = [
            "video hướng dẫn", "link video", "mô tả", "hướng dẫn"
        ]

        COVER_EXACT = ["mục lục", "phiên bản phường/xã", "tổng quan tài liệu", "tổng quan dự án", "thành phố hồ chí minh", "tp hcm, năm 2023", "tp hcm, năm 2026"]
        COVER_CONTAINS = ["uỷ ban nhân dân", "sở nội vụ", "tài liệu hướng dẫn sử dụng", "quản lý thông tin tạo lập cơ sở dữ liệu", "thành phố hồ chí minh", "năm 2023", "năm 2026", "cơ sở dữ liệu"]

        KNOWN_IMAGE_CAPTIONS = [
            "màn hình trang chủ", "màn hình đăng ký", "popup thông tin đăng nhập",
            "màn hình đăng nhập", "giao diện thay đổi mật khẩu", "giao diện thay đổi thông tin",
            "báo cáo định kỳ", "báo cáo tnlđ", "báo cáo atvslđ", "thống kê số liệu"
        ]

        sections: List[Dict[str, Any]] = []
        current_section: Optional[Dict[str, Any]] = None
        current_module = ""

        for p_idx, paragraph in enumerate(doc.paragraphs):
            plain_text = paragraph.text.strip()
            rich_text = extract_rich_markdown_paragraph(paragraph)
            p_rids = paragraph._p.xpath('.//a:blip/@r:embed')
            p_images = [rid_to_url[rId] for rId in p_rids if rId in rid_to_url]
            style_name = paragraph.style.name.lower() if paragraph.style else ""

            is_h1 = "heading 1" in style_name
            is_h2 = "heading 2" in style_name

            # Check if this is a sub-module under Báo cáo định kỳ
            if re.match(r"^\d+\.\s*(tai nạn lao động|an toàn vệ sinh lao động)", plain_text, re.IGNORECASE):
                is_h2 = True
                is_h1 = False

            # If it's a step (e.g. Bước 1: ...), it should NOT trigger a new section split
            if plain_text and re.match(r"^bước\s+\d+", plain_text.lower()):
                is_h1 = False
                is_h2 = False

            # Detect custom uppercase, Roman numeral headings, or explicit Support Contact section
            if not (is_h1 or is_h2):
                if plain_text and len(plain_text) < 60:
                    if "liên hệ hỗ trợ" in plain_text.lower():
                        is_h1 = True
                    elif re.match(r"^(I|II|III|IV|V|VI)\.\s+", plain_text, re.IGNORECASE) or (plain_text.isupper() and len(plain_text.split()) <= 6 and not plain_text.startswith("MỤC LỤC")):
                        if p_idx > 10:  # Avoid document cover header
                            is_h1 = True

            should_split = False
            new_title = plain_text
            new_module = current_module
            new_sub_module = None
            new_level = 1

            if plain_text and (is_h1 or is_h2):
                clean_text_lower = plain_text.lower().strip()
                is_contact_section = "liên hệ hỗ trợ" in clean_text_lower
                
                # Check cover page titles only in initial paragraphs
                is_cover = False
                if p_idx < 30 and not is_contact_section:
                    if any(c in clean_text_lower for c in COVER_CONTAINS) or any(clean_text_lower == c for c in COVER_EXACT):
                        is_cover = True

                is_minor_subheading = any(
                    clean_text_lower == ig or clean_text_lower.startswith(ig)
                    for ig in SUBHEADING_IGNORE
                )

                if not is_cover and not is_minor_subheading:
                    should_split = True
                    if is_contact_section:
                        new_module = "LIÊN HỆ HỖ TRỢ"
                        new_sub_module = None
                        new_title = "LIÊN HỆ HỖ TRỢ"
                        new_level = 1
                        current_module = "LIÊN HỆ HỖ TRỢ"
                    elif is_h1:
                        new_module = plain_text
                        new_sub_module = None
                        new_title = plain_text
                        new_level = 1
                        current_module = plain_text
                    else:
                        new_module = current_module
                        new_sub_module = plain_text
                        new_title = f"{current_module} - {plain_text}"
                        new_level = 2

            if should_split:
                if current_section and (current_section["paragraphs"] or current_section["image_urls"]):
                    sections.append(current_section)

                current_section = {
                    "module": new_module,
                    "sub_module": new_sub_module,
                    "section_title": new_title,
                    "heading_level": new_level,
                    "paragraphs": [],
                    "image_urls": [],
                    "youtube_info": None,
                    "keywords": [],
                    "ai_notes": []
                }

            if current_section is not None:
                if plain_text:
                    # Extract search keywords and AI notes into metadata
                    kw_matches = re.findall(r"\[Từ khóa tìm kiếm:\s*(.*?)\]", plain_text, flags=re.IGNORECASE)
                    if kw_matches:
                        for kw_str in kw_matches:
                            current_section["keywords"].extend([k.strip() for k in re.split(r"[\?;,]\s*", kw_str) if k.strip()])

                    ai_matches = re.findall(r"\[(?:Ghi chú AI|Ghi chú dành cho Trợ lý ảo[^\]]*):\s*(.*?)\]", plain_text, flags=re.IGNORECASE)
                    if ai_matches:
                        for note in ai_matches:
                            current_section["ai_notes"].append(note.strip())

                    # Clean internal AI notes and search keywords from user-facing text
                    cleaned_rich_text = re.sub(r"\[Từ khóa tìm kiếm:[^\]]*\]", "", rich_text, flags=re.IGNORECASE)
                    cleaned_rich_text = re.sub(r"\[(?:Ghi chú AI|Ghi chú dành cho Trợ lý ảo[^\]]*):[^\]]*\]", "", cleaned_rich_text, flags=re.IGNORECASE)
                    cleaned_rich_text = re.sub(r"\[(?:Ghi chú AI|Ghi chú dành cho Trợ lý ảo[^\]]*)\](?:\s*:\s*[^\n]+)?", "", cleaned_rich_text, flags=re.IGNORECASE)
                    cleaned_rich_text = cleaned_rich_text.strip()

                    # Check for YouTube link
                    if "youtube.com" in plain_text or "youtu.be" in plain_text:
                        yt = media_extractor.extract_youtube_info(plain_text)
                        if yt:
                            current_section["youtube_info"] = yt
                        if "[VIDEO]" not in "\n".join(current_section["paragraphs"]):
                            current_section["paragraphs"].append("[VIDEO]")
                    else:
                        # Check if this paragraph is an image caption
                        clean_lower = plain_text.lower().strip()
                        is_caption = any(cap in clean_lower for cap in KNOWN_IMAGE_CAPTIONS) or (len(plain_text) < 40 and not plain_text.startswith("Bước"))
                        if is_caption and current_section["paragraphs"] and current_section["paragraphs"][-1].startswith("[IMAGE_"):
                            # Attach as italic caption directly under image tag
                            current_section["paragraphs"].append(f"*{plain_text.strip()}*")
                        elif not any(clean_lower == ig for ig in SUBHEADING_IGNORE):
                            if cleaned_rich_text:
                                current_section["paragraphs"].append(cleaned_rich_text)

                if p_images:
                    for img in p_images:
                        if img not in current_section["image_urls"]:
                            current_section["image_urls"].append(img)
                            img_idx = len(current_section["image_urls"])
                            current_section["paragraphs"].append(f"[IMAGE_{img_idx}]")

        if current_section and (current_section["paragraphs"] or current_section["image_urls"]):
            sections.append(current_section)

        # Convert sections to structured 2-Tier chunks (Parent & Child) with Rich Markdown and Media URLs
        chunks: List[DocumentChunk] = []
        for i, sec in enumerate(sections):
            raw_text = "\n\n".join(sec["paragraphs"]).strip()
            if not raw_text and not sec["image_urls"]:
                continue

            module_str = sec["module"]
            sub_str = f" - {sec['sub_module']}" if sec["sub_module"] else ""
            header = f"### {module_str}{sub_str}\n"
            content = header + raw_text

            parent_id = f"{doc_prefix}_p{i}_{uuid.uuid4().hex[:6]}"
            section_title = sec.get("section_title", module_str)

            # 1. Parent Chunk (Full functional module for Mode A)
            parent_chunk = DocumentChunk(
                id=parent_id,
                text_content=content,
                metadata=ChunkMetadata(
                    module=module_str,
                    sub_module=sec.get("sub_module"),
                    section_title=section_title,
                    heading_level=sec.get("heading_level", 1),
                    document_source=doc_filename,
                    image_urls=sec["image_urls"],
                    youtube_info=sec["youtube_info"],
                    parent_id=None,
                    section_breadcrumb=f"[{section_title}]",
                    is_child=False,
                    extra_attributes={
                        "keywords": sec.get("keywords", []),
                        "ai_notes": sec.get("ai_notes", [])
                    }
                )
            )
            chunks.append(parent_chunk)

            # 2. Child Chunks (Step-level / Sub-section granularity for Mode B)
            child_chunks = self._extract_child_chunks(
                section_raw=raw_text,
                parent_id=parent_id,
                parent_module=module_str,
                doc_filename=doc_filename,
                doc_prefix=doc_prefix,
                section_idx=i,
                section_images=sec["image_urls"],
                section_youtube=sec["youtube_info"]
            )
            chunks.extend(child_chunks)

        logger.info(f"Parsed {len(chunks)} Rich Markdown chunks from {doc_filename} with {sum(len(c.metadata.image_urls) for c in chunks)} mapped images")
        return chunks

    def _extract_child_chunks(
        self,
        section_raw: str,
        parent_id: str,
        parent_module: str,
        doc_filename: str,
        doc_prefix: str,
        section_idx: int,
        section_images: List[str],
        section_youtube: Optional[Any]
    ) -> List[DocumentChunk]:
        """Tách từng Bước (Step) hoặc Mục con thành các Child Chunk nhỏ gọn (50-150 tokens) cho Mode B."""
        children: List[DocumentChunk] = []
        step_pattern = r"(?m)^(?:(?:\*\*Bước\s+\d+[:\.]?\*\*)|(?:Bước\s+\d+[:\.]))\s*(.*)$"
        step_matches = list(re.finditer(step_pattern, section_raw, re.IGNORECASE))

        if step_matches:
            # Intro chunk before Step 1 if substantive
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
                            image_urls=[],
                            youtube_info=section_youtube,
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

                # Extract images mapped to this step
                img_indices = [int(idx_str) - 1 for idx_str in re.findall(r"\[IMAGE_(\d+)\]", step_body)]
                step_images = [section_images[idx] for idx in img_indices if 0 <= idx < len(section_images)]

                has_video = "[VIDEO]" in step_body
                step_yt = section_youtube if has_video else None

                child_chunk = DocumentChunk(
                    id=f"{doc_prefix}_p{section_idx}_step{s_idx + 1}_{uuid.uuid4().hex[:6]}",
                    text_content=child_text,
                    metadata=ChunkMetadata(
                        module=parent_module,
                        sub_module=clean_step_header,
                        section_title=f"{parent_module} - {clean_step_header}",
                        heading_level=2,
                        document_source=doc_filename,
                        image_urls=step_images,
                        youtube_info=step_yt,
                        parent_id=parent_id,
                        section_breadcrumb=breadcrumb,
                        is_child=True
                    )
                )
                children.append(child_chunk)
        else:
            # Single non-step child chunk
            breadcrumb = f"[Phân hệ: {parent_module}]"
            children.append(DocumentChunk(
                id=f"{doc_prefix}_p{section_idx}_c0_{uuid.uuid4().hex[:6]}",
                text_content=f"{breadcrumb}\nNội dung: {section_raw}",
                metadata=ChunkMetadata(
                    module=parent_module,
                    sub_module=None,
                    section_title=parent_module,
                    heading_level=2,
                    document_source=doc_filename,
                    image_urls=section_images,
                    youtube_info=section_youtube,
                    parent_id=parent_id,
                    section_breadcrumb=breadcrumb,
                    is_child=True
                )
            ))

        return children


docx_parser = DocxParser()
