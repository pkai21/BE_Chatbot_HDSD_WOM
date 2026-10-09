import os
import re
from typing import List, Optional, Dict
from docx.document import Document as DocxDocument
from app.models.chunk import YouTubeInfo
from app.core.config import settings
from app.core.logger import logger


class MediaExtractor:
    def __init__(self, output_dir: str = settings.IMAGE_STORAGE_PATH):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_youtube_info(self, text: str) -> Optional[YouTubeInfo]:
        """
        Extracts YouTube URLs and timestamp parameters from text.
        Supports https://www.youtube.com/watch?v=xxx&t=120s or https://youtu.be/xxx?t=120
        """
        yt_regex = r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+|youtu\.be/[\w-]+)(?:[^\s\)]+)?)"
        match = re.search(yt_regex, text)
        if not match:
            return None

        url = match.group(1)
        timestamp_start = 0
        t_match = re.search(r"[?&]t=(\d+)s?", url)
        if t_match:
            timestamp_start = int(t_match.group(1))

        return YouTubeInfo(
            video_url=url.split("?")[0].split("&t=")[0],
            timestamp_start=timestamp_start,
            display_link=url,
            title="Video hướng dẫn thao tác"
        )

    def extract_and_map_images(self, doc: DocxDocument, doc_prefix: str = "doc") -> Dict[str, str]:
        """
        Extracts all embedded images from docx and maps relationship ID (rId) -> saved image URL.
        """
        rid_to_url = {}
        try:
            for idx, (rId, part) in enumerate(doc.part.related_parts.items()):
                if "image" in part.content_type:
                    ext = part.content_type.split("/")[-1]
                    if ext == "jpeg":
                        ext = "jpg"
                    elif ext == "x-png":
                        ext = "png"
                    
                    filename = f"{doc_prefix}_img_{idx}_{rId}.{ext}"
                    filepath = os.path.join(self.output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(part.blob)

                    import urllib.parse
                    safe_filename = urllib.parse.quote(filename)
                    cdn_url = f"{settings.BASE_IMAGE_CDN_URL}/{safe_filename}"
                    rid_to_url[rId] = cdn_url
                    
            logger.info(f"Extracted {len(rid_to_url)} images from docx to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error extracting images from docx: {e}", exc_info=True)

        return rid_to_url


media_extractor = MediaExtractor()
