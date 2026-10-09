from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class YouTubeInfo(BaseModel):
    video_url: str = Field(..., description="Original YouTube URL")
    timestamp_start: int = Field(0, description="Start timestamp in seconds")
    timestamp_end: Optional[int] = Field(None, description="End timestamp in seconds")
    display_link: str = Field(..., description="YouTube URL with timestamp parameter")
    title: Optional[str] = Field(None, description="Title/topic of the video segment")


class ChunkMetadata(BaseModel):
    module: Optional[str] = Field(None, description="Tên phân hệ / module lớn")
    sub_module: Optional[str] = Field(None, description="Tên chức năng / module con")
    section_title: str = Field(..., description="Tiêu đề mục / Heading trong tài liệu HDSD")
    heading_level: int = Field(1, description="Cấp độ Heading (1, 2, 3...)")
    document_source: str = Field(..., description="Tên file tài liệu gốc")
    image_urls: List[str] = Field(default_factory=list, description="Danh sách URL ảnh UI tương ứng")
    youtube_info: Optional[YouTubeInfo] = Field(None, description="Thông tin video YouTube và timestamp")
    parent_id: Optional[str] = Field(None, description="ID của Parent Chunk tương ứng nếu đây là Child Chunk")
    section_breadcrumb: Optional[str] = Field(None, description="Breadcrumb phân cấp vị trí mục (VD: [Phân hệ: ... > Bước 1])")
    is_child: bool = Field(False, description="True nếu là Child Chunk (Mode B), False nếu là Parent Chunk (Mode A)")
    extra_attributes: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    id: str = Field(..., description="Định danh duy nhất của chunk")
    text_content: str = Field(..., description="Nội dung văn bản hướng dẫn")
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = Field(None, description="Dense embedding vector")
