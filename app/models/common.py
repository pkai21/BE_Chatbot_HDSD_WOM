from pydantic import BaseModel, Field
from typing import TypeVar, Generic, Optional, Any

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool = Field(True, description="Trạng thái thành công")
    message: str = Field("OK", description="Thông báo kết quả")
    data: Optional[T] = Field(None, description="Dữ liệu trả về")
    error: Optional[str] = Field(None, description="Chi tiết lỗi nếu có")
