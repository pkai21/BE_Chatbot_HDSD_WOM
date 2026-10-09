from fastapi import APIRouter
from app.models.common import ApiResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=ApiResponse[dict])
async def health_check():
    return ApiResponse(
        success=True,
        message="Backend API is running smoothly",
        data={
            "status": "healthy",
            "service": "HDSD Multimodal Chatbot API",
            "version": "1.0.0"
        }
    )
