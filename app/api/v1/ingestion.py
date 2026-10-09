import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.common import ApiResponse
from app.ingestion.docx_parser import docx_parser
from app.ingestion.chunker import chunker
from app.services.rag_service import rag_service
from app.core.logger import logger

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

UPLOAD_DIR = "./data/raw_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-docx", response_model=ApiResponse[dict])
async def upload_and_process_docx(file: UploadFile = File(...)):
    if not file.filename.endswith((".docx", ".doc")):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1. Parse docx
        raw_chunks = docx_parser.parse_docx(file_location)

        # 2. Refine Chunks
        refined_chunks = chunker.split_chunks(raw_chunks)

        # 3. Save to Vector Store
        await rag_service.vector_store.add_chunks(refined_chunks)

        return ApiResponse(
            success=True,
            message="Document uploaded, parsed, and indexed successfully",
            data={
                "filename": file.filename,
                "total_chunks": len(refined_chunks)
            }
        )
    except Exception as e:
        logger.error(f"Error processing docx: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
