from fastapi import (
    APIRouter,
    UploadFile,
    HTTPException
)

router = APIRouter()

@router.post("/")
async def predict(file: UploadFile):

    # Validate image type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    return {
        "prediction": "benign",
        "confidence": 0.87
    }