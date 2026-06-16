from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router
import os
from dotenv import load_dotenv
FRONTEND_URL = os.getenv("FRONTEND_URL")
load_dotenv()

PORT = os.getenv("PORT")

app = FastAPI(
    title="Skin Lesion Classifier API",
    description="Backend API for AI-powered skin lesion analysis",
    version="1.0.0"
)
print("Backend server started successfully")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {
        "message": "Skin Lesion Classifier API Running"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

app.include_router(
    predict_router,
    prefix="/predict",
    tags=["Prediction Routes"]
)