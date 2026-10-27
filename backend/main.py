"""FastAPI app. Run with: uvicorn backend.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db import init_db
from backend.drives import router as drives_router
from backend.pipeline import recover_interrupted


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    recover_interrupted()  # drives left "processing" by a stopped server become failed
    yield


app = FastAPI(title="RoadScout", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(drives_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
