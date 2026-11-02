"""FastAPI app. Run with: uvicorn backend.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.db import init_db
from backend.demo import DemoGuardMiddleware, StaticFrontendMiddleware
from backend.drives import router as drives_router
from backend.exports import router as exports_router
from backend.hazards import router as hazards_router
from backend.pipeline import recover_interrupted
from backend.results import router as results_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    if not config.demo_mode():  # the read-only demo never writes
        recover_interrupted()  # drives left "processing" by a stopped server become failed
    yield


app = FastAPI(title="RoadScout", lifespan=lifespan)

# Added innermost first: the demo guard, then the static frontend, then CORS (outermost, so a 403 carries CORS headers).
app.add_middleware(DemoGuardMiddleware)
app.add_middleware(StaticFrontendMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(drives_router)
app.include_router(results_router)
app.include_router(hazards_router)
app.include_router(exports_router)


@app.get("/api/demo")
def demo_status() -> dict[str, bool]:
    """Whether this server is the read-only hosted demo (ROADSCOUT_DEMO)."""
    return {"demo": config.demo_mode()}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
