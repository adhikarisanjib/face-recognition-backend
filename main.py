from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config.database import Base, engine
from app.config.settings import settings
from app.models import *
from app.routers.auth import router as auth_router
from app.routers.face import router as face_router


# Create tables on startup
async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await create_tables()
    yield
    await engine.dispose()


app = FastAPI(
    title="Face Recognition System",
    description="A simple face recognition system using FastAPI and Facenet",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/media", StaticFiles(directory=settings.MEDIA_DIR), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(face_router, prefix="/face", tags=["face"])


@app.get("/")
def health_check():
    return {"message": "Server is running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=settings.DEBUG)
