from fastapi import File, UploadFile
from pydantic import BaseModel, Field

from app.models import UserRole


class RegisterForm(BaseModel):
    email: str
    name: str
    password: str
    role: UserRole = UserRole.USER


class LoginForm(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: UserRole

    class Config:
        from_attributes = True


class PersonForm(BaseModel):
    name: str
    user_id: int | None = None


class PersonResponse(BaseModel):
    id: int
    name: str
    user_id: int | None = None

    class Config:
        from_attributes = True


class FaceForm:
    def __init__(
        self,
        file: UploadFile = File(...),
    ):
        self.file = file


class FaceResponse(BaseModel):
    id: int
    person_id: int
    image_path: str

    class Config:
        from_attributes = True


class RecognitionResponse(BaseModel):
    person_id: int | None = None
    name: str | None = None
    confidence: float | None = None

    class Config:
        from_attributes = True


class RecognitionMediaResponse(BaseModel):
    results: list[RecognitionResponse]
    media_url: str
