from enum import Enum
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config.database import Base


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    role: Mapped[UserRole] = mapped_column(String, default=UserRole.USER.value)

    def __str__(self):
        return self.name


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )

    faces: Mapped[list["Face"]] = relationship("Face", back_populates="person")

    def __str__(self):
        return self.name


class Face(Base):
    __tablename__ = "faces"
    __table_args__ = (
        UniqueConstraint("person_id", "image_path", name="unique_person_image"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"))
    image_path: Mapped[str] = mapped_column(String, nullable=True)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(512), nullable=True)

    person: Mapped[Person] = relationship("Person", back_populates="faces")


class RecognitionResult(Base):
    __tablename__ = "recognition_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ran_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    media_path: Mapped[str] = mapped_column(String, nullable=False)
    recognition_results: Mapped[dict] = mapped_column(JSON, nullable=False)
