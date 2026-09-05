from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.database import get_db
from app.models import User
from app.schema import LoginForm, RegisterForm, TokenResponse, UserResponse
from app.services.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_authenticated_user,
    get_password_hash,
    get_token,
)

router = APIRouter(tags=["auth"])


@router.post("/register", response_model=UserResponse)
async def register(form: RegisterForm, db: AsyncSession = Depends(get_db)):

    result = await db.execute(select(User).where(User.email == form.email))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    new_user = User(
        email=form.email,
        name=form.name,
        password_hash=get_password_hash(form.password),
        role=form.role.value,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user


@router.post("/login", response_model=TokenResponse)
async def login(form: LoginForm, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form.email, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or password")

    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(token: str):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    new_access_token = create_access_token(data={"sub": user_id})

    return TokenResponse(access_token=new_access_token, refresh_token=token)


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    db: AsyncSession = Depends(get_db), token: str = Depends(get_token)
):
    User = await get_authenticated_user(db, token)
    return User
