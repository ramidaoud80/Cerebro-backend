from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal
from schemas.user_schema import (
    UserCreate, UserLogin, TokenResponse,
    VerificationCodeSchema, ResetPasswordRequest, ResetPasswordForm
)
from services.user_service import UserService

router = APIRouter()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.post("/register", response_model=dict)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    service = UserService(db)
    user = await service.create_user(user_data)
    return {"message": "User registered successfully"}

@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    service = UserService(db)
    token = await service.authenticate_user(user_data)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return token

@router.post("/verify", response_model=dict)
async def verify_email(data: VerificationCodeSchema, db: AsyncSession = Depends(get_db)):
    service = UserService(db)
    success = await service.verify_code(data)
    if not success:
        raise HTTPException(status_code=400, detail="Verification failed")
    return {"message": "Email verified"}

@router.post("/request-reset", response_model=dict)
async def request_password_reset(data: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    service = UserService(db)
    success = await service.send_reset_code(data.email)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to send reset code")
    return {"message": "Reset code sent"}

@router.post("/reset-password")
async def reset_password(form_data: ResetPasswordForm, db: AsyncSession = Depends(get_db)):
    service = UserService(db)
    success = await service.reset_password(form_data.email, form_data.token, form_data.new_password)
    if not success:
        raise HTTPException(status_code=400, detail="Reset failed")
    return {"message": "Password reset successful"}
