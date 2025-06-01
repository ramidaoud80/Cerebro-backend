import random, string
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException, status
from models.user_model import User
from schemas.user_schema import (
    UserCreate, UserLogin, VerificationCodeSchema, ResetPasswordForm
)
from utils.auth_utils import Hasher, create_access_token
from utils.email_service import EmailService

class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_user(self, user_data: UserCreate):
        existing_user = await self.db.scalar(select(User).where(User.email == user_data.email))
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = Hasher.get_password_hash(user_data.password)
        verification_code = str(random.randint(100000, 999999))

        new_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            is_verified=False,
            verification_code=verification_code
        )
        self.db.add(new_user)
        await self.db.commit()

        await EmailService.send_verification_email(new_user.email, verification_code)
        return new_user

    async def authenticate_user(self, user_data: UserLogin):
        user = await self.db.scalar(select(User).where(User.email == user_data.email))
        if not user:
            return None
        if not Hasher.verify_password(user_data.password, user.hashed_password):
            return None
        if not user.is_verified:
            raise HTTPException(status_code=403, detail="Email not verified")
        return {"access_token": create_access_token(user.email), "token_type": "bearer"}

    async def verify_code(self, data: VerificationCodeSchema):
        user = await self.db.scalar(select(User).where(User.email == data.email))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.verification_code != data.code:
            raise HTTPException(status_code=400, detail="Invalid verification code")
        user.is_verified = True
        user.verification_code = None
        self.db.add(user)
        await self.db.commit()
        return True

    async def send_reset_code(self, email: str):
        result = await self.db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        reset_code = ''.join(random.choices(string.digits, k=6))
        user.verification_code = reset_code
        self.db.add(user)
        await self.db.commit()
        await EmailService.send_verification_email(user.email, reset_code)
        return True

# services/user_service.py

    async def reset_password(self, email: str, token: str, new_password: str):
        result = await self.db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user.verification_code != token:
            raise HTTPException(status_code=400, detail="Invalid token")

        user.hashed_password = Hasher.get_password_hash(new_password)
        user.verification_code = None
        self.db.add(user)
        await self.db.commit()

        return True
