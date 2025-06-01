from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    is_verified: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ResetPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordForm(BaseModel):
    email: EmailStr
    new_password: str
    token: str
class VerificationCodeSchema(BaseModel):
    email: EmailStr
    code: str