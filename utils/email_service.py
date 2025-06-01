# utils/email_service.py

from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

class Settings(BaseModel):
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: str
    MAIL_FROM_NAME: str
    MAIL_PORT: int
    MAIL_SERVER: str
    MAIL_STARTTLS: bool
    MAIL_SSL_TLS: bool
    USE_CREDENTIALS: bool = True
    VALIDATE_CERTS: bool = True

settings = Settings(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_FROM_NAME=os.getenv("MAIL_FROM_NAME"),
    MAIL_PORT=int(os.getenv("MAIL_PORT")),
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS=os.getenv("MAIL_STARTTLS") == "True",
    MAIL_SSL_TLS=os.getenv("MAIL_SSL_TLS") == "True"
)

conf = ConnectionConfig(**settings.dict())

class EmailService:
    @staticmethod
    async def send_verification_email(email: str, code: str):
        message = MessageSchema(
            subject="Cerebro Email Verification",
            recipients=[email],
            body=f"Your verification code is: {code}",
            subtype="plain"
        )
        fm = FastMail(conf)
        await fm.send_message(message)
