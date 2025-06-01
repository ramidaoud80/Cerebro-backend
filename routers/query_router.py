from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from core.config import SECRET_KEY, ALGORITHM, GEMINI_API_KEY
from PIL import Image
import tempfile
import io
import whisper
import base64
import google.generativeai as genai

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-05-20")

# Whisper voice model
whisper_model = whisper.load_model("base")

# Token verification
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Request schema
class QueryRequest(BaseModel):
    input_type: str
    content: str


# /query: text-only
@router.post("/query")
async def query_model(request: QueryRequest, current_user: str = Depends(get_current_user)):
    try:
        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [{"text": (request.content +" in 3 -5 lines")}]
            }]
        )
        return {"response": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini text query error: {e}")


# /image: image-only
@router.post("/image")
async def image_query(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": "Please analyze this image and describe any findings in 3 -5 lines."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                ]
            }]
        )
        return {"response": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini image query error: {e}")


# /voice: voice + transcription + response
@router.post("/voice")
async def voice_query(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        transcription = whisper_model.transcribe(tmp_path)["text"]

        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [{"text": transcription}]
            }]
        )

        return {
            "transcription": transcription,
            "response": response.text.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini voice query error: {e}")


# /multimodal: image + text (question)
@router.post("/multimodal")
async def multimodal_query(text: str = Form(...), image: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": (text+"in 3 -5 lines")},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                ]
            }]
        )
        return {"response": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini multimodal query error: {e}")
