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
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

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
        logger.info(f"Processing text query for user: {current_user}")
        logger.info(f"Query content: {request.content[:100]}...")  # Log first 100 chars
        
        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [{"text": request.content}]
            }]
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini")
            
        return {"response": response.text.strip()}
    except Exception as e:
        logger.error(f"Gemini text query error: {e}")
        raise HTTPException(status_code=500, detail=f"Text query processing failed: {str(e)}")


# /image: image-only
@router.post("/image")
async def image_query(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        logger.info(f"Processing image query for user: {current_user}")
        logger.info(f"Image file: {file.filename}, content_type: {file.content_type}")
        
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image file type")
            
        image_bytes = await file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Validate and process image
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                logger.info(f"Image format: {img.format}, size: {img.size}")
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPEG for consistent format
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=85)
                image_bytes = img_buffer.getvalue()
        except Exception as img_error:
            logger.error(f"Image processing error: {img_error}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": "Please analyze this image and describe what you see in detail. Include any text, objects, people, actions, or notable features."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                ]
            }]
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini")
            
        logger.info("Image analysis completed successfully")
        return {"response": response.text.strip()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini image query error: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


# /voice: voice + transcription + response
@router.post("/voice")
async def voice_query(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    temp_file_path = None
    try:
        logger.info(f"Processing voice query for user: {current_user}")
        logger.info(f"Audio file: {file.filename}, content_type: {file.content_type}")
        
        # Read and validate audio file
        audio_bytes = await file.read()
        logger.info(f"Audio file size: {len(audio_bytes)} bytes")
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Create temporary file with proper extension
        file_extension = '.wav'  # Default to wav
        if file.content_type:
            if 'wav' in file.content_type:
                file_extension = '.wav'
            elif 'mp3' in file.content_type:
                file_extension = '.mp3'
            elif 'ogg' in file.content_type:
                file_extension = '.ogg'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(audio_bytes)
            temp_file_path = tmp.name

        logger.info(f"Created temporary audio file: {temp_file_path}")

        # Transcribe with Whisper
        try:
            transcription_result = whisper_model.transcribe(
                temp_file_path,
                language=None,  # Auto-detect language
                task='transcribe',
                verbose=False
            )
            transcription = transcription_result.get("text", "").strip()
            logger.info(f"Transcription: {transcription[:100]}...")  # Log first 100 chars
            
        except Exception as whisper_error:
            logger.error(f"Whisper transcription error: {whisper_error}")
            raise Exception(f"Audio transcription failed: {str(whisper_error)}")

        if not transcription:
            raise Exception("No speech detected in audio")

        # Generate response with Gemini
        try:
            response = gemini_model.generate_content(
                contents=[{
                    "role": "user",
                    "parts": [{"text": f"Please respond to this transcribed speech: {transcription}"}]
                }]
            )
            
            if not response or not response.text:
                raise Exception("Empty response from Gemini")
                
            ai_response = response.text.strip()
            logger.info("Voice processing completed successfully")
            
        except Exception as gemini_error:
            logger.error(f"Gemini response error: {gemini_error}")
            raise Exception(f"AI response generation failed: {str(gemini_error)}")

        return {
            "transcription": transcription,
            "response": ai_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")


# /multimodal: image + text (question)
@router.post("/multimodal")
async def multimodal_query(text: str = Form(...), image: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        logger.info(f"Processing multimodal query for user: {current_user}")
        logger.info(f"Text: {text[:100]}...")  # Log first 100 chars
        logger.info(f"Image file: {image.filename}, content_type: {image.content_type}")
        
        # Validate image file
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image file type")
            
        image_bytes = await image.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Validate and process image
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                logger.info(f"Image format: {img.format}, size: {img.size}")
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPEG for consistent format
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=85)
                image_bytes = img_buffer.getvalue()
        except Exception as img_error:
            logger.error(f"Image processing error: {img_error}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                ]
            }]
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini")
            
        logger.info("Multimodal processing completed successfully")
        return {"response": response.text.strip()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini multimodal query error: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal processing failed: {str(e)}")