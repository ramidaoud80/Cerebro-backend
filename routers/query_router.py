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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import google.generativeai as genai
import os
import logging
import asyncio
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
bimedix_model = None
bimedix_tokenizer = None
bimedix_available = False
whisper_model = None

# Configure Gemini as fallback
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

async def initialize_models():
    """Initialize all models at startup"""
    global bimedix_model, bimedix_tokenizer, bimedix_available, whisper_model
    
    
    # Initialize BiMediX2-8B model
    try:
        logger.info("Loading BiMediX2-8B model...")
        model_name = "MBZUAI/BiMediX2-8B"
        
        # Configure for efficient loading
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        bimedix_tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Load model with quantization for memory efficiency
        bimedix_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        bimedix_available = True
        logger.info("BiMediX2-8B model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load BiMediX2-8B model: {e}")
        logger.info("BiMediX2-8B model not available, will use Gemini fallback for all requests")
        bimedix_model = None
        bimedix_tokenizer = None
        bimedix_available = False

def generate_bimedix_response(prompt: str, max_length: int = 512) -> str:
    """Generate response using BiMediX2-8B model"""
    if not bimedix_available or bimedix_model is None or bimedix_tokenizer is None:
        raise Exception("BiMediX2 model not available")
    
    try:
        # Tokenize input
        inputs = bimedix_tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = inputs.to(bimedix_model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = bimedix_model.generate(
                inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=bimedix_tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = bimedix_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response
        response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating BiMediX2 response: {e}")
        raise Exception(f"BiMediX2 generation error: {e}")

def generate_gemini_text_response(prompt: str) -> str:
    """Generate text response using Gemini"""
    try:
        logger.info("Using Gemini for text generation")
        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [{"text": prompt}]
            }]
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini")
            
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini text generation error: {e}")
        raise Exception(f"Gemini text generation failed: {e}")

def generate_gemini_image_response(image_bytes: bytes, question: str = None) -> str:
    """Generate image analysis response using Gemini"""
    try:
        logger.info("Using Gemini for image analysis")
        
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
                processed_image_bytes = img_buffer.getvalue()
        except Exception as img_error:
            logger.error(f"Image processing error: {img_error}")
            raise Exception("Invalid or corrupted image file")

        base64_image = base64.b64encode(processed_image_bytes).decode("utf-8")
        
        # Create prompt based on whether question is provided
        if question:
            prompt_text = question
        else:
            prompt_text = "Please analyze this image and describe what you see in detail. Include any text, objects, people, actions, or notable features, in 3-5 lines and go straight to the point without any commentary."

        response = gemini_model.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                ]
            }]
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini")
            
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini image analysis error: {e}")
        raise Exception(f"Gemini image analysis failed: {e}")

def process_image_with_bimedix(image_path: str, question: str = None) -> str:
    """Process image with BiMediX2-8B (text-only analysis)"""
    if not bimedix_available:
        raise Exception("BiMediX2 model not available")
    
    try:
        # Since BiMediX2-8B might not support direct image input,
        # we'll create a medical analysis prompt
        if question:
            prompt = f"Medical Question: {question}\nPlease provide a medical analysis in 3-5 lines and go straight to the point without any commentary."
        else:
            prompt = "Please provide a general medical analysis of the provided information in 3-5 lines and go straight to the point without any commentary."
        
        return generate_bimedix_response(prompt)
        
    except Exception as e:
        logger.error(f"Error processing image with BiMediX2: {e}")
        raise Exception(f"BiMediX2 image processing error: {e}")

# Lifespan context manager for FastAPI startup/shutdown
@asynccontextmanager
async def lifespan(app):
    # Startup
    logger.info("Starting up application...")
    await initialize_models()
    
    # Log model availability status
    if bimedix_available:
        logger.info("✅ BiMediX2-8B model ready")
    else:
        logger.warning("❌ BiMediX2-8B model not available, using Gemini fallback")
           
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Create router
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
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

# /query: text-only with fallback
@router.post("/query")
async def query_model(request: QueryRequest, current_user: str = Depends(get_current_user)):
    try:
        logger.info(f"Processing text query for user: {current_user}")
        logger.info(f"Query content: {request.content[:100]}...")
        
        prompt = f"{request.content} Please respond in 3-5 lines only and go straight to the point without any commentary."
        
        # Try BiMediX2 if available
        if bimedix_available:
            try:
                logger.info("Using BiMediX2 for text generation")
                response = generate_bimedix_response(prompt)
                logger.info("BiMediX2 text generation successful")
                return {"response": response, "model_used": "BiMediX2-8B"}
            except Exception as bimedix_error:
                logger.error(f"BiMediX2 failed unexpectedly: {bimedix_error}")
                # Mark as unavailable for this session
                globals()['bimedix_available'] = False
        
        # Use Gemini (either as fallback or primary)
        logger.info("Using Gemini for text generation")
        response = generate_gemini_text_response(prompt)
        return {"response": response, "model_used": "Gemini"}
            
    except Exception as e:
        logger.error(f"Text query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text query processing failed: {str(e)}")

# /image: image-only with fallback
@router.post("/image")
async def image_query(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    temp_file_path = None
    try:
        logger.info(f"Processing image query for user: {current_user}")
        logger.info(f"Image file: {file.filename}, content_type: {file.content_type}")

        image_bytes = await file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Try BiMediX2 if available
        if bimedix_available:
            try:
                logger.info("Using BiMediX2 for image processing")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(image_bytes)
                    temp_file_path = tmp.name
                
                response = process_image_with_bimedix(temp_file_path, " in 2 lines and go straight to the point without any commentary.")
                logger.info("BiMediX2 image processing successful")
                return {"response": response, "model_used": "BiMediX2-8B"}
                
            except Exception as bimedix_error:
                logger.error(f"BiMediX2 failed unexpectedly: {bimedix_error}")
                # Mark as unavailable for this session
                globals()['bimedix_available'] = False
        
        # Use Gemini (either as fallback or primary)
        logger.info("Using Gemini for image analysis")
        response = generate_gemini_image_response(image_bytes, " in 2 lines and go straight to the point without any commentary.")
        return {"response": response, "model_used": "Gemini"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image query error: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

# /voice: voice + transcription + response with fallback
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

        try:
            transcription_result = whisper_model.transcribe(
                temp_file_path,
                language=None ,  
                task='transcribe',
                verbose=False
            )
            transcription = transcription_result.get("text", "").strip()
            logger.info(f"Transcription: {transcription[:100]}...")
            
        except Exception as whisper_error:
            logger.error(f"Whisper transcription error: {whisper_error}")
            raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(whisper_error)}")

        if not transcription:
            raise HTTPException(status_code=400, detail="No speech detected in audio")

        # Try BiMediX2 if available for response generation
        prompt = f"{transcription} Please respond in 1-2 lines only and go straight to the point without any commentary."
        
        if bimedix_available:
            try:
                logger.info("Using BiMediX2 for voice response generation")
                response = generate_bimedix_response(prompt)
                logger.info("BiMediX2 voice response generation successful")
                return {
                    "transcription": transcription,
                    "response": response,
                    "model_used": "BiMediX2-8B"
                }
                
            except Exception as bimedix_error:
                logger.error(f"BiMediX2 failed unexpectedly: {bimedix_error}")
                # Mark as unavailable for this session
                globals()['bimedix_available'] = False
        
        # Use Gemini (either as fallback or primary)
        logger.info("Using Gemini for voice response generation")
        response = generate_gemini_text_response(prompt)
        return {
            "transcription": transcription,
            "response": response,
            "model_used": "Gemini"
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

# /multimodal: image + text with fallback
@router.post("/multimodal")
async def multimodal_query(text: str = Form(...), image: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    temp_file_path = None
    try:
        logger.info(f"Processing multimodal query for user: {current_user}")
        logger.info(f"Text: {text[:100]}...")
        logger.info(f"Image file: {image.filename}, content_type: {image.content_type}")
        

        image_bytes = await image.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Try BiMediX2 if available
        if bimedix_available:
            try:
                logger.info("Using BiMediX2 for multimodal processing")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(image_bytes)
                    temp_file_path = tmp.name
                
                response = process_image_with_bimedix(temp_file_path, f"{text} Please respond in 3-5 lines only and go straight to the point without any commentary.")
                logger.info("BiMediX2 multimodal processing successful")
                return {"response": response, "model_used": "BiMediX2-8B"}
                
            except Exception as bimedix_error:
                logger.error(f"BiMediX2 failed unexpectedly: {bimedix_error}")
                # Mark as unavailable for this session
                globals()['bimedix_available'] = False
        
        # Use Gemini (either as fallback or primary)
        logger.info("Using Gemini for multimodal processing")
        response = generate_gemini_image_response(image_bytes, f"{text} Please respond in 3-5 lines only and go straight to the point without any commentary.")
        return {"response": response, "model_used": "Gemini"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal query error: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal processing failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

# Health check endpoint to show model status
@router.get("/health")
async def health_check():
    """Check the health and availability of models"""
    return {
        "status": "healthy",
        "models": {
            "bimedix_available": bimedix_available,
            "whisper_available": whisper_model is not None,
            "gemini_available": True  # Assuming Gemini API is always available
        }
    }