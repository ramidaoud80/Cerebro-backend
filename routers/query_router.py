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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Global variables for BiMediX model and tokenizer
bimedix_model = None
bimedix_tokenizer = None

# Configure Gemini as fallback
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

# Whisper voice model
whisper_model = whisper.load_model("base")

def load_bimedix_model():
    """Load BiMediX2-8B model with optimizations"""
    global bimedix_model, bimedix_tokenizer
    
    if bimedix_model is None or bimedix_tokenizer is None:
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
            
            logger.info("BiMediX2-8B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading BiMediX2-8B model: {e}")
            raise Exception(f"BiMediX2 model loading error: {e}")

def generate_bimedix_response(prompt: str, max_length: int = 512) -> str:
    """Generate response using BiMediX2-8B model"""
    try:
        # Ensure model is loaded
        load_bimedix_model()
        
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
    """Generate text response using Gemini as fallback"""
    try:
        logger.info("Using Gemini fallback for text generation")
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
        raise Exception(f"Gemini fallback failed: {e}")

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
            prompt_text = "Please analyze this image and describe what you see in detail. Include any text, objects, people, actions, or notable features."

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
    try:
        # Since BiMediX2-8B might not support direct image input,
        # we'll create a medical analysis prompt
        if question:
            prompt = f"Medical Question: {question}\nPlease provide a medical analysis in 3-5 lines."
        else:
            prompt = "Please provide a general medical analysis of the provided information in 3-5 lines."
        
        return generate_bimedix_response(prompt)
        
    except Exception as e:
        logger.error(f"Error processing image with BiMediX2: {e}")
        raise Exception(f"BiMediX2 image processing error: {e}")

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
        
        prompt = f"{request.content} Please respond in 3-5 lines only."
        
        # Try BiMediX2 first
        try:
            logger.info("Attempting BiMediX2 text generation")
            response = generate_bimedix_response(prompt)
            logger.info("BiMediX2 text generation successful")
            return {"response": response, "model_used": "BiMediX2-8B"}
        except Exception as bimedix_error:
            logger.warning(f"BiMediX2 failed, using Gemini fallback: {bimedix_error}")
            # Fallback to Gemini
            response = generate_gemini_text_response(f"{request.content} Please respond in 3-5 lines only.")
            return {"response": response, "model_used": "Gemini (fallback)"}
            
    except Exception as e:
        logger.error(f"Both models failed for text query: {e}")
        raise HTTPException(status_code=500, detail=f"Text query processing failed: {str(e)}")

# /image: image-only with fallback
@router.post("/image")
async def image_query(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    temp_file_path = None
    try:
        logger.info(f"Processing image query for user: {current_user}")
        logger.info(f"Image file: {file.filename}, content_type: {file.content_type}")
        
        # Validate image file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image file type")
            
        image_bytes = await file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Try BiMediX2 first
        try:
            logger.info("Attempting BiMediX2 multimodal processing")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_bytes)
                temp_file_path = tmp.name
            
            response = process_image_with_bimedix(temp_file_path, "Please respond in 3-5 lines only.")
            logger.info("BiMediX2 multimodal processing successful")
            
            return {"response": response, "model_used": "BiMediX2-8B"}
            
        except Exception as bimedix_error:
            logger.warning(f"BiMediX2 failed, using Gemini fallback: {bimedix_error}")
            # Fallback to Gemini
            response = generate_gemini_image_response(image_bytes, "Please respond in 3-5 lines only.")
            return {"response": response, "model_used": "Gemini (fallback)"}
        
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

        # Transcribe with Whisper
        try:
            transcription_result = whisper_model.transcribe(
                temp_file_path,
                language=None,  # Auto-detect language
                task='transcribe',
                verbose=False
            )
            transcription = transcription_result.get("text", "").strip()
            logger.info(f"Transcription: {transcription[:100]}...")
            
        except Exception as whisper_error:
            logger.error(f"Whisper transcription error: {whisper_error}")
            raise Exception(f"Audio transcription failed: {str(whisper_error)}")

        if not transcription:
            raise Exception("No speech detected in audio")

        # Try BiMediX2 first for response generation
        try:
            logger.info("Attempting BiMediX2 voice response generation")
            prompt = f"{transcription} Please respond in 2 lines only."
            response = generate_bimedix_response(prompt)
            logger.info("BiMediX2 voice response generation successful")
            
            return {
                "response": response,
                "model_used": "BiMediX2-8B"
            }
            
        except Exception as bimedix_error:
            logger.warning(f"BiMediX2 failed, using Gemini fallback: {bimedix_error}")
            # Fallback to Gemini
            response = generate_gemini_text_response(f"{transcription} Please respond in 2 lines only.")
            
            return {
                "response": response,
                "model_used": "Gemini (fallback)"
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
        
        # Validate image file
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image file type")
            
        image_bytes = await image.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Try BiMediX2 first
        try:
            logger.info("Attempting BiMediX2 multimodal processing")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_bytes)
                temp_file_path = tmp.name
            
            response = process_image_with_bimedix(temp_file_path, f"{text}Please respond in 3-5 lines only.")
            logger.info("BiMediX2 multimodal processing successful")
            
            return {"response": response, "model_used": "BiMediX2-8B"}
            
        except Exception as bimedix_error:
            logger.warning(f"BiMediX2 failed, using Gemini fallback: {bimedix_error}")
            # Fallback to Gemini
            response = generate_gemini_image_response(image_bytes, f"{text}Please respond in 3-5 lines only.")
            return {"response": response, "model_used": "Gemini (fallback)"}
        
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

# Initialize model on startup (optional - can be done lazily)
@router.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        logger.info("Initializing BiMediX2-8B model on startup...")
        load_bimedix_model()
    except Exception as e:
        logger.warning(f"Could not pre-load BiMediX2 model on startup: {e}")
        logger.info("BiMediX2 model will be loaded on first request, Gemini fallback available")