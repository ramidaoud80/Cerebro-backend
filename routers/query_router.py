from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class QueryInput(BaseModel):
    input_type: str  # "text", "voice", "image"
    content: str     # The query text or dummy content

@router.post("/query")
def handle_query(data: QueryInput):
    return {
        "message": f"Received a {data.input_type} input.",
        "content": data.content
    }
