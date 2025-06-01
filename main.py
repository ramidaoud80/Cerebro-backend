from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth_router, query_router
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(query_router.router)

@app.get("/")
def root():
    return {"message": "Cerebro backend is running with BiMediX2 powering all endpoints"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
