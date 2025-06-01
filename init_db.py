# init_db.py
import asyncio
from database import engine, Base
from models.user_model import User

async def init_db():
    print("Creating database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database created successfully.")

if __name__ == "__main__":
    asyncio.run(init_db())
