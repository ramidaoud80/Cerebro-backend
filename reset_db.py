# reset_db.py
import asyncio
from database import engine
from models.user_model import Base  # or wherever your Base is defined

async def reset():
    print("Resetting database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database reset complete.")

if __name__ == "__main__":
    asyncio.run(reset())
