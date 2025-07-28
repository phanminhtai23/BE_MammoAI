from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGO_URI, DATABASE_NAME

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
users_collection = db["user"]
users_session_collection = db["user_session"]
verification_codes_collection = db["verification_code"]
predictions_collection = db["prediction"]  # Collection cho lưu dữ liệu dự đoán AI
models_collection = db["model"]  # Collection cho lưu thông tin AI models

print("Database connection successful !")