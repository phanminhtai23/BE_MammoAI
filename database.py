from motor.motor_asyncio import AsyncIOMotorClient
from config import DATABASE_NAME, DEVELOPMENT_MODE, MONGO_URI_LOCAL, MONGO_URI_PROD

MONGO_URI = ""
if DEVELOPMENT_MODE == "true":
    MONGO_URI = MONGO_URI_LOCAL
else:
    MONGO_URI = MONGO_URI_PROD

# print("MONGO_URI:", MONGO_URI)

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
users_collection = db["user"]
users_session_collection = db["user_session"]
verification_codes_collection = db["verification_code"]
predictions_collection = db["prediction"]  # Collection cho lưu dữ liệu dự đoán AI
models_collection = db["model"]  # Collection cho lưu thông tin AI models

print("Database connection successful !")
