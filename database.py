from motor.motor_asyncio import AsyncIOMotorClient
from config import DATABASE_NAME, MONGO_URI
from dotenv import load_dotenv
import os

load_dotenv()

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PEM_PATH = os.path.join(BASE_DIR, "certs", "global-bundle.pem")

# MONGO_URI = (
#     f"mongodb://{os.getenv('MONGO_USERNAME')}:{os.getenv('MONGO_PASSWORD')}@"
#     f"{os.getenv('MONGO_HOST')}:27017/"
#     "?ssl=true"
#     f"&tlsCAFile={PEM_PATH}"
#     "&retryWrites=false"
# )
# print("MONGO_URI:", MONGO_URI)

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
users_collection = db["user"]
users_session_collection = db["user_session"]
verification_codes_collection = db["verification_code"]
predictions_collection = db["prediction"]  # Collection cho lưu dữ liệu dự đoán AI
models_collection = db["model"]  # Collection cho lưu thông tin AI models

print("Database connection successful !")
