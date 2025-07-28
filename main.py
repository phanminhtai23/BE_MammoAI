from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI
from routes.route_user import router as users_router
from routes.route_admin import router as admin_router
from routes.route_email import router as email_router
from routes.route_model import router as model_router
from routes.route_prediction import router as prediction_router
# from routes.drugs import router as drugs_router
# from routes.ddi import router as ddi_router
import uvicorn
from config import HOST, PORT
from fastapi.middleware.cors import CORSMiddleware
from config import FRONTEND_URL
from services.model_ai import initialize_model_service

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://127.0.0.1",
    "https://localhost",
    "https://127.0.0.1",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
    "http://127.0.0.1:3003",
    "http://127.0.0.1:3004",
    "http://127.0.0.1:3005",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:3003",
    "http://localhost:3004",
    "http://localhost:3005",
    "http://localhost:3006",
    "https://127.0.0.1:3001",
    "https://127.0.0.1:3002",
    "https://127.0.0.1:3003",
    "https://127.0.0.1:3004",
    "https://127.0.0.1:3005",
    "https://localhost:3001",
    "https://localhost:3002",
    "https://localhost:3003",
    "https://localhost:3004",
    "https://localhost:3005",
    "https://localhost:3006",
    FRONTEND_URL,
    "*",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router, prefix="/user", tags=["User"])
app.include_router(email_router, prefix="/email", tags=["Email"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(model_router, prefix="/model", tags=["Model"])
app.include_router(prediction_router, prefix="/prediction", tags=["Prediction"])
# app.include_router(drugs_router, prefix="/drugs", tags=["Drugs"])
# app.include_router(ddi_router, prefix="/ddi", tags=["DDI"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize services khi app khởi động
    """
    print("🚀 Starting up application...")
    
    # Initialize model service
    try:
        success = await initialize_model_service()
        if success:
            print("✅ Model service initialized successfully")
        else:
            print("⚠️ Model service initialization failed, will retry when needed")
    except Exception as e:
        print(f"❌ Error initializing model service: {e}")
        print("⚠️ Model will be loaded when needed")

# @app.get("/docs", include_in_schema=False)
# async def custom_swagger_ui():
#     return get_swagger_ui_html(
#         openapi_url="/openapi.json",
#         title="Custom Swagger UI",
#         swagger_ui_version="0.1.0",  # Thay đổi version tại đây
#     )


if __name__ == "__main__":
    print(f"Server is running at: http://{HOST}:{PORT} !!!!")
    uvicorn.run("main:app", host=HOST, port=int(PORT), reload=True)
