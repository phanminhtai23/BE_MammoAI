from fastapi import FastAPI
from routes.route_user import router as users_router
from routes.route_admin import router as admin_router
from routes.route_email import router as email_router
from routes.route_model import router as model_router
from routes.route_prediction import router as prediction_router
from contextlib import asynccontextmanager

# from routes.drugs import router as drugs_router
# from routes.ddi import router as ddi_router
import uvicorn
from config import HOST, PORT
from fastapi.middleware.cors import CORSMiddleware
from config import FRONTEND_URL
from services.model_ai import initialize_model_service
from starlette.middleware.proxy_headers import ProxyHeadersMiddleware

origins = [
    FRONTEND_URL,
    "*",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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
    
    yield
    
    # Shutdown
    print("🛑 Shutting down application...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    ProxyHeadersMiddleware,
    trusted_hosts="*",
)

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

if __name__ == "__main__":
    print(f"Server is running at: http://{HOST}:{PORT} !!!!")
    uvicorn.run("main:app", host=HOST, port=int(PORT), reload=True)
