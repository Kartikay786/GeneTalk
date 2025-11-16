from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict

app = FastAPI(title="Dog Emotion Classifier API")

# Enable CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(predict.router)

@app.get("/")
def home():
    return {"message": "🐶 Dog Emotion Classifier API is running!"}
