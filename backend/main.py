from fastapi import FastAPI
from backend.routes import upload, run

app = FastAPI()

app.include_router(upload.router)
app.include_router(run.router)

@app.get("/")
def root():
    return {"message": "InsightForge API running"}
