from fastapi import APIRouter, UploadFile, File
import pandas as pd
import os

router = APIRouter()

DATA_STORAGE = {}

def load_dataframe(file):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file.file)
    elif ext == ".xlsx":
        return pd.read_excel(file.file)
    elif ext == ".json":
        return pd.read_json(file.file)
    elif ext == ".parquet":
        return pd.read_parquet(file.file)
    return None

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    df = load_dataframe(file)

    if df is None:
        return {"error": "Unsupported file format"}

    DATA_STORAGE["raw_df"] = df

    return {
        "message": "File uploaded successfully",
        "columns": list(df.columns),
        "rows": len(df)
    }
