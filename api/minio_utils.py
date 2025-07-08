# minio_utils.py
from minio import Minio
from io import BytesIO
import pickle
import os

# Initialize MinIO client
minio_client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False  # Set True if using https
)

def load_pickle_from_minio(bucket: str, object_name: str):
    try:
        response = minio_client.get_object(bucket, object_name)
        data = response.read()
        return pickle.loads(data)
    except Exception as e:
        print(f"[❌] Failed to load {object_name} from MinIO: {e}")
        raise
