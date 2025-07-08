from minio import Minio
import io

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "instacart-data"

# Create the bucket if it doesn't exist
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
    print(f"✅ Created bucket: {bucket_name}")
else:
    print(f"📦 Bucket already exists: {bucket_name}")

# Folder prefixes to simulate (S3-style)
prefixes = ["raw/", "processed/", "models/", "logs/"]

# Dummy content
content = b"placeholder"
content_length = len(content)

for prefix in prefixes:
    object_name = prefix + "placeholder.txt"
    client.put_object(
        bucket_name,
        object_name,
        data=io.BytesIO(content),  # ✅ wrap bytes in BytesIO
        length=content_length,
    )
    print(f"📁 Created folder structure: {object_name}")
