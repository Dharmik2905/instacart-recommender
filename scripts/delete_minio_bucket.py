from minio import Minio
from minio.error import S3Error

# Connect to MinIO
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "instacart-data"

try:
    # Step 1: List and delete all objects
    objects_to_delete = client.list_objects(bucket_name, recursive=True)
    delete_list = [obj.object_name for obj in objects_to_delete]

    if delete_list:
        print(f"🗑️ Deleting {len(delete_list)} objects from '{bucket_name}'...")
        for obj in delete_list:
            client.remove_object(bucket_name, obj)
        print("✅ All objects deleted.")

    # Step 2: Delete bucket
    client.remove_bucket(bucket_name)
    print(f"✅ Bucket '{bucket_name}' deleted successfully.")

except S3Error as e:
    print(f"❌ Error: {e}")
