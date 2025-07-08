
import os
import json
from airflow import settings
from airflow.models import Connection

def create_minio_connection():
    """
    Create MinIO connection in Airflow
    """
    
    # MinIO connection details (Docker setup)
    conn_id = 'minio_default'
    conn_type = 'aws'
    host = 'localhost'  # Docker MinIO host
    port = 9000        # Docker MinIO port (default)
    login = 'minioadmin'        # Default MinIO access key
    password = 'minioadmin'     # Default MinIO secret key
    
    # Extra configuration as proper JSON for Docker MinIO
    extra_config = {
        "endpoint_url": "http://localhost:9000",  # Docker MinIO URL
        "aws_access_key_id": "minioadmin",        # Default MinIO access key
        "aws_secret_access_key": "minioadmin",    # Default MinIO secret key
        "region_name": "us-east-1"                # Required for some operations
    }
    
    # Convert to JSON string
    extra_json = json.dumps(extra_config)
    
    # Create connection
    new_conn = Connection(
        conn_id=conn_id,
        conn_type=conn_type,
        host=host,
        port=port,
        login=login,
        password=password,
        extra=extra_json  # This must be valid JSON string
    )
    
    # Get session
    session = settings.Session()
    
    try:
        # Check if connection already exists
        existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
        
        if existing_conn:
            print(f"Connection '{conn_id}' already exists. Updating...")
            existing_conn.conn_type = conn_type
            existing_conn.host = host
            existing_conn.port = port
            existing_conn.login = login
            existing_conn.password = password
            existing_conn.extra = extra_json
        else:
            print(f"Creating new connection '{conn_id}'...")
            session.add(new_conn)
        
        session.commit()
        print(f"MinIO connection '{conn_id}' created/updated successfully!")
        
        # Print connection details for verification
        print(f"Connection details:")
        print(f"  ID: {conn_id}")
        print(f"  Type: {conn_type}")
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print(f"  Login: {login}")
        print(f"  Extra: {extra_json}")
        
    except Exception as e:
        print(f"Error creating connection: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def test_connection():
    """
    Test the MinIO connection
    """
    try:
        from airflow.providers.amazon.aws.hooks.s3 import S3Hook
        
        print("Testing MinIO connection...")
        hook = S3Hook(aws_conn_id='minio_default')
        
        # Try to get the S3 client and list buckets
        s3_client = hook.get_conn()
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
        print(f"Successfully connected! Found {len(buckets)} buckets: {buckets}")
        
    except Exception as e:
        print(f"Connection test failed: {e}")
        print("Make sure MinIO is running and credentials are correct")
        print("If MinIO is not running, start it with: minio server /path/to/data")

if __name__ == '__main__':
    print("Setting up MinIO connection for Airflow...")
    
    # Update these values for your MinIO setup
    print("\nIMPORTANT: Update the connection details in this script:")
    print("- host: Change from 'localhost' to your MinIO server")
    print("- port: Change from 9000 to your MinIO port")  
    print("- login/password: Change from 'minioadmin' to your actual credentials")
    print("- endpoint_url: Update the URL in extra config")
    
    create_minio_connection()
    
    # Test the connection
    test_connection()