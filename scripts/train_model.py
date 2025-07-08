#!/usr/bin/env python3
"""
Model Training Script for E-commerce Reorder Prediction
Converts notebook code into a standalone script for training CatBoost model
"""

from minio import Minio
from io import BytesIO

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost
import warnings
warnings.filterwarnings('ignore')

# Configuration
SEED_VALUE = 42
DATA_PATH = '/Users/dharmikbhagat/ecommerce-recommender/data/processed'
MODEL_OUTPUT_PATH = '/Users/dharmikbhagat/ecommerce-recommender/notebooks/models'
MLFLOW_TRACKING_URI = "file:///Users/dharmikbhagat/ecommerce-recommender/notebooks/mlruns"  # Adjust as needed

# MinIO Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "instacart-data"
MINIO_SECURE = False

def setup_mlflow():
    """Setup MLflow tracking"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("ecommerce_reorder_prediction")
        print("✓ MLflow setup completed")
    except Exception as e:
        print(f"Warning: MLflow setup failed: {e}")
        print("Continuing without MLflow tracking...")

def setup_minio():
    """Setup MinIO client and create bucket if it doesn't exist"""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        
        # Create bucket if it doesn't exist
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            print(f"✓ Created MinIO bucket: {MINIO_BUCKET}")
        else:
            print(f"✓ MinIO bucket exists: {MINIO_BUCKET}")
            
        return client
    except Exception as e:
        print(f"Warning: MinIO setup failed: {e}")
        return None

def upload_to_minio(client, file_path, object_name=None, prefix="v2"):
    """Upload file to MinIO bucket"""
    if client is None:
        print(f"Skipping MinIO upload for {file_path} - client not available")
        return False
        
    try:
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        # Add prefix to object name
        if prefix:
            object_name = f"pickle/{prefix}/{object_name}"
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
            
        with open(file_path, "rb") as file_data:
            file_size = os.path.getsize(file_path)
            client.put_object(
                MINIO_BUCKET,
                object_name,
                file_data,
                length=file_size
            )
            print(f"✓ Uploaded {object_name} to MinIO bucket '{MINIO_BUCKET}' ({file_size} bytes)")
        return True
    except Exception as e:
        print(f"Error uploading {file_path} to MinIO: {e}")
        return False

def load_data_local():
    """Load data files from local directory"""
    
    print("Loading data files from local directory...")
    
    try:
        # Load processed feature files
        day_reorder_rate = pd.read_csv(os.path.join(DATA_PATH, 'day_reorder_rate.csv'))
        hour_reorder_rate = pd.read_csv(os.path.join(DATA_PATH, 'hour_reorder_rate.csv'))
        p_days_since_prior_order_reorder_rate = pd.read_csv(os.path.join(DATA_PATH, 'p_days_since_prior_order_reorder_rate.csv'))
        u_days_since_prior_order_reorder_rate = pd.read_csv(os.path.join(DATA_PATH, 'u_days_since_prior_order_reorder_rate.csv'))
        days_since_prior_reorder_rate = pd.read_csv(os.path.join(DATA_PATH, 'days_since_prior_reorder_rate.csv'))
        prior_combined = pd.read_csv(os.path.join(DATA_PATH, 'prior_combined.csv'))
        train_combined = pd.read_csv(os.path.join(DATA_PATH, 'train_combined.csv'))
        user_product_features = pd.read_csv(os.path.join(DATA_PATH, 'user_product_features.csv'))
        
        print(f"✓ All files loaded successfully")
        print(f"✓ User product features: {user_product_features.shape}")
        print(f"✓ Train combined: {train_combined.shape}")
        
        return {
            'day_reorder_rate': day_reorder_rate,
            'hour_reorder_rate': hour_reorder_rate,
            'p_days_since_prior_order_reorder_rate': p_days_since_prior_order_reorder_rate,
            'u_days_since_prior_order_reorder_rate': u_days_since_prior_order_reorder_rate,
            'days_since_prior_reorder_rate': days_since_prior_reorder_rate,
            'prior_combined': prior_combined,
            'train_combined': train_combined,
            'user_product_features': user_product_features
        }
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return None

def create_merged_dataset(data_files):
    """Merge all features to create the final training dataset"""
    
    print("\n" + "="*50)
    print("CREATING MERGED DATASET")
    print("="*50)
    
    # Start with user-product features as base
    merged_df = data_files['user_product_features'].copy()
    print(f"Starting with user-product features: {merged_df.shape}")
    
    # Get train order data
    train_order_data = data_files['train_combined'].copy()
    print(f"Train order data shape: {train_order_data.shape}")
    
    # Process train orders
    upd_train_orders = train_order_data[['user_id','order_id','product_id','reordered']].copy()
    
    # Get last orders for each user
    last_orders = upd_train_orders.groupby(['user_id'])['order_id'].max().reset_index()
    last_orders.rename(columns={'order_id': 'new_order_id'}, inplace=True)
    
    # Get order details
    order_details = train_order_data[['order_id','order_dow','order_hour_of_day','days_since_prior_order']].drop_duplicates()
    
    # Start merging process
    print("\nMerging datasets...")
    
    # Merge with train orders
    train_orders_merged_df = pd.merge(merged_df, upd_train_orders, 
                                    how='left', on=['user_id','product_id'])
    print(f"After merging with train orders: {train_orders_merged_df.shape}")
    
    # Merge with last orders
    train_orders_merged_df = pd.merge(train_orders_merged_df, last_orders, on='user_id')
    
    # Clean up order_id columns
    train_orders_merged_df.drop("order_id", axis=1, inplace=True)
    train_orders_merged_df.rename(columns={'new_order_id':'order_id'}, inplace=True)
    
    # Merge with order details
    train_orders_merged_df = pd.merge(train_orders_merged_df, order_details, on='order_id')
    print(f"After merging with order details: {train_orders_merged_df.shape}")
    
    # Fill missing reordered values
    train_orders_merged_df['reordered'] = train_orders_merged_df['reordered'].fillna(0.0)
    
    # Merge with time-based features
    print("\nMerging with time-based features...")
    
    # Hour reorder rate
    train_orders_merged_df = pd.merge(train_orders_merged_df, 
                                    data_files['hour_reorder_rate'], 
                                    on=['product_id','order_hour_of_day'], 
                                    how='left')
    train_orders_merged_df['hour_reorder_rate'] = train_orders_merged_df['hour_reorder_rate'].fillna(0.0)
    
    # Day reorder rate
    train_orders_merged_df = pd.merge(train_orders_merged_df, 
                                    data_files['day_reorder_rate'], 
                                    on=['product_id','order_dow'], 
                                    how='left')
    train_orders_merged_df['day_reorder_rate'] = train_orders_merged_df['day_reorder_rate'].fillna(0.0)
    
    # Product days since prior order reorder rate
    train_orders_merged_df = pd.merge(train_orders_merged_df, 
                                    data_files['p_days_since_prior_order_reorder_rate'], 
                                    on=['product_id','days_since_prior_order'], 
                                    how='left')
    train_orders_merged_df['p_days_since_prior_order_reorder_rate'] = train_orders_merged_df['p_days_since_prior_order_reorder_rate'].fillna(0.0)
    
    # User days since prior order reorder rate
    train_orders_merged_df = pd.merge(train_orders_merged_df, 
                                    data_files['u_days_since_prior_order_reorder_rate'], 
                                    on=['user_id','days_since_prior_order'], 
                                    how='left')
    train_orders_merged_df['u_days_since_prior_order_reorder_rate'] = train_orders_merged_df['u_days_since_prior_order_reorder_rate'].fillna(0.0)
    
    # Days since prior reorder rate
    train_orders_merged_df = pd.merge(train_orders_merged_df, 
                                    data_files['days_since_prior_reorder_rate'], 
                                    on=["user_id","product_id",'days_since_prior_order'], 
                                    how='left')
    train_orders_merged_df['days_since_prior_reorder_rate'] = train_orders_merged_df['days_since_prior_reorder_rate'].fillna(0.0)
    
    print(f"Final merged dataset shape: {train_orders_merged_df.shape}")
    
    return train_orders_merged_df

def prepare_ml_data(df):
    """Prepare data for machine learning"""
    
    print("\n" + "="*50)
    print("PREPARING DATA FOR ML")
    print("="*50)
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()} total missing values")
    
    # Separate features and target
    # Remove non-predictive columns
    columns_to_drop = ['user_id', 'product_id', 'order_id']
    
    # Check which columns exist in the dataset
    cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    X = df.drop(cols_to_drop + ['reordered'], axis=1)
    y = df['reordered']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED_VALUE, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Train target distribution: {y_train.value_counts().to_dict()}")
    print(f"Validation target distribution: {y_val.value_counts().to_dict()}")
    
    # Normalize features for neural network and logistic regression
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    
    print("✓ Data normalization completed")
    
    return X_train, X_val, y_train, y_val, X_train_norm, X_val_norm, scaler

def train_catboost(X_train, X_test, y_train, y_test, plot_importance=True, save=True, file_name=None):
    """
    Train CatBoost model, log metrics and model with MLflow, and optionally save and plot importance.

    Returns:
        c_model: Trained CatBoost model
        predict_y: Predicted probabilities (class 1) for validation data
    """
    start_time = datetime.now()
    print("Training Started:")

    c_model = CatBoostClassifier(
        task_type="CPU",
        verbose=100,
        depth=13,
        iterations=2000,
        learning_rate=0.02,
        scale_pos_weight=1.0,
        random_seed=SEED_VALUE
    )

    c_model.fit(X_train, y_train)

    print("Training Completed")
    end_time = datetime.now()
    duration = end_time - start_time
    hours, seconds = divmod(duration.total_seconds(), 3600)
    print(f"Total Time: {int(hours)} hours {int(seconds)} seconds")

    # Predict probabilities
    predict_proba = c_model.predict_proba(X_test)
    val_logloss = log_loss(y_test, predict_proba, labels=[0, 1])
    print("The Test log loss is:", val_logloss)

    # Log metrics to MLflow if available
    try:
        mlflow.log_param("model_type", "CatBoostClassifier")
        mlflow.log_params({
            "depth": 13,
            "iterations": 2000,
            "learning_rate": 0.02,
            "scale_pos_weight": 1.0,
            "task_type": "CPU"
        })
        mlflow.log_metric("log_loss_val", val_logloss)
        mlflow.catboost.log_model(c_model, "catboost_model")
    except Exception as e:
        print(f"MLflow logging failed: {e}")

    # Save model
    if save and file_name:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pickle.dump(c_model, open(file_name, "wb"))
        print(f"Model saved to: {file_name}")

    # Plot feature importance
    if plot_importance and hasattr(X_train, 'columns'):
        feature_names = X_train.columns.to_numpy()
        f_imp = pd.DataFrame({
            'features': feature_names,
            'feature_importance': c_model.get_feature_importance()
        })
        f_imp.sort_values(by='feature_importance', ascending=False, inplace=True)

        print("Feature Importance")
        plt.figure(figsize=(10, 10))
        sns.barplot(x=f_imp['feature_importance'], y=f_imp['features'])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(MODEL_OUTPUT_PATH, 'feature_importance.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {plot_path}")
        plt.show()

    return c_model, predict_proba[:, 1]  # Return probability for class 1

def visualize_reorder_patterns(data_files):
    """Create visualizations of reorder patterns"""
    
    if data_files is None:
        return
    
    print("Creating reorder pattern visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reorder Patterns Analysis', fontsize=16, fontweight='bold')
    
    # 1. Day of week reorder rates
    if 'day_reorder_rate' in data_files:
        day_data = data_files['day_reorder_rate'].groupby('order_dow')['day_reorder_rate'].mean()
        axes[0, 0].bar(day_data.index, day_data.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Reorder Rate by Day of Week')
        axes[0, 0].set_xlabel('Day of Week (0=Sunday)')
        axes[0, 0].set_ylabel('Reorder Rate')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Hour of day reorder rates
    if 'hour_reorder_rate' in data_files:
        hour_data = data_files['hour_reorder_rate'].groupby('order_hour_of_day')['hour_reorder_rate'].mean()
        axes[0, 1].plot(hour_data.index, hour_data.values, marker='o', linewidth=2, markersize=4)
        axes[0, 1].set_title('Average Reorder Rate by Hour of Day')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Reorder Rate')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. User product features distribution
    if 'user_product_features' in data_files:
        up_features = data_files['user_product_features']
        if 'up_order_rate' in up_features.columns:
            axes[1, 0].hist(up_features['up_order_rate'], bins=50, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Distribution of User-Product Order Rates')
            axes[1, 0].set_xlabel('Order Rate')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Days since prior order pattern
    if 'days_since_prior_reorder_rate' in data_files:
        days_data = data_files['days_since_prior_reorder_rate']
        if 'days_since_prior_order' in days_data.columns:
            days_summary = days_data.groupby('days_since_prior_order')['days_since_prior_reorder_rate'].mean()
            # Limit to first 30 days for better visualization
            days_summary = days_summary.head(30)
            axes[1, 1].bar(days_summary.index, days_summary.values, alpha=0.7, color='salmon')
            axes[1, 1].set_title('Reorder Rate by Days Since Prior Order')
            axes[1, 1].set_xlabel('Days Since Prior Order')
            axes[1, 1].set_ylabel('Reorder Rate')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(MODEL_OUTPUT_PATH, 'reorder_patterns.png')
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Reorder patterns visualization saved to: {viz_path}")
    plt.show()
    
    return viz_path

def main():
    """Main function to orchestrate the entire training pipeline"""
    
    print("="*60)
    print("E-COMMERCE REORDER PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Setup
    setup_mlflow()
    minio_client = setup_minio()
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    # Initialize file paths
    model_path = None
    scaler_path = None
    feature_names_path = None
    viz_path = None
    feature_importance_path = None
    
    try:
        with mlflow.start_run(run_name="catboost_training_pipeline"):
            # Load data
            print("\n1. Loading data...")
            data_files = load_data_local()
            
            if data_files is None:
                print("Failed to load data files. Exiting...")
                sys.exit(1)
            
            # Create visualizations
            print("\n2. Creating visualizations...")
            viz_path = visualize_reorder_patterns(data_files)
            
            # Create merged dataset
            print("\n3. Creating merged dataset...")
            final_dataset = create_merged_dataset(data_files)
            print(f"\nFinal dataset created with shape: {final_dataset.shape}")
            print(f"Target distribution:\n{final_dataset['reordered'].value_counts()}")
            
            # Display dataset info
            print(f"\nDataset Info:")
            print(f"Features: {final_dataset.shape[1] - 1}")
            print(f"Samples: {final_dataset.shape[0]}")
            print(f"Positive class ratio: {final_dataset['reordered'].mean():.4f}")
            
            # Show feature names
            feature_cols = [col for col in final_dataset.columns if col not in ['user_id', 'product_id', 'order_id', 'reordered']]
            print(f"\nFeature columns ({len(feature_cols)}):")
            for col in feature_cols:
                print(f"  - {col}")
            
            # Prepare ML data
            print("\n4. Preparing data for ML...")
            X_train, X_val, y_train, y_val, X_train_norm, X_val_norm, scaler = prepare_ml_data(final_dataset)
            
            # Train CatBoost model
            print("\n5. Training CatBoost model...")
            model_path = os.path.join(MODEL_OUTPUT_PATH, 'catboost_model_v2.pkl')
            catboost_model, predictions = train_catboost(
                X_train, X_val, y_train, y_val,
                plot_importance=True,
                save=True,
                file_name=model_path
            )
            
            # Save scaler
            scaler_path = os.path.join(MODEL_OUTPUT_PATH, 'scaler.pkl')
            pickle.dump(scaler, open(scaler_path, "wb"))
            print(f"Scaler saved to: {scaler_path}")
            
            # Save feature names
            feature_names_path = os.path.join(MODEL_OUTPUT_PATH, 'feature_names.pkl')
            pickle.dump(list(X_train.columns), open(feature_names_path, "wb"))
            print(f"Feature names saved to: {feature_names_path}")
            
            # Define feature importance plot path
            feature_importance_path = os.path.join(MODEL_OUTPUT_PATH, 'feature_importance.png')
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Model saved at: {model_path}")
            print(f"Scaler saved at: {scaler_path}")
            print(f"Feature names saved at: {feature_names_path}")
            
            # Upload all files to MinIO
            print("\n6. Uploading files to MinIO...")
            upload_success = []
            
            if model_path and os.path.exists(model_path):
                success = upload_to_minio(minio_client, model_path, prefix="v2")
                upload_success.append(("Model", success))
            
            if scaler_path and os.path.exists(scaler_path):
                success = upload_to_minio(minio_client, scaler_path, prefix="v2")
                upload_success.append(("Scaler", success))
            
            if feature_names_path and os.path.exists(feature_names_path):
                success = upload_to_minio(minio_client, feature_names_path, prefix="v2")
                upload_success.append(("Feature names", success))
            
            if viz_path and os.path.exists(viz_path):
                success = upload_to_minio(minio_client, viz_path, prefix="v2")
                upload_success.append(("Reorder patterns visualization", success))
            
            if feature_importance_path and os.path.exists(feature_importance_path):
                success = upload_to_minio(minio_client, feature_importance_path, prefix="v2")
                upload_success.append(("Feature importance plot", success))
            
            # Summary of uploads
            print("\n" + "="*60)
            print("MINIO UPLOAD SUMMARY")
            print("="*60)
            for item_name, success in upload_success:
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"{item_name}: {status}")
                
            total_uploads = len(upload_success)
            successful_uploads = sum(1 for _, success in upload_success if success)
            print(f"\nTotal files uploaded: {successful_uploads}/{total_uploads}")
            
            if successful_uploads == total_uploads:
                print("🎉 All files successfully uploaded to MinIO!")
            else:
                print("⚠️  Some files failed to upload to MinIO")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()