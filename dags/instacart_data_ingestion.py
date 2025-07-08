from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import io
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'instacart_feature_engineering',
    default_args=default_args,
    description='Instacart Market Basket Analysis Feature Engineering Pipeline',
    schedule='@daily',
    catchup=False,
    tags=['instacart', 'feature-engineering', 'ml'],
)

# MinIO/S3 connection configuration
MINIO_CONN_ID = 'minio_default'
BUCKET_NAME = 'instacart-data'
RAW_DATA_PREFIX = 'raw/'
PROCESSED_DATA_PREFIX = 'processed/'

def read_csv_from_minio(file_path, **context):
    """Read CSV file from MinIO"""
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    
    try:
        file_content = s3_hook.read_key(
            key=file_path,
            bucket_name=BUCKET_NAME
        )
        df = pd.read_csv(io.StringIO(file_content))
        logging.info(f"Successfully read {file_path} with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        raise

def write_csv_to_minio(df, file_path, **context):
    """Write DataFrame to MinIO as CSV"""
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_hook.load_string(
            string_data=csv_buffer.getvalue(),
            key=file_path,
            bucket_name=BUCKET_NAME,
            replace=True
        )
        logging.info(f"Successfully wrote {file_path} with shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error writing {file_path}: {str(e)}")
        raise

def extract_raw_data(**context):
    """Extract and validate raw data from MinIO"""
    files_to_extract = [
        'aisles.csv',
        'departments.csv', 
        'orders.csv',
        'products.csv',
        'order_products__prior.csv',
        'order_products__train.csv'
    ]
    
    extracted_data = {}
    
    for file_name in files_to_extract:
        file_path = f"{RAW_DATA_PREFIX}{file_name}"
        df = read_csv_from_minio(file_path, **context)
        extracted_data[file_name.replace('.csv', '')] = df
        
        # Basic validation
        if df.empty:
            raise ValueError(f"{file_name} is empty!")
        
        logging.info(f"Extracted {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Store in XCom for downstream tasks
    context['task_instance'].xcom_push(key='raw_data', value=extracted_data)
    
    return "Raw data extraction completed"

def merge_product_data(**context):
    """Merge products with aisles and departments"""
    raw_data = context['task_instance'].xcom_pull(key='raw_data', task_ids='extract_raw_data')
    
    products = raw_data['products']
    aisles = raw_data['aisles']
    departments = raw_data['departments']
    
    # Step 1: Merge products with aisles
    products_merged = products.merge(aisles, on='aisle_id', how='left')
    logging.info(f"After aisles merge: {products_merged.shape}")
    
    # Step 2: Merge with departments
    products_merged = products_merged.merge(departments, on='department_id', how='left')
    logging.info(f"After departments merge: {products_merged.shape}")
    
    # Write merged products to MinIO
    write_csv_to_minio(products_merged, f"{PROCESSED_DATA_PREFIX}products_merged.csv", **context)
    
    context['task_instance'].xcom_push(key='products_merged', value=products_merged)
    
    return f"Products merged successfully: {products_merged.shape}"

def create_prior_combined_data(**context):
    """Create combined prior dataset"""
    raw_data = context['task_instance'].xcom_pull(key='raw_data', task_ids='extract_raw_data')
    products_merged = context['task_instance'].xcom_pull(key='products_merged', task_ids='merge_product_data')
    
    orders = raw_data['orders']
    order_products_prior = raw_data['order_products__prior']
    
    # Step 1: Merge order_products_prior with orders
    prior_combined = order_products_prior.merge(orders, on='order_id', how='left')
    logging.info(f"After orders merge: {prior_combined.shape}")
    
    # Step 2: Merge with products_merged
    prior_combined = prior_combined.merge(products_merged, on='product_id', how='left')
    logging.info(f"Final prior_combined shape: {prior_combined.shape}")
    
    # Write to MinIO
    write_csv_to_minio(prior_combined, f"{PROCESSED_DATA_PREFIX}prior_combined.csv", **context)
    
    context['task_instance'].xcom_push(key='prior_combined', value=prior_combined)
    
    return f"Prior combined dataset created: {prior_combined.shape}"

def create_train_combined_data(**context):
    """Create combined train dataset"""
    raw_data = context['task_instance'].xcom_pull(key='raw_data', task_ids='extract_raw_data')
    products_merged = context['task_instance'].xcom_pull(key='products_merged', task_ids='merge_product_data')
    
    orders = raw_data['orders']
    order_products_train = raw_data['order_products__train']
    
    # Step 1: Merge order_products_train with orders
    train_combined = order_products_train.merge(orders, on='order_id', how='left')
    logging.info(f"After orders merge: {train_combined.shape}")
    
    # Step 2: Merge with products_merged
    train_combined = train_combined.merge(products_merged, on='product_id', how='left')
    logging.info(f"Final train_combined shape: {train_combined.shape}")
    
    # Write to MinIO
    write_csv_to_minio(train_combined, f"{PROCESSED_DATA_PREFIX}train_combined.csv", **context)
    
    context['task_instance'].xcom_push(key='train_combined', value=train_combined)
    
    return f"Train combined dataset created: {train_combined.shape}"

def max_streak(reorder_list):
    """Helper function to calculate maximum streak of reorders"""
    if not reorder_list:
        return 0
    
    max_streak = 0
    current_streak = 0
    
    for reorder in reorder_list:
        if reorder == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def generate_product_features(**context):
    """Generate product-based features (7 features)"""
    prior_combined = context['task_instance'].xcom_pull(key='prior_combined', task_ids='create_prior_combined_data')
    products_merged = context['task_instance'].xcom_pull(key='products_merged', task_ids='merge_product_data')
    
    # Create empty dataframe
    product_features = pd.DataFrame(columns=['product_id'])
    product_features['product_id'] = prior_combined['product_id'].sort_values().unique()
    
    # Feature 1: product_reorder_rate
    # Fix: Don't create DataFrame with column name that conflicts with index
    reorder_stats = prior_combined.groupby(['product_id','reordered'])['reordered'].count().groupby(level=0).apply(lambda x: x / float(x.sum())).reset_index()
    reorder_stats.columns = ['product_id', 'reordered', 'reorder_rate']
    
    new_df = reorder_stats[reorder_stats['reordered']==1]
    new_df['reorder_rate'] = new_df['reorder_rate'] * new_df['reordered']
    
    # Handle products never reordered
    new_df_1 = reorder_stats[(reorder_stats['reordered']==0) & (reorder_stats['reorder_rate']==float(1.0))]
    new_df_1['reorder_rate'] = new_df_1['reorder_rate'] * new_df_1['reordered']
    new_df = pd.concat([new_df, new_df_1])
    
    new_df.drop('reordered', axis=1, inplace=True)
    new_df.sort_values(by='product_id', inplace=True)   
    new_df = new_df.reset_index(drop=True)
    
    product_features = product_features.merge(new_df, on='product_id', how='left')
    product_features['product_reorder_rate'] = product_features['reorder_rate'].fillna(0)
    product_features.drop('reorder_rate', axis=1, inplace=True)
    
    # Feature 2: avg_pos_incart
    mean_position = prior_combined.groupby('product_id')['add_to_cart_order'].mean().reset_index(name='avg_pos_incart')
    product_features = product_features.merge(mean_position, on='product_id', how='left')
    
    # Features 3-5: Generate NMF reduced features
    # Create boolean features for product categories
    temp_products = products_merged.copy()
    temp_products['organic'] = temp_products['product_name'].apply(lambda x: 'organic' in str(x).lower()).astype(int)
    temp_products['isYogurt'] = temp_products['aisle_id'].apply(lambda x: x==120).astype(int)
    temp_products['isProduce'] = temp_products['department_id'].apply(lambda x: x==4).astype(int)
    temp_products['isFrozen'] = temp_products['department_id'].apply(lambda x: x==1).astype(int)
    temp_products['isdairy'] = temp_products['department_id'].apply(lambda x: x==16).astype(int)
    temp_products['isbreakfast'] = temp_products['department_id'].apply(lambda x: x==14).astype(int)
    temp_products['issnack'] = temp_products['department_id'].apply(lambda x: x==19).astype(int)
    temp_products['isbeverage'] = temp_products['department_id'].apply(lambda x: x==7).astype(int)
    
    new_product_feat = temp_products[['organic', 'isYogurt', 'isProduce', 'isFrozen', 'isdairy', 'isbreakfast', 'issnack', 'isbeverage']]
    
    # Apply NMF
    nmf = NMF(n_components=3, random_state=42)
    W = nmf.fit_transform(new_product_feat)
    prod_data = pd.DataFrame(normalize(W))
    prod_data.columns = ['p_reduced_feat_1', 'p_reduced_feat_2', 'p_reduced_feat_3']
    prod_data['product_id'] = temp_products['product_id'].values
    
    product_features = product_features.merge(prod_data, on='product_id', how='left')
    
    # Features 6-7: Aisle and department reorder rates
    # Aisle reorder rate
    aisle_stats = prior_combined.groupby('aisle_id').agg({
        'reordered': ['count', 'sum']
    }).reset_index()
    aisle_stats.columns = ['aisle_id', 'total_orders', 'reorders']
    aisle_stats['aisle_reorder_rate'] = aisle_stats['reorders'] / aisle_stats['total_orders']
    
    # Department reorder rate  
    dept_stats = prior_combined.groupby('department_id').agg({
        'reordered': ['count', 'sum']
    }).reset_index()
    dept_stats.columns = ['department_id', 'total_orders', 'reorders']
    dept_stats['department_reorder_rate'] = dept_stats['reorders'] / dept_stats['total_orders']
    
    # Merge back to product features
    product_aisle = products_merged[['product_id', 'aisle_id', 'department_id']].drop_duplicates()
    product_features = product_features.merge(product_aisle, on='product_id', how='left')
    product_features = product_features.merge(aisle_stats[['aisle_id', 'aisle_reorder_rate']], on='aisle_id', how='left')
    product_features = product_features.merge(dept_stats[['department_id', 'department_reorder_rate']], on='department_id', how='left')
    
    # Clean up - drop the intermediate columns we don't need in final output
    product_features = product_features.drop(['aisle_id', 'department_id'], axis=1)
    
    # Write to MinIO
    write_csv_to_minio(product_features, f"{PROCESSED_DATA_PREFIX}product_features.csv", **context)
    
    context['task_instance'].xcom_push(key='product_features', value=product_features)
    
    return f"Product features generated: {product_features.shape} - 7 features"

def generate_user_features(**context):
    """Generate user-based features (6 features)"""
    prior_combined = context['task_instance'].xcom_pull(key='prior_combined', task_ids='create_prior_combined_data')
    
    # Create empty dataframe
    user_features = pd.DataFrame(columns=['user_id'])
    user_features['user_id'] = prior_combined['user_id'].sort_values().unique()
    
    # Feature 1: user_reorder_rate
    user_reorder_rate = prior_combined.groupby(["user_id","reordered"])['reordered'].count().groupby(level=0).apply(lambda x: x / float(x.sum())).reset_index(name='reorder_rate')
    user_reorder_rate = user_reorder_rate.pivot(index='user_id', columns='reordered', values=['reorder_rate']) 
    user_reorder_rate = pd.DataFrame(user_reorder_rate.to_records())
    user_reorder_rate.columns = ['user_id','reorder_0', 'reorder_1']
    user_reorder_rate.fillna(0, inplace=True)
    user_features = user_features.merge(user_reorder_rate[['user_id', 'reorder_1']], on='user_id', how='left')
    user_features.rename(columns={'reorder_1': 'user_reorder_rate'}, inplace=True)
    
    # Feature 2: user_unique_products
    unique_products = prior_combined.groupby("user_id")['product_name'].nunique().reset_index(name='user_unique_products')
    user_features = user_features.merge(unique_products, on='user_id', how='left')
    
    # Feature 3: user_total_products
    total_products = prior_combined.groupby("user_id")['product_name'].size().reset_index(name='user_total_products')
    user_features = user_features.merge(total_products, on='user_id', how='left')
    
    # Feature 4: user_avg_cart_size
    cart_size = prior_combined.groupby(["user_id","order_id"])['add_to_cart_order'].count().reset_index(name='cart_size')\
                                                                .groupby('user_id')['cart_size'].mean().reset_index(name='user_avg_cart_size')
    user_features = user_features.merge(cart_size, on='user_id', how='left')
    
    # Feature 5: user_avg_days_between_orders
    days_between = prior_combined.groupby(["user_id","order_id"])['days_since_prior_order'].max().reset_index(name='days_between')\
                                                                .groupby('user_id')['days_between'].mean().reset_index(name='user_avg_days_between_orders')
    user_features = user_features.merge(days_between, on='user_id', how='left')
    
    # Feature 6: user_reordered_products_ratio
    reordered_unique = prior_combined[prior_combined['reordered']==1].groupby("user_id")['product_name'].nunique().reset_index(name='user_reordered_products')
    user_features = user_features.merge(reordered_unique, on='user_id', how='left')
    user_features['user_reordered_products'].fillna(0, inplace=True)
    user_features['user_reordered_products_ratio'] = user_features['user_reordered_products'] / user_features['user_unique_products']
    user_features.drop('user_reordered_products', axis=1, inplace=True)
    
    # Write to MinIO
    write_csv_to_minio(user_features, f"{PROCESSED_DATA_PREFIX}user_features.csv", **context)
    
    context['task_instance'].xcom_push(key='user_features', value=user_features)
    
    return f"User features generated: {user_features.shape} - 6 features"

def generate_user_product_features(**context):
    """Generate user-product interaction features (5 features)"""
    prior_combined = context['task_instance'].xcom_pull(key='prior_combined', task_ids='create_prior_combined_data')
    
    # Create empty dataframe with unique user-product pairs
    u_p = prior_combined.groupby(["user_id","product_id"]).size().reset_index(name='interaction_count')
    user_product_features = u_p[['user_id', 'product_id']].copy()
    
    # Feature 1: u_p_order_rate
    user_order_counts = prior_combined.groupby("user_id").size()
    product_order_counts = prior_combined.groupby(["user_id","product_id"]).size()
    order_rate = (product_order_counts / user_order_counts).reset_index(name='u_p_order_rate')
    user_product_features = user_product_features.merge(order_rate, on=['user_id', 'product_id'], how='left')
    
    # Feature 2: u_p_reorder_rate
    total_orders_per_up = prior_combined.groupby(["user_id","product_id"]).size()
    reorders_per_up = prior_combined[prior_combined["reordered"]==1].groupby(["user_id","product_id"]).size()
    reorder_rate = (reorders_per_up / total_orders_per_up).reset_index(name='u_p_reorder_rate')
    reorder_rate.fillna(0, inplace=True)
    user_product_features = user_product_features.merge(reorder_rate, on=['user_id', 'product_id'], how='left')
    user_product_features['u_p_reorder_rate'].fillna(0, inplace=True)
    
    # Feature 3: u_p_avg_position
    avg_position = prior_combined.groupby(["user_id","product_id"])['add_to_cart_order'].mean().reset_index(name='u_p_avg_position')
    user_product_features = user_product_features.merge(avg_position, on=['user_id', 'product_id'], how='left')
    
    # Feature 4: u_p_orders_since_last
    last_order_per_up = prior_combined.groupby(["user_id","product_id"])['order_number'].max().reset_index(name='last_order')
    last_order_per_user = prior_combined.groupby("user_id")['order_number'].max().reset_index(name='user_last_order')
    
    orders_since = last_order_per_up.merge(last_order_per_user, on='user_id', how='left')
    orders_since['u_p_orders_since_last'] = orders_since['user_last_order'] - orders_since['last_order']
    user_product_features = user_product_features.merge(orders_since[['user_id', 'product_id', 'u_p_orders_since_last']], 
                                                        on=['user_id', 'product_id'], how='left')
    
    # Feature 5: max_streak
    streak_data = prior_combined.groupby(["user_id","product_id"])['reordered'].apply(list).reset_index(name='reorder_list')
    streak_data['max_streak'] = streak_data['reorder_list'].apply(max_streak)
    user_product_features = user_product_features.merge(streak_data[['user_id', 'product_id', 'max_streak']], 
                                                        on=['user_id', 'product_id'], how='left')
    
    # Write to MinIO
    write_csv_to_minio(user_product_features, f"{PROCESSED_DATA_PREFIX}user_product_features.csv", **context)
    
    context['task_instance'].xcom_push(key='user_product_features', value=user_product_features)
    
    return f"User-Product features generated: {user_product_features.shape} - 5 features"

def create_final_feature_set(**context):
    """Combine all features into final dataset"""
    product_features = context['task_instance'].xcom_pull(key='product_features', task_ids='generate_product_features')
    user_features = context['task_instance'].xcom_pull(key='user_features', task_ids='generate_user_features')
    user_product_features = context['task_instance'].xcom_pull(key='user_product_features', task_ids='generate_user_product_features')
    
    # Start with user-product features as base
    final_features = user_product_features.copy()
    
    # Merge user features
    final_features = final_features.merge(user_features, on='user_id', how='left')
    
    # Merge product features
    final_features = final_features.merge(product_features, on='product_id', how='left')
    
    # Write final feature set to MinIO
    write_csv_to_minio(final_features, f"{PROCESSED_DATA_PREFIX}final_features.csv", **context)
    
    # Generate feature summary
    feature_summary = {
        'total_features': final_features.shape[1] - 2,  # Exclude user_id and product_id
        'total_rows': final_features.shape[0],
        'product_features': 7,
        'user_features': 6, 
        'user_product_features': 5,
        'columns': list(final_features.columns)
    }
    
    context['task_instance'].xcom_push(key='feature_summary', value=feature_summary)
    
    return f"Final feature set created: {final_features.shape} with {feature_summary['total_features']} features"

# Define tasks
start_task = EmptyOperator(
    task_id='start_pipeline',
    dag=dag,
)

extract_task = PythonOperator(
    task_id='extract_raw_data',
    python_callable=extract_raw_data,
    dag=dag,
)

merge_products_task = PythonOperator(
    task_id='merge_product_data',
    python_callable=merge_product_data,
    dag=dag,
)

create_prior_task = PythonOperator(
    task_id='create_prior_combined_data',
    python_callable=create_prior_combined_data,
    dag=dag,
)

create_train_task = PythonOperator(
    task_id='create_train_combined_data',
    python_callable=create_train_combined_data,
    dag=dag,
)

product_features_task = PythonOperator(
    task_id='generate_product_features',
    python_callable=generate_product_features,
    dag=dag,
)

user_features_task = PythonOperator(
    task_id='generate_user_features',
    python_callable=generate_user_features,
    dag=dag,
)

user_product_features_task = PythonOperator(
    task_id='generate_user_product_features',
    python_callable=generate_user_product_features,
    dag=dag,
)

final_features_task = PythonOperator(
    task_id='create_final_feature_set',
    python_callable=create_final_feature_set,
    dag=dag,
)

end_task = EmptyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Define task dependencies
start_task >> extract_task
extract_task >> merge_products_task
merge_products_task >> [create_prior_task, create_train_task]
create_prior_task >> [product_features_task, user_features_task, user_product_features_task]
[product_features_task, user_features_task, user_product_features_task] >> final_features_task
final_features_task >> end_task