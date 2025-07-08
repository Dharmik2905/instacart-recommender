import pandas as pd
import pickle
from datetime import datetime
import mlflow
from api.minio_utils import load_pickle_from_minio
BUCKET = "instacart-data" 
PREFIX = "pickle/"
def get_recommendations(user_id):
    start_time = datetime.now()
    now = datetime.now()
    order_hour_of_day = now.hour
    order_dow = now.weekday()
    today = int(now.strftime("%d"))

    ulp = load_pickle_from_minio(BUCKET, f"{PREFIX}user_last_purchase.pkl")

    with mlflow.start_run(run_name="inference", nested=True):
        mlflow.set_tag("model", "catboost_model")
        mlflow.log_param("user_id", user_id)
        mlflow.log_param("hour", order_hour_of_day)
        mlflow.log_param("dow", order_dow)

        if user_id not in ulp['user_id'].values:
            top = load_pickle_from_minio(BUCKET, f"{PREFIX}top10_products.pkl")
            top_products = top[
                (top['order_dow'] == order_dow) &
                (top['order_hour_of_day'] == order_hour_of_day)
            ]['product_name'].values.tolist()
            mlflow.set_tag("mode", "cold_start")
            mlflow.log_metric("recommend_count", len(top_products))
            mlflow.log_metric("inference_time_sec", (datetime.now() - start_time).total_seconds())
            return {"cold_start": True, "recommendations": top_products}

        user_last_order_date = ulp[ulp['user_id'] == user_id]['date'].values[0]
        days_since_prior_order = today - int(user_last_order_date.split('-')[-1])

        hour_rate = load_pickle_from_minio(BUCKET, f"{PREFIX}hour_reorder_rate.pkl")
        day_rate = load_pickle_from_minio(BUCKET, f"{PREFIX}day_reorder_rate.pkl")
        p_days_rate = load_pickle_from_minio(BUCKET, f"{PREFIX}p_days_since_prior_order_reorder_rate.pkl")
        u_days_rate = load_pickle_from_minio(BUCKET, f"{PREFIX}u_days_since_prior_order_reorder_rate.pkl")
        up_days_rate = load_pickle_from_minio(BUCKET, f"{PREFIX}days_since_prior_reorder_rate.pkl")
        merged_up = load_pickle_from_minio(BUCKET, f"{PREFIX}merged_user_product_features.pkl")

        features = merged_up[merged_up['user_id'] == user_id]
        hour_r = hour_rate[hour_rate['order_hour_of_day'] == order_hour_of_day]
        day_r = day_rate[day_rate['order_dow'] == order_dow]
        p_days = p_days_rate[p_days_rate['days_since_prior_order'] == days_since_prior_order]
        u_days = u_days_rate[(u_days_rate['user_id'] == user_id) & (u_days_rate['days_since_prior_order'] == days_since_prior_order)]
        up_days = up_days_rate[(up_days_rate['user_id'] == user_id) & (up_days_rate['days_since_prior_order'] == days_since_prior_order)]

        if p_days.empty:
            p_days = pd.DataFrame({
                'product_id': features['product_id'],
                'days_since_prior_order': days_since_prior_order,
                'p_days_since_prior_order_reorder_rate': 0.0
            })
        if up_days.empty:
            up_days = pd.DataFrame({
                'user_id': user_id,
                'product_id': features['product_id'],
                'days_since_prior_order': days_since_prior_order,
                'days_since_prior_reorder_rate': 0.0
            })
        if u_days.empty:
            u_days = pd.DataFrame({
                'user_id': [user_id],
                'days_since_prior_order': [days_since_prior_order],
                'u_days_since_prior_order_reorder_rate': [0.0]
            })

        df = features.merge(up_days, on=['user_id', 'product_id'])
        df['days_since_prior_order'] = days_since_prior_order
        df = df.merge(hour_r, on='product_id', how='left')
        df = df.merge(day_r, on='product_id', how='left')
        df = df.merge(p_days, on=['product_id', 'days_since_prior_order'], how='left')
        df = df.merge(u_days, on=['user_id', 'days_since_prior_order'], how='left')

        df.fillna(0.0, inplace=True)

        model = load_pickle_from_minio(BUCKET, f"{PREFIX}catboost_model.pkl")

        df.columns = df.columns.str.replace('_x$', '', regex=True)
        df = df.loc[:, ~df.columns.str.endswith('_y')]
        df = df.loc[:, ~df.columns.duplicated()]

        X = df.drop(['user_id', 'product_id'], axis=1)
        ypred = model.predict_proba(X)[:, -1]
        df['preds'] = ypred

        top_products = df.sort_values(by='preds', ascending=False)['product_id'].head(10).tolist()
        product_map = load_pickle_from_minio(BUCKET, f"{PREFIX}product_mappings.pkl")
        final_names = product_map[product_map['product_id'].isin(top_products)]['product_name'].tolist()

        mlflow.set_tag("mode", "personalized")
        mlflow.log_metric("recommend_count", len(final_names))
        mlflow.log_metric("inference_time_sec", (datetime.now() - start_time).total_seconds())

        return {"cold_start": False, "recommendations": final_names}

def recommend_for_new_user():
    now = datetime.now()
    order_hour_of_day = now.hour
    order_dow = now.weekday()

    top = load_pickle_from_minio(BUCKET, f"{PREFIX}top10_products.pkl")
    top_products = top[
        (top['order_dow'] == order_dow) &
        (top['order_hour_of_day'] == order_hour_of_day)
    ]['product_name'].values.tolist()
    top_products = {i: prod for i, prod in enumerate(top_products)}

    with mlflow.start_run(run_name="cold_start", nested=True):
        mlflow.set_tag("mode", "cold_start_new_user")
        mlflow.log_param("order_hour", order_hour_of_day)
        mlflow.log_param("order_dow", order_dow)
        mlflow.log_metric("recommend_count", len(top_products))

    return {"cold_start": True, "recommendations": top_products}
