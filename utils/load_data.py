import pandas as pd
import os
def load_instacart_data(base_path='../data/raw'):
   # loading all Instacart CSVs as a dictionary of DataFrames
    
    data = {
        'orders': pd.read_csv(os.path.join(base_path, 'orders.csv')),
        'products': pd.read_csv(os.path.join(base_path, 'products.csv')),
        'order_products_prior': pd.read_csv(os.path.join(base_path, 'order_products__prior.csv')),
        'order_products_train': pd.read_csv(os.path.join(base_path, 'order_products__train.csv')),
        'departments': pd.read_csv(os.path.join(base_path, 'departments.csv')),
        'aisles': pd.read_csv(os.path.join(base_path, 'aisles.csv')),
    }

    reviews_path = os.path.join(base_path, 'Reviews.csv')
    if os.path.exists(reviews_path):
        data['reviews'] = pd.read_csv(reviews_path)

    return data


