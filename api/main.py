# main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from api.model_utils import get_recommendations, recommend_for_new_user


app = FastAPI(title="Instacart Recommender")

@app.get("/")
def read_root():
    return {"message": "Instacart Recommendation API is live!"}

@app.get("/recommend/user")
def recommend_user(user_id: int = Query(..., description="User ID to generate recommendations")):
    result = get_recommendations(user_id)
    return result
@app.get("/recommend/new")
def recommend_new_user():
    result = recommend_for_new_user()
    return result