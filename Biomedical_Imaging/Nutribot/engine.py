# engine.py
import pandas as pd
from .pyg_converter import nx_to_pyg
from .train_gcn import train_gcn
from .recommend import recommend_top_k
from .knapsack import knapsack_select
from .graph_builder import G  # Your NetworkX graph
from .chat import get_bot_response

# Load dataset once
food_df = pd.read_csv("Biomedical_Imaging/Nutribot/data/dataset.csv")  # Adjust path if needed

def get_recommendations(calorie_limit: int):
    """
    Generate food recommendations based on calorie limit.
    """
    # Run GCN pipeline
    pyg_data, node_map = nx_to_pyg(G)
    model = train_gcn(pyg_data)
    top_foods = recommend_top_k(model, pyg_data, node_map, k=20)

    # Prepare foods with calorie and name info
    formatted_foods = []
    for food_id_str, score in top_foods:
        food_id = int(food_id_str.split("_")[1])  # from 'food_12' -> 12
        row = food_df[food_df["food_id"] == food_id]  # Update this key if needed
        if not row.empty:
            name = row.iloc[0]["name"]
            calories = int(row.iloc[0]["calories"])
            formatted_foods.append({
                "name": name,
                "calories": calories,
                "score": float(score)
            })

    # Apply knapsack
    selected_foods = knapsack_select(formatted_foods, calorie_limit)

    return {
        "top_20": formatted_foods,
        "optimized_selection": selected_foods
    }

def get_nutribot_response(user_input: str):
    """
    Get NutriBot's response to a user query.
    """
    return get_bot_response(user_input)
