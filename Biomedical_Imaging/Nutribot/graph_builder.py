# graph_builder.py
import pandas as pd
import networkx as nx
from .user_simulator import generate_user_profile

# Load the food dataset
food_df = pd.read_csv("Biomedical_Imaging/Nutribot/data/food_dataset.csv")

# Simulate a user
user = generate_user_profile()
user_id = user["user_id"]

# Create a bipartite graph
G = nx.Graph()

# Add user node
G.add_node(f"user_{user_id}", bipartite=0, **user)

# Add food nodes and connect them to the user based on preferences
for _, food in food_df.iterrows():
    food_node = f"food_{food['food_id']}"
    G.add_node(food_node, bipartite=1, **food.to_dict())

    # Matching criteria
    match_score = 0
    if user["likes_spicy"] == food["spicy"]:
        match_score += 1
    if user["likes_sweet"] == food["sweet"]:
        match_score += 1
    if user["is_vegan"] == food["vegan"]:
        match_score += 1
    if user["prefers_high_protein"] and food["protein"] >= 3:
        match_score += 1

    if match_score >= 2:  # Threshold: at least 2 preferences match
        G.add_edge(f"user_{user_id}", food_node, weight=match_score)

print(f"✅ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
