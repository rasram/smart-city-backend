# main.py
from pyg_converter import nx_to_pyg
from train_gcn import train_gcn
from recommend import recommend_top_k
from graph_builder import G
from knapsack import knapsack_select  # Your knapsack function
import pandas as pd
import sys

# Step 1: Prepare data
pyg_data, node_map = nx_to_pyg(G)
model = train_gcn(pyg_data)
top_foods = recommend_top_k(model, pyg_data, node_map, k=20)  # [('food_2', 0.78), ...]

print(top_foods)




# Step 2: Load food dataset
food_df = pd.read_csv("data/dataset.csv")  # Ensure it has 'id', 'name', 'calories'

# Step 3: Ask user for calorie limit
try:
    calorie_limit = int(input("Enter your target calorie limit (in calories): "))
except ValueError:
    print("❌ Invalid input. Please enter a number.")
    sys.exit(1)

# Step 4: Match top_foods with their metadata
formatted_foods = []
for food_id_str, score in top_foods:
    food_id = int(food_id_str.split("_")[1])
    row = food_df[food_df["food_id"] == food_id]
    if not row.empty:
        food_info = row.iloc[0]
        formatted_foods.append({
            "name": food_info["name"],
            "calories": int(food_info["calories"]),
            "score": float(score)
        })

print("\n📊 Top 20 foods with calorie info:")
for f in formatted_foods:
    print(f"{f['name']}: {f['calories']} cal, score = {f['score']:.2f}")

# Step 5: Apply Knapsack DP
selected_foods = knapsack_select(formatted_foods, calorie_limit)

# Step 6: Display Results
print("\n🍽️ NutriBot's optimized food recommendations under your calorie goal:\n")
for food in selected_foods:
    print(f"- {food['name']} ({food['calories']} cal, Score: {food['score']:.2f})")

total_calories = sum(f['calories'] for f in selected_foods)
total_score = sum(f['score'] for f in selected_foods)
print(f"\n✅ Total Calories: {total_calories} | Total Score: {total_score:.2f}")
