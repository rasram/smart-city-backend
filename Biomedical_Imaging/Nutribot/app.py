# app.py
import streamlit as st
import pandas as pd
from pyg_converter import nx_to_pyg
from train_gcn import train_gcn
from recommend import recommend_top_k
from knapsack import knapsack_select
from graph_builder import G  # Your NetworkX graph
from chat import get_bot_response

# Load dataset
food_df = pd.read_csv("Biomedical_Imaging/Nutribot/data/food_dataset.csv")  # Adjust path if needed

st.title("🥗 NutriBot - Smart Food Recommender")
st.markdown("Get the best food suggestions under your calorie goal!")

# Ask user for calorie limit
calorie_limit = st.number_input("Enter your target calorie limit (in calories):", min_value=100, max_value=3000, step=50)

# Button to start recommendation
if st.button("Get Recommendations"):

    # Run GCN pipeline
    st.info("Training model and generating recommendations...")
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

    # Show top 20 recommendations
    st.subheader("📊 Top 20 Recommended Foods")
    for item in formatted_foods:
        st.write(f"{item['name']}: {item['calories']} cal, score = {item['score']:.2f}")

    # Apply knapsack
    selected_foods = knapsack_select(formatted_foods, calorie_limit)

    if selected_foods:
        st.subheader("🍽️ Optimized Selection (Based on Your Calorie Goal)")
        for food in selected_foods:
            st.write(f"- {food['name']} ({food['calories']} cal, Score: {food['score']:.2f})")

        total_cal = sum(f['calories'] for f in selected_foods)
        total_score = sum(f['score'] for f in selected_foods)
        st.success(f"✅ Total Calories: {total_cal} | Total Score: {total_score:.2f}")
    else:
        st.warning("⚠️ No suitable combination found under your calorie limit.")

st.markdown("---")
st.header("🤖 Chat with NutriBot")

user_input = st.text_input("Ask something about food, nutrition, or recommendations...")

if st.button("Ask NutriBot"):
    if user_input.strip():
        st.info("NutriBot is thinking...")
        response = get_bot_response(user_input)
        st.success(response)
    else:
        st.warning("Please enter a question.")
