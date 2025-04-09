import google.generativeai as genai
from google.generativeai import GenerativeModel
import pandas as pd
import os
from dotenv import load_dotenv

# Load datasets
nutrition_df = pd.read_csv("Biomedical_Imaging/Nutribot/data/food_dataset.csv")

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(project_root, '.env')

load_dotenv(dotenv_path)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
global model
model = GenerativeModel('gemini-1.5-pro') 

# Build context combining both datasets
def build_context():
    context = (
        "You are a helpful assistant that provides detailed information about food nutrition "
        "and gives easy-to-follow recipes. Use the nutrition values and recipe procedures to "
        "assist users with meal planning, diet tracking, or cooking instructions.\n\n"
    )

    context += "Here is the detailed nutrition and recipe information for each food item:\n\n"

    for _, row in nutrition_df.iterrows():
        context += (
            f"🍽️ **Food Name**: {row['name']}\n"
            f"   - Calories: {row['calories']} kcal\n"
            f"   - Spicy: {'Yes' if row['spicy'] else 'No'}\n"
            f"   - Sweet: {'Yes' if row['sweet'] else 'No'}\n"
            f"   - Vegan: {'Yes' if row['vegan'] else 'No'}\n"
            f"   - Protein Level: {row['protein']}/5\n"
            f"   - Carbohydrates Level: {row['carbs']}/5\n"
            f"   - Fat Level: {row['fat']}/5\n\n"
            f"👩‍🍳 **Recipe Instructions**:\n"
            f"   {row['recipe']}\n"
            f"{'-'*60}\n"
        )

    return context


# Get response from LLM
def get_bot_response(user_query):
    context = build_context()
    prompt = context + f"\nUser query: {user_query}"

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"
