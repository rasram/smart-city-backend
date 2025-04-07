# create_food_dataset.py

import pandas as pd
import random

food_names = [
    "Grilled Chicken", "Veg Salad", "Spicy Paneer", "Fruit Bowl", "Butter Chicken",
    "Mushroom Curry", "Pasta Alfredo", "Choco Cake", "Tofu Stir Fry", "Tomato Soup",
    "Cheese Sandwich", "Green Smoothie", "Lentil Soup", "Pancakes", "Fried Rice",
    "Veg Biryani", "Mutton Curry", "Chicken Tikka", "Dal Tadka", "Sushi Rolls",
    "Veggie Wrap", "Cereal Bowl", "Omelette", "Keto Salad", "Peanut Butter Toast",
    "Chocolate Muffin", "Apple Pie", "Gulab Jamun", "Rajma Chawal", "Upma",
    "Idli Sambar", "Dosa", "Bhel Puri", "Rasam Rice", "Fish Curry", "Egg Curry",
    "Chana Masala", "Veg Manchurian", "Spring Rolls", "Hummus Pita", "Paneer Wrap",
    "Nachos", "Corn Soup", "Fruit Yogurt"
]

# Define recipe generator based on keywords
def generate_recipe(name):
    name = name.lower()
    if "chicken" in name:
        return "Marinate chicken with spices and yogurt. Grill or cook until done. Serve with rice or naan."
    elif "salad" in name:
        return "Mix fresh veggies or greens with olive oil, lemon juice, and seasoning. Chill and serve."
    elif "paneer" in name:
        return "Sauté paneer cubes with onions, tomatoes, and Indian spices. Garnish with coriander."
    elif "fruit" in name:
        return "Chop fresh fruits and mix with honey or yogurt. Serve chilled."
    elif "cake" in name or "muffin" in name or "pie" in name:
        return "Mix flour, sugar, eggs, and flavoring. Bake until golden and fluffy."
    elif "soup" in name:
        return "Boil main ingredients with spices, blend, and simmer. Serve hot with croutons or bread."
    elif "wrap" in name or "roll" in name:
        return "Stuff tortilla with veggies or meat and sauces. Wrap and grill lightly."
    elif "biryani" in name or "fried rice" in name:
        return "Cook rice with vegetables or meat and spices. Mix well and serve hot."
    elif "curry" in name:
        return "Cook the main ingredient with onions, tomatoes, and rich curry masala."
    elif "smoothie" in name:
        return "Blend fruits with yogurt or milk until smooth. Add honey to taste."
    elif "dal" in name or "chana" in name or "rajma" in name:
        return "Boil lentils/beans and cook with tomatoes, garlic, and spices. Serve with rice."
    elif "omelette" in name:
        return "Beat eggs with onion, chili, salt. Fry until golden and fluffy."
    elif "dosa" in name or "idli" in name or "upma" in name:
        return "Prepare batter or mix, cook on pan or steam, and serve with chutney/sambar."
    elif "nachos" in name:
        return "Layer chips with cheese, beans, salsa, and jalapenos. Bake and serve."
    elif "pasta" in name:
        return "Cook pasta and toss with creamy Alfredo sauce and veggies or chicken."
    elif "hummus" in name or "pita" in name:
        return "Blend chickpeas with tahini and olive oil. Serve with warm pita bread."
    elif "sushi" in name:
        return "Roll vinegared rice with fish or veggies in seaweed. Slice and serve."
    else:
        return "Recipe not available for this item."

data = []

for i, name in enumerate(food_names):
    food = {
        "food_id": i,
        "name": name,
        "calories": random.randint(120, 500),
        "spicy": random.randint(0, 1),
        "sweet": random.randint(0, 1),
        "vegan": random.randint(0, 1),
        "protein": random.randint(1, 5),
        "carbs": random.randint(1, 5),
        "fat": random.randint(1, 5),
        "recipe": generate_recipe(name)
    }
    data.append(food)

df = pd.DataFrame(data)
df.to_csv("food_dataset.csv", index=False)

print("✅ Dataset with recipes created! Total items:", len(df))
