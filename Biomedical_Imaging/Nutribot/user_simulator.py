# user_simulator.py
import random

def generate_user_profile():
    user_profile = {
        "user_id": 0,  # assuming single user for now
        "likes_spicy": random.randint(0, 1),
        "likes_sweet": random.randint(0, 1),
        "is_vegan": random.randint(0, 1),
        "prefers_high_protein": random.randint(0, 1),
        "calorie_limit": random.choice([300, 400, 500, 600])
    }
    return user_profile

if __name__ == "__main__":
    user = generate_user_profile()
    print("🧍 Simulated User Preferences:")
    for key, value in user.items():
        print(f" - {key}: {value}")
