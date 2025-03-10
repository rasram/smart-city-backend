import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:5000"

st.title("Food Recommendation Chatbot")

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a function", ["Recommend Food", "Find Closest Restaurant", "Rank Popular Dishes", "Correct Food Name"])

if page == "Recommend Food":
    st.subheader("Food Recommendation System")
    budget = st.number_input("Enter your budget:", min_value=1, value=10)
    method = st.selectbox("Select recommendation method:", ["greedy", "dp"])
    if st.button("Get Recommendations"):
        response = requests.get(f"{BASE_URL}/recommend", params={"budget": budget, "method": method})
        if response.status_code == 200:
            st.write("Recommended Dishes:", response.json()["Recommended Dishes"])
        else:
            st.error("Error fetching recommendations")

elif page == "Find Closest Restaurant":
    st.subheader("Find the Closest Restaurant")
    start_location = st.text_input("Enter your starting location:", "A")
    if st.button("Find Closest Restaurant"):
        response = requests.get(f"{BASE_URL}/closest_restaurant", params={"location": start_location})
        if response.status_code == 200:
            st.write("Closest Restaurant Path:", response.json()["Closest Restaurant Path"])
        else:
            st.error("Error fetching restaurant path")

elif page == "Rank Popular Dishes":
    st.subheader("Popular Dish Rankings")
    if st.button("Get Rankings"):
        response = requests.get(f"{BASE_URL}/rank_dishes")
        if response.status_code == 200:
            st.write("Dish Rankings:", response.json()["Dish Rankings"])
        else:
            st.error("Error fetching rankings")

elif page == "Correct Food Name":
    st.subheader("Correct Food Name (Typo Handling)")
    query = st.text_input("Enter a food name:")
    if st.button("Correct Name"):
        response = requests.get(f"{BASE_URL}/correct_food", params={"query": query})
        if response.status_code == 200:
            st.write("Corrected Food Name:", response.json()["Corrected Food Name"])
        else:
            st.error("Error correcting food name")
