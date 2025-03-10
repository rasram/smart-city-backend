from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample food dataset (name, price, calories, popularity score)
food_items = [
    ("Salad", 5, 150, 8),
    ("Pizza", 10, 600, 10),
    ("Burger", 8, 700, 9),
    ("Sushi", 12, 300, 7),
    ("Pasta", 9, 450, 8),
]

# Restaurant graph (adjacency list format)
restaurants = {
    "A": {"B": 2, "C": 5},
    "B": {"A": 2, "C": 3, "D": 4},
    "C": {"A": 5, "B": 3, "D": 2},
    "D": {"B": 4, "C": 2},
}

# Greedy Knapsack Algorithm for food selection
def greedy_knapsack(budget):
    sorted_food = sorted(food_items, key=lambda x: x[2] / x[1], reverse=True)
    selected_items = []
    total_price = 0
    for food in sorted_food:
        if total_price + food[1] <= budget:
            selected_items.append(food[0])
            total_price += food[1]
    return selected_items

# 0/1 Knapsack DP for optimal meal selection
def knapsack_dp(budget):
    n = len(food_items)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(budget + 1):
            if food_items[i - 1][1] <= w:
                dp[i][w] = max(food_items[i - 1][3] + dp[i - 1][w - food_items[i - 1][1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    
    selected_items = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(food_items[i - 1][0])
            w -= food_items[i - 1][1]
    
    return selected_items

# Dijkstra’s Algorithm for closest restaurant
def dijkstra(graph, start):
    unvisited = set(graph.keys())
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {}
    
    while unvisited:
        current = min(unvisited, key=lambda node: distances[node])
        unvisited.remove(current)
        
        for neighbor, weight in graph[current].items():
            new_distance = distances[current] + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current
    
    target = min(distances, key=distances.get)
    path = []
    while target in previous_nodes:
        path.insert(0, target)
        target = previous_nodes[target]
    path.insert(0, start)
    
    return path

def find_closest_restaurant(start):
    return dijkstra(restaurants, start)

# PageRank Algorithm for ranking popular dishes
def rank_dishes():
    popularity = {food[0]: food[3] for food in food_items}
    return sorted(popularity.items(), key=lambda x: x[1], reverse=True)

# Compute Levenshtein Distance from scratch
def levenshtein_distance(str1, str2):
    len1, len2 = len(str1), len(str2)
    dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[len1][len2]

# Handling typos with Levenshtein Distance
def correct_food_name(query):
    return min(food_items, key=lambda x: levenshtein_distance(query, x[0]))[0]

@app.route('/recommend', methods=['GET'])
def recommend_food():
    budget = int(request.args.get('budget', 10))
    method = request.args.get('method', 'greedy')
    if method == 'greedy':
        recommendations = greedy_knapsack(budget)
    else:
        recommendations = knapsack_dp(budget)
    return jsonify({'Recommended Dishes': recommendations})

@app.route('/closest_restaurant', methods=['GET'])
def closest_restaurant():
    start = request.args.get('location', 'A')
    path = find_closest_restaurant(start)
    return jsonify({'Closest Restaurant Path': path})

@app.route('/rank_dishes', methods=['GET'])
def rank_popular_dishes():
    ranked = rank_dishes()
    return jsonify({'Dish Rankings': ranked})

@app.route('/correct_food', methods=['GET'])
def correct_food():
    query = request.args.get('query', '')
    corrected_name = correct_food_name(query)
    return jsonify({'Corrected Food Name': corrected_name})

if __name__ == '__main__':
    app.run(debug=True)
