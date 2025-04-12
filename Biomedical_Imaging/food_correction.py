# Sample food dataset (name, price, calories, popularity score)
food_items = [
    ("Salad", 5, 150, 8),
    ("Pizza", 10, 600, 10),
    ("Burger", 8, 700, 9),
    ("Sushi", 12, 300, 7),
    ("Pasta", 9, 450, 8),
]

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