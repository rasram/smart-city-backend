def knapsack_select(foods, calorie_limit):
    n = len(foods)
    dp = [[0] * (calorie_limit + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        food = foods[i - 1]
        for w in range(1, calorie_limit + 1):
            if food["calories"] <= w:
                dp[i][w] = max(food["score"] + dp[i - 1][w - food["calories"]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    # Traceback to get selected items
    selected = []
    w = calorie_limit
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(foods[i - 1])
            w -= foods[i - 1]["calories"]

    return selected[::-1]  # Return in original order
