# recommend.py
import torch
import torch.nn.functional as F

def recommend_top_k(model, data, node_mapping, k=20):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

    # Find the index of the user node
    user_node = [idx for name, idx in node_mapping.items() if "user" in name][0]
    user_vec = out[user_node]

    food_scores = []
    for name, idx in node_mapping.items():
        if "food" in name:
            food_vec = out[idx]
            similarity = F.cosine_similarity(user_vec.unsqueeze(0), food_vec.unsqueeze(0)).item()
            food_scores.append((name, similarity))

    # Sort and get top k
    top_k = sorted(food_scores, key=lambda x: x[1], reverse=True)[:k]
    print("\n🍽️ Top-20 Recommended Foods:")
    for name, score in top_k:
        print(f"{name}: similarity = {score:.4f}")

    return top_k
