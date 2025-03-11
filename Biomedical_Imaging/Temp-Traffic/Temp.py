import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import messagebox

INF = float('inf')

# Generate a fixed synthetic graph
def generate_fixed_graph(n, density=0.5, weight_range=(1, 10)):
    graph = np.full((n, n), INF, dtype=float)
    for i in range(n):
        graph[i][i] = 0
    
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < density:
                graph[i][j] = random.randint(*weight_range)
    return graph

def floyd_warshall(graph, n):
    dist = np.array(graph, dtype=float)
    next_node = np.full((n, n), -1, dtype=int)
    
    for i in range(n):
        for j in range(n):
            if graph[i][j] != INF and i != j:
                next_node[i][j] = j
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    return dist, next_node

def get_path(next_node, u, v):
    if next_node[u][v] == -1:
        return []
    path = [u]
    while u != v:
        u = next_node[u][v]
        path.append(u)
    return path

def second_shortest_path(graph, n, u, v, traffic_edges):
    dist, next_node = floyd_warshall(graph, n)
    shortest_path = get_path(next_node, u, v)
    if not shortest_path:
        return INF, []
    
    second_best = INF
    second_path = []
    
    for i in range(len(shortest_path) - 1):
        modified_graph = np.array(graph, dtype=float)
        modified_graph[shortest_path[i]][shortest_path[i+1]] = INF  
        
        for edge in traffic_edges:
            modified_graph[edge[0]][edge[1]] = INF
        
        new_dist, new_next_node = floyd_warshall(modified_graph, n)
        new_path = get_path(new_next_node, u, v)
        
        if new_path and new_dist[u][v] < second_best:
            second_best = new_dist[u][v]
            second_path = new_path
    
    return second_best, second_path

def generate_fixed_traffic(graph, n, probability=0.3):
    traffic_edges = []
    for i in range(n):
        for j in range(n):
            if graph[i][j] != INF and i != j and random.random() < probability:
                traffic_edges.append((i, j))
    return traffic_edges

def simulate_traffic(graph, n, u, v, traffic_edges):
    dist, next_node = floyd_warshall(graph, n)
    shortest_path = get_path(next_node, u, v)
    
    if any((shortest_path[i], shortest_path[i+1]) in traffic_edges for i in range(len(shortest_path) - 1)):
        second_dist, second_path = second_shortest_path(graph, n, u, v, traffic_edges)
        return second_path, second_dist, True
    else:
        return shortest_path, dist[u][v], False

def visualize_graph(graph, title, path=[], traffic_edges=[]):
    G = nx.DiGraph()
    n = len(graph)
    
    for i in range(n):
        for j in range(n):
            if graph[i][j] != INF and i != j:
                G.add_edge(i, j, weight=graph[i][j])
    
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    
    edge_colors = ['red' if edge in traffic_edges else 'blue' if edge in zip(path, path[1:]) else 'gray' for edge in G.edges()]
    
    plt.figure()
    plt.title(title)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, edge_color=edge_colors, width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def find_path():
    u = int(start_entry.get())
    v = int(end_entry.get())
    final_path, final_dist, traffic = simulate_traffic(graph, nodes, u, v, traffic_edges)
    
    visualize_graph(graph, "Original Graph")
    visualize_graph(graph, "Shortest Path", final_path, traffic_edges)
    
    if traffic:
        second_dist, second_path = second_shortest_path(graph, nodes, u, v, traffic_edges)
        messagebox.showinfo("Route", f"Traffic detected! Use second shortest path: {second_path} with distance {second_dist}")
        visualize_graph(graph, "Second Shortest Path", second_path, traffic_edges)
    else:
        messagebox.showinfo("Route", f"Shortest path: {final_path} with distance {final_dist}")

# Generate fixed dataset
nodes = 6
graph = generate_fixed_graph(nodes, density=0.6, weight_range=(1, 15))
traffic_edges = generate_fixed_traffic(graph, nodes)

distances, _ = floyd_warshall(graph, nodes)
print("Shortest Distance Matrix:")
print(distances)

# Tkinter GUI
root = tk.Tk()
root.title("Traffic Route Management")

tk.Label(root, text="Start Node:").grid(row=0, column=0)
start_entry = tk.Entry(root)
start_entry.grid(row=0, column=1)

tk.Label(root, text="End Node:").grid(row=1, column=0)
end_entry = tk.Entry(root)
end_entry.grid(row=1, column=1)

tk.Button(root, text="Find Path", command=find_path).grid(row=2, columnspan=2)
root.mainloop()