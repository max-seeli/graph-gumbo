import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import os
from typing import Tuple

def main():
    # Define the graphs P3 and K3
    P3 = nx.path_graph(3)
    K3 = nx.complete_graph(3)
    
    # Create Cartesian, Strong, and Tensor products of P3 and K3
    G_cartesian = nx.cartesian_product(P3, K3)
    G_strong = nx.strong_product(P3, K3)
    G_tensor = nx.tensor_product(P3, K3)
    
    # Plotting the graphs
    fig_factors, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # Original Graphs
    nx.draw(P3, ax=axes[0], with_labels=True, node_color='lightblue', font_weight='bold')
    axes[0].set_title("Path Graph P3")
    nx.draw(K3, ax=axes[1], with_labels=True, node_color='lightgreen', font_weight='bold')
    axes[1].set_title("Complete Graph K3")
    
    plt.show()
    
    fig_products, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    pos = nx.spring_layout(G=G_cartesian, seed=14)
    
    # Cartesian Product
    nx.draw(G_cartesian, pos=pos, ax=axes[0], with_labels=True, node_color='lightcoral', font_weight='bold')
    axes[0].set_title("Cartesian Product P3 □ K3")
    
    # Strong Product
    nx.draw(G_strong, pos=pos, ax=axes[1], with_labels=True, node_color='lightcoral', font_weight='bold')
    axes[1].set_title("Strong Product P3 ⊠ K3")
    
    # Tensor Product
    nx.draw(G_tensor, pos=pos, ax=axes[2], with_labels=True, node_color='lightcoral', font_weight='bold')
    axes[2].set_title("Tensor Product P3 ⊗ K3")
    
    plt.show()
    
    
    # Save the plots at 'img/plots' for a paper in the pdf format
    if not os.path.exists('img/plots'):
        os.makedirs('img/plots')
    fig_factors.savefig('img/plots/graph_factor_examples.pdf', format='pdf', dpi=300, bbox_inches='tight')
    fig_products.savefig('img/plots/graph_product_examples.pdf', format='pdf', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()