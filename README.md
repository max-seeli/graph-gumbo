# graph-gumbo

![Gumbo](./img/gumba.png)

Welcome to graph-gumbo! This project is an exploration into the world of graph theory, focusing on the interplay between different graph products and graph embeddings.

## Project Overview
At the core of this project is the study of three major graph products:

- Cartesian Product: The product of two graphs that pairs the vertices of the two graphs and connects pairs if they were adjacent in their respective graphs.
- Strong Product: A blend of Cartesian and direct products, this product connects pairs of vertices that are adjacent in at least one of the original graphs.
- Tensor Product: Also known as the direct product, it connects vertices only when both corresponding vertices are connected in their original graphs.

The project investigates how these products impact the expressiveness of various graph embeddings, focusing on:

- Cycle Counting: An algorithm for detecting cycles within a graph, which is used to understanding the graph's topology.
- Weisfeiler-Lehman (WL) Embedding: A powerful method for graph isomorphism testing.

## Getting Started
To get started, clone this repository and install the required dependencies. The project is written using Python 3.11.5, and the dependencies can be installed using the following command:

**Pip**
```
pip install -r requirements.txt
```

**Conda**
```
conda create --name graph-gumbo --file requirements.txt
```
## Gumbo
The name **gumbo** pays homage to the [Gumbo](https://en.wikipedia.org/wiki/Gumbo) dish, which is a stew that combines a variety of ingredients to create a delicious meal. Similarly, this project combines a variety of graph products to create a powerful graph embedding.

