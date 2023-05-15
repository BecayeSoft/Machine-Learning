# Self-Organizing Maps

## What are Self-Organizing Maps (SOMs)?
Self-organizing maps, also known as Kohonen maps, are a type of artificial neural network that are used for clustering and dimensionality reduction. 

The goal of a SOM is to map high-dimensional input data onto a lower-dimensional grid while preserving the topological relationships between the input data points. Which is useful to reveal correlations that are not easy to understand.

The output of a self-organizing map is a two-dimensional grid of neurons, where each neuron represents a *specific region* of the *input space*. The neurons that are close to each other on the grid are also close to each other in the input space, which makes SOMs useful for visualizing high-dimensional data.
When a input is feed to the SOM, the neuron that is closest to that vector of features is the winner which represent a region

In the resulting grid, each square (neuron) is like a kid who knows one piece of the information. Kids that are close to each other in the grid know similar things. 
By looking at the big picture, we can see how all the kids relate to each other.

**Note**: SOMs are unsupervised, their nodes are not connected and they do not use backpropagation.

### Applications
We can use SOMs in:
* Clustering: group similar entities
* Data Visualization: visualize high dimensional space in 2D

More concretely:
* Recommender system: group similar movies, music, product reviews, etc.
* Identify trends and pattern: find similar employees, customers, social media topics, stock sectors performances, etc.

### Limitations
* Self-Organizing Maps do not perform well while working with categorical data and even worse for mixed types of data.
* The model preparation time is comparatively very slow and hard to train against the slowly evolving data.


Example
Following shows similar countries in 2D map extracted from possibly a lot of features such as GDP, quality of life, etc.

## How do SOMs Learn?
1. Assign random weights to the nodes
2. Calculate the distance between the data point (input row)  and each node
3. The smaller the distance, the closer is the node to the data point
4. Choose the closest node and drag it closer to the data point
5. When you drag a node, its neighbors (in the radius) are also dragged. The closer the node, the more drastically it is dragged.
6. Decrease the radius gradually, so that nodes farther away from the BMU are less affected by the updates.
7. Decrease the learning rate gradually
8. Repeat steps 2 to 7

💡 The closest node is known as  BMU (Best Matching Unit) or Winning Node.

## References

* https://algobeans.com/2017/11/02/self-organizing-map/
* https://analyticsindiamag.com/beginners-guide-to-self-organizing-maps/
* https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4