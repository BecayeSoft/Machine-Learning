# Self-Organizing Maps

Self-organizing maps (SOMs), also known as Kohonen maps, are a type of artificial neural network that are used for clustering and dimensionality reduction. 

## Overview of SOMs

The goal of a SOM is to map high-dimensional input data onto a lower-dimensional grid while preserving the topological relationships (neurons that are spatially close on the map are also similar in terms of the input data they represent) between the input data points.

The output of a self-organizing map is a two-dimensional grid of neurons, where each neuron represents a *specific region* of the *input space*. The neurons that are close to each other on the grid are also close to each other in the input space, which makes SOMs useful for visualizing high-dimensional data.
When a input is feed to the SOM, the neuron that is closest to that vector of features is the winner which represent a region

The output of a SOM is a 2D grid (of neurons) where each square (neuron) is like a kid who knows one piece of the information. Kids that are close to each other in the grid know similar things. 
By looking at the big picture, we can see how all the kids relate to each other.

The following is a hexagonal SOM. Neurons that are close to each other have similar colors and represent similar countries.

![image](https://github.com/BecayeSoft/Machine-Learning/assets/87549214/82fe282a-2f34-4f45-85a3-9f52d20a6f80)

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

💡 The closest node is known as  BMU (Best Matching Unit) or Winning Node.

<img width="463" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/94005a3f-275d-4158-89be-eb6559117756">

5. When you drag a node, its neighbors (in the radius) are also dragged. The closer the node, the more drastically it is dragged.

<img width="452" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/8cab4928-2022-4712-892b-7527f17a8487">

6. Decrease the radius gradually, so that nodes farther away from the BMU are less affected by the updates.

<img width="272" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/9eecae83-df9f-40e5-9062-c5c9d6ab7dbb">

7. Repeat steps 2 to 7

In the resulting map, each winning node is assigned a particular cluster.

<img width="274" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/03a992b1-fcc6-41c6-ba29-c7a61e0a0cda">

## Example on Iris data
The following code uses the iris dataset.

When creating a SOM, we can use the distance map which is a 2D array (same size as the grid. E.g.: 12x12) where each element represents the average distance between a neuron and its neighbors.

```
u_matrix = som.distance_map().T
print(u_matrix)
```

```
[[0.081933 0.165542 0.177204 0.587590 0.667328 0.431389 0.356502 0.183952]
 [0.124169 0.240682 0.338728 0.826771 1.000000 0.640208 0.617496 0.432710]
 [0.144016 0.252570 0.461191 0.948480 0.854452 0.647242 0.627267 0.458241]
 [0.138695 0.365867 0.675865 0.845140 0.637664 0.536530 0.560507 0.291140]
 [0.192298 0.572329 0.831626 0.578588 0.394775 0.467025 0.422079 0.244352]
 [0.428055 0.803671 0.713910 0.407951 0.385777 0.431611 0.434741 0.235023]
 [0.513925 0.745877 0.520288 0.347792 0.403915 0.440234 0.439856 0.243294]
 [0.226807 0.369797 0.231559 0.262100 0.241235 0.302932 0.252746 0.163932]]
```

We plot the results. Light shades represent the clusters whereas the dark ones represents the division of the clusters. 
However, some people plot their distance map the other way where dark shades represents clusters.

```
plt.pcolor(u_matrix, cmap='bone_r')
plt.colorbar()
```
<img width="304" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/b398409d-9bea-4d4e-a9eb-c3861c2de61b">


## References

* [Deep Learning A-Z™ 2023: Neural Networks, AI & ChatGPT Bonus](https://www.udemy.com/course/deeplearning/) - Course by SuperDataScience
* https://algobeans.com/2017/11/02/self-organizing-map/
* https://analyticsindiamag.com/beginners-guide-to-self-organizing-maps/
* https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4
