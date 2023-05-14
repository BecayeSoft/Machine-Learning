# Self-Organizing Maps

Self-organizing maps (SOMs), also known as Kohonen maps, are a type of artificial neural network that are used for clustering and dimensionality reduction. 

* `Credit-card-applications`: SOM to detect outliers
* `Iris`: SOM to group plant species into clusters
* `SOM-from-scratch`: SOM with no library.

My [article](https://medium.com/p/9bb88b9a620a) can also be found on Medium.

## Overview of SOMs

The goal of a SOM is to map high-dimensional input data onto a lower-dimensional grid while preserving the topological relationships (neurons that are spatially close on the map are also similar in terms of the input data they represent) between the input data points.

The output of a SOM is a 2D grid (of neurons) where each square (neuron) is like a kid who knows one piece of the information. Kids that are close to each other in the grid know similar things. 
By looking at the big picture, we can see how all the kids relate to each other.

The following is a hexagonal SOM. Neurons that are close to each other have similar colors and represent similar countries.

![image](https://github.com/BecayeSoft/Machine-Learning/assets/87549214/82fe282a-2f34-4f45-85a3-9f52d20a6f80)

Source: [SEG Wiki](https://wiki.seg.org/wiki/File:Self_Organizing_Map.png)

Notice how Canada and the USA are close to each other.

**Note**: SOMs are unsupervised, their nodes (neurons) are not connected and they do not use backpropagation.

### Applications
We can use SOMs for:
* Clustering: grouping similar entities.
* Data Visualization: visualizing high dimensional space in 2D.
* Anomaly detection: identifying uncommon behavior.
* etc.

More concretely, we can use SOMs to:
* Build recommender system: grouping similar movies, music, product reviews, etc.
* Identify trends and patterns: find similar employees, customers, social media topics, stock sector performances, etc.
* Detect outliers: Detect and prevent fraud, security breaches, and other abnormal behavior.

### Limitations
SOMs have some limitations, such as:
* Poor performance with categorical data
* Lack of interpretability
* Limited ability to handle high-dimensional data
* Limited generalization ability: not well-suited for unseen data
* Difficulty with non-linear relationships

## How do SOMs Learn?
A SOM is a grid of neurons. Each neuron has N weights, where N is the number of dimensions (features) in the data. 
The training process can be summarized in seven steps:

1. Assign random weights to the neurons.
2. Select an input data point.
3. Calculate the distance between the data point and each neuron.
4. Identify the neuron with the smallest distance to the data point. 

This neuron is referred to as the Best Matching Unit (BMU) or winning neuron. A small distance means that the neuron is very close to the data point.

<img width="463" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/94005a3f-275d-4158-89be-eb6559117756">

5. Drag the winning neuron closer to the data point.
When you drag a neuron, its neighbors (in the radius) are also dragged. The closer the neuron, the more drastically it is dragged, resulting in a larger change in its weight vector.

<img width="452" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/8cab4928-2022-4712-892b-7527f17a8487">

6. Decrease the radius gradually, so that neurons farther away from the BMU are less affected by the updates.

<img width="272" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/9eecae83-df9f-40e5-9062-c5c9d6ab7dbb">

> A radius that is too small could lead to overfitting while a bigger one could underfit the data.

7. Repeat steps 2 to 7

In the resulting map, each winning neuron is assigned to a particular cluster based on its location and the distribution of the input data. Neurons that are close to each other in the SOM are likely to be assigned to the same cluster, reflecting similarities in the underlying data patterns.

<img width="274" alt="image" src="https://github.com/BecayeSoft/Machine-Learning/assets/87549214/03a992b1-fcc6-41c6-ba29-c7a61e0a0cda">

## Example on the iris dataset
Here is an example of using SOMs on the Iris dataset to classify different species based on their petal and sepal measurements.
When creating a SOM, we can use the distance map to visualize the results. The distance map is a 2D array with the same size as the grid (e.g. 12x12), where each element represents the average distance between a neuron and its neighbors.

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

We can visualize the results, with light shades indicating the clustering and darker shades representing the boundaries between clusters. 
However, some may prefer to plot their distance map the other way around with dark shades representing the clusters instead.
```
plt.pcolor(u_matrix, cmap='bone_r')
plt.colorbar()
```
![image](https://github.com/BecayeSoft/Machine-Learning/assets/87549214/3cfad061-a97a-435e-8c7a-d2a6d5d491d2)

Notice the three different clusters.
> Note that the code above only shows the first plot, which is on the left.

## References

* [Deep Learning A-Z™ 2023: Neural Networks, AI & ChatGPT Bonus](https://www.udemy.com/course/deeplearning/) - Course by SuperDataScience
* https://algobeans.com/2017/11/02/self-organizing-map/
* https://analyticsindiamag.com/beginners-guide-to-self-organizing-maps/
* https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4
