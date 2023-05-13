import random
import numpy as np
import matplotlib.pyplot as plt


class SOM:

    def __init__(self, input_size, map_size):
        """
        Initializes a Self Organizing Maps.

        Parameters
        ----------
        input_size: is the number of features in the input data
        map_size: tuple
            Size of the SOM map, which is the number of neurons or nodes in the grid.
        weights: np array
            Weights of the SOM neurons in a 2D numpy array of the same shape as `map_size`.
        """
        self.input_size = input_size
        self.map_size = map_size
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)

    def train(self, data, n_iter, lr):
        """
        Train the SOM.

        Parameters
        ----------
        data: 2D np array
            Input data. An array of shape (n_samples, input_size).
        n_iter: number
            Number of iterations
        lr: float
            Learning rate
        """

        # Select random row from the data
        # Find the Best Matching Unit
        # Update neurons weights in the  BMU neighborhood
        for i in range(n_iter):
            v = data[random.randint(0, len(data) - 1)]
            bmu = self._find_bmu(v)
            self._update_weights(v, bmu, lr, i, n_iter)

    def _find_bmu(self, v):
        """
        Find the best matching unit (BMU) by computing the euclidan distance 
        between the input vector `v` and the neurons weights.

        The BMU is the neuron which is the closest to the input vector `v`.
        
        Parameters
        ----------
        v: np array
            Input vector, which is a row from the dataset.

        Returns
        -------
        bmu: tuple
            (x,y) coordinates of the BMU in the map
        """

        # Euclidean distance between `v` and weights of each neuron
        distances = np.linalg.norm(self.weights - v, axis=2)
        bmu_index = np.argmin(distances)

        # convert the index i of the BMU in a (x,y) coordinates in the map
        bmu = np.unravel_index(bmu_index, self.map_size)

        return bmu

    def _update_weights(self, v, bmu, lr, iteration, n_iter):
        """
        Update the weights of all the neurons in the SOM.

        Parameters
        ----------
        v: np array
            input vector that was fed into the SOM.
            
        bmu: tuple
            Coordinates of the BMU in the SOM grid.

        lr: scalar
            Learning rate for the current iteration. A scalar value between 0 and 1 
            that controls the amount by which the weights are updated.

        iter: integer
            Current iteration of the SOM training process.

        n_iter: integer
            Total number of iterations to train the the SOM.
        """
        n_rows = self.map_size[0]
        n_cols = self.map_size[1]

        # Determine radius and compute the Gaussian neighborhood
        radius = self.map_size[0] / 2 * np.exp(-iteration / n_iter)
        neighborhood = self._gaussian(bmu, radius)

        # decrase learning rate
        decay = np.exp(-iteration / n_iter)

        # Update the weights of the neurons in the neighborhood of the BMU
        for i in range(n_rows):
            for j in range(n_cols):
                # weights of the neuron at position (i,j)
                w = self.weights[i, j]

                # distance between current neuron position and BMU position
                distance = np.linalg.norm([i - bmu[0], j - bmu[1]])

                # if the current neuron is in the radius
                if distance <= radius:
                    # 
                    influence = neighborhood[i, j] * decay
                    w += lr * influence * (v - w)

                self.weights[i, j] = w

    def _gaussian(self, bmu, radius):
        """
        Compute the Euclidian distance between the BMU and each neuron,
        then return a value inversely proportional to the distance,
        i.e the lower the distance (close neuron), the higher the returned value (influence).
        The distance are computed using the coordinates of the neurons on the map.
        """

        # distance between each neuron of the map and the BMU
        x, y = np.indices(self.map_size)
        distance = np.linalg.norm([x - bmu[0], y - bmu[1]], axis=0)  # 2D array

        sigma = radius / 2

        # compute the Gaussian function on each neuron based on the distance 
        # this will help us determine the weights
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))

    def predict(self, data):
        """
        Find the BMU for each input vector then return the coordinates
        that represent the cluster of the input vetor.
        
        Parameters
        ----------
        data: 2D np array
            Input data. An array of shape (n_samples, input_size).
        """
        predictions = []

        for input_vector in data:
            bmu = self._find_bmu(input_vector)
            predictions.append(bmu)

        return predictions

    def quantization_error(self, data):
        error = 0.0
        for input_vector in data:
            bmu = self._find_bmu(input_vector)
            error += np.linalg.norm(input_vector - self.weights[bmu])
        return error / len(data)

    def visualize_som(self, data):
        # Get the weight vectors from the SOM object
        weights = self.weights

        # Predict the BMU for each input vector
        bmu_indices = self.predict(data)

        # Create a colormap with a unique color for each BMU
        colormap = {}
        for bmu_index in bmu_indices:
            if bmu_index not in colormap:
                colormap[bmu_index] = np.random.rand(3,)

        # Create a 2D grid of subplots for each neuron in the SOM
        fig, axs = plt.subplots(self.map_size[0], self.map_size[1], figsize=(10, 10))
        # Flatten the subplots array so we can loop through it
        axs = axs.ravel()

        # Loop through each neuron in the SOM and plot its weight vector as a point
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                # Get the weight vector for this neuron
                w = weights[i, j]
                # Get the BMU index for this neuron
                bmu_index = np.unravel_index(np.argmin(np.linalg.norm(w - self.weights, axis=2)), self.map_size)
                # Get the color for this BMU index
                color = colormap[bmu_index]
                # Plot the weight vector as a point on the corresponding subplot
                axs[i * self.map_size[1] + j].scatter(w[0], w[1], color=color)
                axs[i * self.map_size[1] + j].set_xticks([])
                axs[i * self.map_size[1] + j].set_yticks([])

        # Add a title to the figure
        fig.suptitle('SOM Visualization')
        # Show the plot
        plt.show()

