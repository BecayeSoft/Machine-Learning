from som import SOM
import numpy as np
import math

# Generate random data
data = np.random.rand(100, 3)
n_features = data.shape[1]

map_area = 5 * math.sqrt(data.shape[0])
map_height = map_width = math.ceil(math.sqrt(map_area))
map_size = (map_height, map_width)

som = SOM(input_size=n_features, map_size=map_size)
som.train(data, n_iter=100, lr=0.1)
predictions = som.predict()
som.visualize_som(data)
