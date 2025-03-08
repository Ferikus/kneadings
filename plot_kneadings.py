import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


data = np.load('kneadings_results.npz')

kneadings_weighted_sum_set = data['results']
sweep_size = data['sweep_size']
a_start = data['a_start']
a_end = data['a_end']
b_start = data['b_start']
b_end = data['b_end']

colorMapLevels = 2**8
blue = np.linspace(0.01, 1, colorMapLevels)
red = 1 - blue
# green = np.random.random(colorMapLevels) * 0.8
green = np.linspace(0.2, 0.8, colorMapLevels)
RGB = np.column_stack((red, green, blue))
custom_cmap = ListedColormap(RGB)


plt.figure(figsize=(8, 8))

plt.imshow(
    np.reshape(kneadings_weighted_sum_set, (sweep_size, sweep_size), 'F'),
    extent=[a_start, a_end, b_start, b_end],
    cmap=custom_cmap,
    # norm=Normalize(0, 100000),
    vmin=-0.1,
    vmax=1,
    origin='lower',
    aspect='auto'
)

plt.xlabel('Параметр a')
plt.ylabel('Параметр b')
plt.title('Карта режимов')
plt.show()
