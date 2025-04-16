import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2

from normalization import *
from convert import decimal_to_binary, binary_to_decimal

data = np.load(r'../kneadings_results.npz')

kneadings_weighted_sum_set = data['results']
sweep_size = data['sweep_size']
a_start = data['a_start']
a_end = data['a_end']
b_start = data['b_start']
b_end = data['b_end']

colorMapLevels = 2**8
blue = np.linspace(0.01, 1.0, colorMapLevels)
red = 1 - blue
green = np.random.random(colorMapLevels) * 0.8
# green = np.linspace(0.8, 1.0, colorMapLevels)
RGB = np.column_stack((red, green, blue))
custom_cmap = ListedColormap(RGB)

kneadings_norm = []
for i in range(len(kneadings_weighted_sum_set)):
    if kneadings_weighted_sum_set[i] in [-0.1, -0.2]:
        kneadings_norm.append(kneadings_weighted_sum_set[i])
        continue
    kneading_bin = decimal_to_binary(kneadings_weighted_sum_set[i])
    kneading_bin_norm = normalize_kneading(kneading_bin)
    kneading_dec_norm = binary_to_decimal(kneading_bin_norm[0])
    print(f"{kneadings_weighted_sum_set[i]} ({kneading_bin}) после нормализации: {kneading_dec_norm} ({kneading_bin_norm})")
    kneadings_norm.append(kneading_dec_norm)
    # kneadings_norm.append(1.0/len(kneading_bin_norm[1]))

plt.figure(figsize=(8, 8))
plt.imshow(
    np.reshape(kneadings_norm, (sweep_size, sweep_size), 'F'),
    extent=[a_start, a_end, b_start, b_end],
    cmap=custom_cmap,
    vmin=-0.1,
    vmax=1,
    origin='lower',
    aspect='auto'
)
plt.xlabel('Параметр a')
plt.ylabel('Параметр b')
plt.title('Карта режимов')
plt.savefig('mode_map.png', dpi=300, bbox_inches='tight')
plt.show()

# cv2.imwrite('output.png', high_contrast)