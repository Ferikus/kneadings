import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from normalization import *
from convert import decimal_to_binary, binary_to_decimal

# data = np.load(r'../kneadings_results.npz')
#
# kneadings_weighted_sum_set = data['results']
# sweep_size = data['sweep_size']
# a_start = data['a_start']
# a_end = data['a_end']
# b_start = data['b_start']
# b_end = data['b_end']

data_kneadings = np.load(r'../cuda_sweep/sweep_fbpo.npz')
data_inits = np.load(r'../system_analysis/inits.npz')

kneadings_weighted_sum_set = data_kneadings['kneadings']
a_count = data_inits['left_n'] + data_inits['right_n'] + 1
b_count = data_inits['up_n'] + data_inits['down_n'] + 1
a_start = -2.67 - data_inits['left_n'] * 0.01
a_end = -2.67 + data_inits['right_n'] * 0.01
b_start = -1.61268422884276 - data_inits['down_n'] * 0.01
b_end = -1.61268422884276 + data_inits['up_n'] * 0.01

colorMapLevels = 2**8
blue = np.linspace(0.01, 1.0, colorMapLevels)
red = 1 - blue
green = np.random.random(colorMapLevels) * 0.8
# green = np.linspace(0.8, 1.0, colorMapLevels)
RGB = np.column_stack((red, green, blue))
custom_cmap = ListedColormap(RGB)

# kneadings_norm = []
# for i in range(len(kneadings_weighted_sum_set)):
#     if kneadings_weighted_sum_set[i] in [-0.1, -0.2]:
#         kneadings_norm.append(kneadings_weighted_sum_set[i])
#         continue
#     kneading_bin = decimal_to_binary(kneadings_weighted_sum_set[i])
#     kneading_bin_norm = normalize_kneading(kneading_bin)
#     kneading_dec_norm = binary_to_decimal(kneading_bin_norm[0])
#     print(f"{kneadings_weighted_sum_set[i]} ({kneading_bin}) после нормализации: {kneading_dec_norm} ({kneading_bin_norm})")
#     kneadings_norm.append(kneading_dec_norm)
#     # kneadings_norm.append(1.0/len(kneading_bin_norm[1]))

plt.figure(figsize=(8, 8))
plt.imshow(
    np.reshape(kneadings_weighted_sum_set, (a_count, b_count), 'F'),
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
plt.savefig('mode_map_1.png', dpi=300, bbox_inches='tight')
plt.show()