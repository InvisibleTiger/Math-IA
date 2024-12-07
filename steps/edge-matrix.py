import numpy as np

gradient_matrix = np.loadtxt('gradient_matrix.txt')

threshold = 150

edge_matrix = np.where(gradient_matrix >= threshold, 1, 0)

np.savetxt('edge_matrix.txt', edge_matrix, fmt='%d')

print("Edge matrix created and saved to 'edge_matrix.txt'")
