import numpy as np

def generate_weight(num_items, dim, std_deviation):
        
    w = np.random.normal(0, scale = std_deviation, size = (dim,1))
    w = w / np.linalg.norm(w)

    return w


