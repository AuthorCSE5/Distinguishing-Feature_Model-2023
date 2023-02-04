import numpy as np

def generate_score(num_items):
        
    U = np.random.rand(num_items)
    U = U / np.linalg.norm(U)

    return U
