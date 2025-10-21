import numpy as np

def simulate_deuteranopia(frame):
    """
    Simulates how a person with Deuteranopia perceives colors.
    """
    matrix = np.array([
        [0.625, 0.7, -0.025],
        [0.7, 0.3, 0.0],
        [0.0, 0.3, 0.7]
    ])
    transformed = frame.dot(matrix.T)
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return transformed
