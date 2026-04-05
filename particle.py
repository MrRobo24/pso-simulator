import numpy as np


class Particle:
    def __init__(self, bounds, fitness_fn):
        """
        bounds: list of (min, max) tuples for each dimension
                e.g. [(-10, 10), (-10, 10)] for a 2D space
        """
        dimensions = len(bounds)

        # Position: random point inside the search space
        self.position = np.array([
            np.random.uniform(low, high)
            for low, high in bounds
        ])

        # Velocity: random small step in any direction
        self.velocity = np.array([
            np.random.uniform(-(high - low), (high - low))
            for low, high in bounds
        ])

        # Memory: best position this particle has personally visited
        self.best_position = self.position.copy()
        self.best_fitness = fitness_fn(self.position)

    def update(self, fitness_fn):
        """Move the particle one step, then update personal best if improved."""
        self.position = self.position + self.velocity

        current_fitness = fitness_fn(self.position)
        if current_fitness < self.best_fitness:
            self.best_fitness = current_fitness
            self.best_position = self.position.copy()
