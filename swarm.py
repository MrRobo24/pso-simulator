import numpy as np
from particle import Particle


class Swarm:
    def __init__(self, n_particles, bounds, fitness_fn, w=0.7, c1=1.5, c2=1.5):
        """
        n_particles: number of particles in the swarm
        bounds:      list of (min, max) per dimension
        fitness_fn:  function that scores a position (lower = better)
        w:           inertia weight — how much old velocity is retained
        c1:          cognitive coefficient — pull toward personal best
        c2:          social coefficient — pull toward global best
        """
        self.fitness_fn = fitness_fn
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = [Particle(bounds, fitness_fn) for _ in range(n_particles)]

        # Global best: the best position any particle has ever found
        best_particle = min(self.particles, key=lambda p: p.best_fitness)
        self.gbest_position = best_particle.best_position.copy()
        self.gbest_fitness = best_particle.best_fitness

    def step(self):
        """Advance the swarm by one iteration."""
        for particle in self.particles:
            self._update_velocity(particle)
            particle.update(self.fitness_fn)

            # If this particle found a new global best, record it
            if particle.best_fitness < self.gbest_fitness:
                self.gbest_fitness = particle.best_fitness
                self.gbest_position = particle.best_position.copy()

    def _update_velocity(self, particle):
        r1 = np.random.uniform(0, 1, size=particle.velocity.shape)
        r2 = np.random.uniform(0, 1, size=particle.velocity.shape)

        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social    = self.c2 * r2 * (self.gbest_position - particle.position)

        particle.velocity = self.w * particle.velocity + cognitive + social
