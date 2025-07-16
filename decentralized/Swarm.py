import random
from typing import Dict, Tuple
from Particle import Particle

class Swarm:

    def __init__(self, num_particles: int, dimensions: int, bounds: Tuple[float, float]):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds

        # Створити частинки
        self.particles = [
            Particle(i, dimensions, bounds)
            for i in range(num_particles)
        ]

        # Налаштувати топологію (за замовчуванням - кільце)
        self.setup_ring_topology()

        # Глобальна статистика (тільки для аналізу)
        self.global_best_fitness = float('inf')
        self.global_best_position = None
        self.iteration_count = 0

        # Історія для візуалізації
        self.convergence_history = []

    def setup_ring_topology(self):
        for i in range(self.num_particles):
            left_neighbor = (i - 1) % self.num_particles
            right_neighbor = (i + 1) % self.num_particles

            self.particles[i].add_neighbor(left_neighbor)
            self.particles[i].add_neighbor(right_neighbor)

    def setup_star_topology(self, hub_particle: int = 0):
        # Очистити існуючі зв'язки
        for particle in self.particles:
            particle.neighbors.clear()

        # Hub зв'язаний з усіма
        for i in range(self.num_particles):
            if i != hub_particle:
                self.particles[hub_particle].add_neighbor(i)
                self.particles[i].add_neighbor(hub_particle)

    def setup_random_topology(self, connections_per_particle: int = 3):
        # Очистити існуючі зв'язки
        for particle in self.particles:
            particle.neighbors.clear()

        for i in range(self.num_particles):
            # Вибрати випадкових сусідів
            possible_neighbors = list(range(self.num_particles))
            possible_neighbors.remove(i)

            num_connections = min(connections_per_particle, len(possible_neighbors))
            neighbors = random.sample(possible_neighbors, num_connections)

            for neighbor in neighbors:
                self.particles[i].add_neighbor(neighbor)
                self.particles[neighbor].add_neighbor(i)  # Двосторонній зв'язок

    def perform_iteration(self, fitness_function):
        self.iteration_count += 1

        # 1. Кожна частинка оцінює свій fitness
        for particle in self.particles:
            particle.evaluate_fitness(fitness_function)

        # 2. Частинки обмінюються інформацією з сусідами
        self.exchange_information()

        # 3. Кожна частинка оновлює свою швидкість та позицію
        for particle in self.particles:
            particle.update_pso_vector()

        # 4. Оновити глобальну статистику (тільки для аналізу)
        self.update_global_statistics()

    def exchange_information(self):
        # Зібрати інформацію від всіх частинок
        all_info = {}
        for particle in self.particles:
            all_info[particle.id] = particle.share_information()

        # Кожна частинка отримує інформацію тільки від своїх сусідів
        for particle in self.particles:
            for neighbor_id in particle.neighbors:
                if neighbor_id in all_info:
                    particle.receive_neighbor_info(all_info[neighbor_id])

    def update_global_statistics(self):
        current_best_fitness = float('inf')
        current_best_position = None

        for particle in self.particles:
            if particle.personal_best_fitness < current_best_fitness:
                current_best_fitness = particle.personal_best_fitness
                current_best_position = particle.personal_best_position.copy()

        if current_best_fitness < self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best_position = current_best_position

        self.convergence_history.append(self.global_best_fitness)

    def get_swarm_status(self) -> Dict:
        particles_status = [particle.get_status() for particle in self.particles]

        return {
            'iteration': self.iteration_count,
            'global_best_fitness': self.global_best_fitness,
            'global_best_position': self.global_best_position,
            'particles': particles_status,
            'convergence_history': self.convergence_history.copy()
        }