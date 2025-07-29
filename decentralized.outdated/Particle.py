import numpy as np
from typing import Dict, Set, Tuple

"""
Автономна частинка в децентралізованому PSO.
Кожна частинка знає тільки про себе та своїх сусідів.
"""
class Particle:

    def __init__(self, particle_id: int, dimensions: int, bounds: Tuple[float, float]):
        self.id = particle_id
        self.dimensions = dimensions
        self.bounds = bounds

        # PSO параметри
        self.w = 0.7  # інерція
        self.c1 = 1.5  # когнітивний коефіцієнт
        self.c2 = 1.5  # соціальний коефіцієнт

        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)

        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('inf')
        self.current_fitness = float('inf')

        # Знання про сусідів
        self.neighbors: Set[int] = set()
        self.neighbor_best_positions: Dict[int, np.ndarray] = {}
        self.neighbor_best_fitness: Dict[int, float] = {}

        # Локальні знання про найкращу позицію серед сусідів
        self.local_best_position = self.position.copy()
        self.local_best_fitness = float('inf')

        # Історія для аналізу
        self.fitness_history = []
        self.position_history = []

    def add_neighbor(self, neighbor_id: int):
        self.neighbors.add(neighbor_id)

    def remove_neighbor(self, neighbor_id: int):
        self.neighbors.discard(neighbor_id)
        self.neighbor_best_positions.pop(neighbor_id, None)
        self.neighbor_best_fitness.pop(neighbor_id, None)

    def evaluate_fitness(self, fitness_function):
        self.current_fitness = fitness_function(self.position)

        # Оновити особистий best
        if self.current_fitness < self.personal_best_fitness:
            self.personal_best_fitness = self.current_fitness
            self.personal_best_position = self.position.copy()

        # Зберегти історію
        self.fitness_history.append(self.current_fitness)
        self.position_history.append(self.position.copy())

    def share_information(self) -> Dict:
        return {
            'id': self.id,
            'position': self.position.copy(),
            'fitness': self.current_fitness,
            'personal_best_position': self.personal_best_position.copy(),
            'personal_best_fitness': self.personal_best_fitness
        }

    def receive_neighbor_info(self, neighbor_info: Dict):
        neighbor_id = neighbor_info['id']

        # Зберегти інформацію про сусіда
        self.neighbor_best_positions[neighbor_id] = neighbor_info['personal_best_position']
        self.neighbor_best_fitness[neighbor_id] = neighbor_info['personal_best_fitness']

        self.update_local_best()

    def update_local_best(self):
        # Почати з власного best
        best_fitness = self.personal_best_fitness
        best_position = self.personal_best_position.copy()

        # Порівняти з сусідами
        for neighbor_id in self.neighbors:
            if neighbor_id in self.neighbor_best_fitness:
                neighbor_fitness = self.neighbor_best_fitness[neighbor_id]
                if neighbor_fitness < best_fitness:
                    best_fitness = neighbor_fitness
                    best_position = self.neighbor_best_positions[neighbor_id].copy()

        self.local_best_fitness = best_fitness
        self.local_best_position = best_position

    def update_pso_vector(self):
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)

        # PSO формула швидкості
        cognitive_component = self.c1 * r1 * (self.personal_best_position - self.position)
        social_component = self.c2 * r2 * (self.local_best_position - self.position)

        self.velocity = (self.w * self.velocity +
                         cognitive_component +
                         social_component)

        # Обмеження швидкості
        max_velocity = 0.2 * (self.bounds[1] - self.bounds[0])
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

        # Оновлення позиції
        self.position += self.velocity

        # Обмеження позиції межами
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def get_status(self) -> Dict:
        return {
            'id': self.id,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'current_fitness': self.current_fitness,
            'personal_best_fitness': self.personal_best_fitness,
            'local_best_fitness': self.local_best_fitness,
            'num_neighbors': len(self.neighbors),
            'neighbors': list(self.neighbors)
        }