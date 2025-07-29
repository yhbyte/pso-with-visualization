import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import random
from dataclasses import dataclass
from typing import List, Tuple
import imageio
import os

@dataclass
class Drone:
    """Клас для представлення дрона"""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    id: int
    battery: float = 100.0  # Відсоток заряду батареї

class DroneSwarmPSO:
    """PSO алгоритм для оптимізації траєкторій рою дронів"""

    def __init__(self, n_drones=10, map_size=(100, 100), n_obstacles=15, n_fire_zones=5):
        # Параметри PSO
        self.w = 0.7        # Інерційна вага
        self.c1 = 1.5       # Когнітивний коефіцієнт
        self.c2 = 1.5       # Соціальний коефіцієнт

        # Параметри системи
        self.n_drones = n_drones
        self.map_size = map_size
        self.max_velocity = 5.0
        self.min_drone_distance = 5.0
        self.communication_range = 30.0
        self.battery_consumption_rate = 0.5  # % за одиницю відстані

        # Вагові коефіцієнти цільової функції
        self.alpha = 0.3  # Енергія
        self.beta = 0.3   # Час
        self.gamma = 0.2  # Безпека
        self.delta = 0.2  # Покриття

        # Ініціалізація середовища
        self._initialize_environment(n_obstacles, n_fire_zones)

        # Ініціалізація дронів
        self.drones = self._initialize_drones()

        # Глобальний найкращий результат
        self.global_best_position = None
        self.global_best_fitness = float('inf')

        # Історія для візуалізації
        self.history = []

    def _initialize_environment(self, n_obstacles, n_fire_zones):
        """Ініціалізація перешкод та зон пожеж"""
        # Генерація перешкод (будівлі, дерева)
        self.obstacles = []
        for _ in range(n_obstacles):
            x = random.uniform(10, self.map_size[0] - 10)
            y = random.uniform(10, self.map_size[1] - 10)
            width = random.uniform(5, 15)
            height = random.uniform(5, 15)
            self.obstacles.append((x, y, width, height))

        # Генерація зон пожеж (високий пріоритет для моніторингу)
        self.fire_zones = []
        for _ in range(n_fire_zones):
            x = random.uniform(20, self.map_size[0] - 20)
            y = random.uniform(20, self.map_size[1] - 20)
            radius = random.uniform(8, 15)
            self.fire_zones.append((x, y, radius))

    def _initialize_drones(self):
        """Ініціалізація початкових позицій дронів"""
        drones = []
        base_x, base_y = self.map_size[0] / 2, 5  # База внизу карти

        for i in range(self.n_drones):
            # Початкові позиції навколо бази
            angle = 2 * np.pi * i / self.n_drones
            x = base_x + 10 * np.cos(angle)
            y = base_y + 10 * np.sin(angle)

            position = np.array([x, y])
            velocity = np.random.uniform(-1, 1, 2)

            drone = Drone(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=float('inf'),
                id=i,
                battery=100.0
            )
            drones.append(drone)

        return drones

    def fitness_function(self, positions):
        """Обчислення цільової функції для набору позицій дронів"""
        fitness = 0

        # 1. Енергетичні витрати (відстань від бази)
        base = np.array([self.map_size[0] / 2, 5])
        energy_cost = 0
        for pos in positions:
            distance_from_base = np.linalg.norm(pos - base)
            energy_cost += distance_from_base
        fitness += self.alpha * energy_cost

        # 2. Покриття зон пожеж
        coverage_penalty = 0
        for fire_x, fire_y, radius in self.fire_zones:
            fire_center = np.array([fire_x, fire_y])
            min_distance = float('inf')
            for pos in positions:
                dist = np.linalg.norm(pos - fire_center)
                min_distance = min(min_distance, dist)
            # Штраф, якщо жоден дрон не покриває зону
            if min_distance > radius:
                coverage_penalty += min_distance - radius
        fitness += self.delta * coverage_penalty

        # 3. Безпека (відстань до перешкод)
        safety_penalty = 0
        for pos in positions:
            for obs_x, obs_y, width, height in self.obstacles:
                # Перевірка колізій з перешкодами
                if (obs_x <= pos[0] <= obs_x + width and
                    obs_y <= pos[1] <= obs_y + height):
                    safety_penalty += 100  # Великий штраф за колізію
                else:
                    # Штраф за близькість до перешкод
                    center = np.array([obs_x + width/2, obs_y + height/2])
                    dist = np.linalg.norm(pos - center)
                    if dist < 10:
                        safety_penalty += 10 - dist
        fitness += self.gamma * safety_penalty

        # 4. Відстань між дронами (уникнення зіткнень)
        collision_penalty = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.min_drone_distance:
                    collision_penalty += self.min_drone_distance - dist
        fitness += collision_penalty * 10  # Високий пріоритет

        return fitness

    def update_velocity(self, drone):
        """Оновлення швидкості дрона за формулою PSO"""
        r1, r2 = random.random(), random.random()

        cognitive = self.c1 * r1 * (drone.best_position - drone.position)
        social = self.c2 * r2 * (self.global_best_position - drone.position)

        drone.velocity = self.w * drone.velocity + cognitive + social

        # Обмеження швидкості
        speed = np.linalg.norm(drone.velocity)
        if speed > self.max_velocity:
            drone.velocity = drone.velocity / speed * self.max_velocity

    def update_position(self, drone):
        """Оновлення позиції дрона"""
        drone.position += drone.velocity

        # Обмеження позиції в межах карти
        drone.position[0] = np.clip(drone.position[0], 0, self.map_size[0])
        drone.position[1] = np.clip(drone.position[1], 0, self.map_size[1])

        # Оновлення заряду батареї
        distance = np.linalg.norm(drone.velocity)
        drone.battery -= distance * self.battery_consumption_rate
        drone.battery = max(0, drone.battery)

    def optimize(self, max_iterations=100):
        """Основний цикл оптимізації"""
        for iteration in range(max_iterations):
            # Збір поточних позицій
            positions = [drone.position for drone in self.drones]

            # Оновлення найкращих результатів
            for i, drone in enumerate(self.drones):
                fitness = self.fitness_function([drone.position])

                if fitness < drone.best_fitness:
                    drone.best_fitness = fitness
                    drone.best_position = drone.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = drone.position.copy()

            # Оновлення швидкостей та позицій
            for drone in self.drones:
                if drone.battery > 0:  # Дрон може рухатися тільки з зарядом
                    self.update_velocity(drone)
                    self.update_position(drone)

            # Зберігання історії для візуалізації
            self.history.append({
                'iteration': iteration,
                'positions': [d.position.copy() for d in self.drones],
                'batteries': [d.battery for d in self.drones],
                'fitness': self.global_best_fitness
            })

            # Адаптивне зменшення інерційної ваги
            self.w = 0.9 - (0.5 * iteration / max_iterations)

    def visualize_animation(self, filename='drone_swarm_optimization.gif'):
        """Створення анімації оптимізації"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Налаштування основної карти
        ax1.set_xlim(0, self.map_size[0])
        ax1.set_ylim(0, self.map_size[1])
        ax1.set_aspect('equal')
        ax1.set_title('Оптимізація траєкторій рою дронів', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (км)')
        ax1.set_ylabel('Y (км)')
        ax1.grid(True, alpha=0.3)

        # Відображення перешкод
        for x, y, w, h in self.obstacles:
            rect = Rectangle((x, y), w, h, facecolor='gray', edgecolor='black', alpha=0.7)
            ax1.add_patch(rect)

        # Відображення зон пожеж
        for x, y, r in self.fire_zones:
            circle = Circle((x, y), r, facecolor='red', alpha=0.3, edgecolor='darkred', linewidth=2)
            ax1.add_patch(circle)

        # База
        base_x, base_y = self.map_size[0] / 2, 5
        base = Circle((base_x, base_y), 3, facecolor='blue', edgecolor='darkblue', linewidth=2)
        ax1.add_patch(base)
        ax1.text(base_x, base_y - 5, 'БАЗА', ha='center', fontweight='bold')

        # Ініціалізація елементів дронів
        drone_points = ax1.scatter([], [], c='green', s=100, marker='^', edgecolor='black', linewidth=1.5)
        drone_trails = []
        drone_labels = []

        for i in range(self.n_drones):
            trail, = ax1.plot([], [], 'g-', alpha=0.3, linewidth=1)
            drone_trails.append(trail)
            label = ax1.text(0, 0, '', fontsize=8, ha='center')
            drone_labels.append(label)

        # Налаштування графіка фітнес-функції
        ax2.set_xlabel('Ітерація')
        ax2.set_ylabel('Значення цільової функції')
        ax2.set_title('Збіжність алгоритму PSO', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        fitness_line, = ax2.plot([], [], 'b-', linewidth=2)

        # Текст статистики
        stats_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                             verticalalignment='top', fontsize=10)

        def animate(frame):
            if frame >= len(self.history):
                return

            data = self.history[frame]
            positions = np.array(data['positions'])
            batteries = data['batteries']

            # Оновлення позицій дронів
            drone_points.set_offsets(positions)

            # Кольори дронів залежно від заряду батареї
            colors = ['green' if b > 50 else 'yellow' if b > 20 else 'red' for b in batteries]
            drone_points.set_color(colors)

            # Оновлення траєкторій
            for i, (trail, label) in enumerate(zip(drone_trails, drone_labels)):
                trail_data = [h['positions'][i] for h in self.history[:frame+1]]
                if trail_data:
                    trail_data = np.array(trail_data)
                    trail.set_data(trail_data[:, 0], trail_data[:, 1])

                # Оновлення міток дронів
                if frame < len(self.history):
                    pos = positions[i]
                    label.set_position((pos[0], pos[1] + 2))
                    label.set_text(f'D{i+1}\n{batteries[i]:.0f}%')

            # Оновлення графіка фітнес-функції
            iterations = [h['iteration'] for h in self.history[:frame+1]]
            fitness_values = [h['fitness'] for h in self.history[:frame+1]]
            fitness_line.set_data(iterations, fitness_values)
            ax2.relim()
            ax2.autoscale_view()

            # Оновлення статистики
            active_drones = sum(1 for b in batteries if b > 0)
            avg_battery = np.mean(batteries)
            stats_text.set_text(
                f'Ітерація: {data["iteration"]}\n'
                f'Активні дрони: {active_drones}/{self.n_drones}\n'
                f'Середній заряд: {avg_battery:.1f}%\n'
                f'Фітнес: {data["fitness"]:.2f}'
            )

        # Створення анімації
        anim = animation.FuncAnimation(fig, animate, frames=len(self.history),
                                     interval=100, blit=False, repeat=True)

        # Збереження як GIF
        writer = animation.PillowWriter(fps=10)
        anim.save(filename, writer=writer)
        plt.close()

        print(f"Анімація збережена як {filename}")

        # Створення фінального знімку
        self.plot_final_result()

    def plot_final_result(self):
        """Візуалізація фінального результату"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Налаштування карти
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        ax.set_aspect('equal')
        ax.set_title('Фінальне розташування дронів', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (км)')
        ax.set_ylabel('Y (км)')
        ax.grid(True, alpha=0.3)

        # Відображення елементів середовища
        for x, y, w, h in self.obstacles:
            rect = Rectangle((x, y), w, h, facecolor='gray', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)

        for x, y, r in self.fire_zones:
            circle = Circle((x, y), r, facecolor='red', alpha=0.3, edgecolor='darkred', linewidth=2)
            ax.add_patch(circle)

        # База
        base_x, base_y = self.map_size[0] / 2, 5
        base = Circle((base_x, base_y), 3, facecolor='blue', edgecolor='darkblue', linewidth=2)
        ax.add_patch(base)

        # Фінальні позиції дронів
        final_positions = self.history[-1]['positions']
        final_batteries = self.history[-1]['batteries']

        for i, (pos, battery) in enumerate(zip(final_positions, final_batteries)):
            color = 'green' if battery > 50 else 'yellow' if battery > 20 else 'red'
            ax.scatter(pos[0], pos[1], c=color, s=150, marker='^', edgecolor='black', linewidth=2)
            ax.text(pos[0], pos[1] + 2, f'D{i+1}', ha='center', fontweight='bold')

            # Радіус покриття
            coverage = Circle((pos[0], pos[1]), 10, fill=False, edgecolor=color, linestyle='--', alpha=0.5)
            ax.add_patch(coverage)

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Дрон (заряд > 50%)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', markersize=10, label='Дрон (заряд 20-50%)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Дрон (заряд < 20%)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Перешкода'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Зона пожежі', alpha=0.3),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='База')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig('final_drone_positions.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Фінальний результат збережено як final_drone_positions.png")

# Головна функція для запуску
def main():
    # Створення та налаштування системи
    print("Ініціалізація системи дронів...")
    swarm = DroneSwarmPSO(
        n_drones=12,           # Кількість дронів
        map_size=(100, 100),   # Розмір карти (км)
        n_obstacles=20,        # Кількість перешкод
        n_fire_zones=6         # Кількість зон пожеж
    )

    print("Запуск оптимізації...")
    swarm.optimize(max_iterations=150)

    print("Створення візуалізації...")
    swarm.visualize_animation('drone_swarm_optimization.gif')

    # Виведення результатів
    print(f"\nРезультати оптимізації:")
    print(f"Фінальне значення цільової функції: {swarm.global_best_fitness:.4f}")
    print(f"Кількість активних дронів: {sum(1 for d in swarm.drones if d.battery > 0)}/{swarm.n_drones}")
    print(f"Середній залишок заряду: {np.mean([d.battery for d in swarm.drones]):.1f}%")

if __name__ == "__main__":
    main()