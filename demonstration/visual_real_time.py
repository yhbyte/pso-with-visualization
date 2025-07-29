import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
from datetime import datetime
import seaborn as sns

# Налаштування для кращого вигляду графіків
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class ExperimentConfig:
    """Конфігурація експерименту для дослідження впливу викидів"""
    num_particles: int
    outlier_probability: float  # Ймовірність на кожну ітерацію
    outlier_type: str  # 'freeze', 'remove', 'random_move'
    max_iterations: int


class Particle:
    """Частинка рою з можливістю перетворення на викид"""

    def __init__(self, bounds: Tuple[float, float], dimensions: int = 2):
        self.dimensions = dimensions
        self.bounds = bounds
        # Гарантуємо випадкову ініціалізацію позиції
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.is_outlier = False
        self.outlier_type = None

    def update_velocity(self, global_best: np.ndarray, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """Стандартне оновлення швидкості PSO без обмежень"""
        if self.is_outlier and self.outlier_type == 'freeze':
            return

        r1, r2 = np.random.random(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        """Стандартне оновлення позиції PSO"""
        if self.is_outlier:
            if self.outlier_type == 'freeze':
                return
            elif self.outlier_type == 'remove':
                return
            elif self.outlier_type == 'random_move':
                # Випадковий рух з більшою амплітудою
                random_move = np.random.uniform(-1, 1, self.dimensions)
                self.position += random_move
            else:  # normal movement for other outlier types
                self.position += self.velocity
        else:
            self.position += self.velocity

        # Обмеження позиції межами
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def make_outlier(self, outlier_type: str):
        """Перетворення частинки на викид"""
        self.is_outlier = True
        self.outlier_type = outlier_type

        if outlier_type == 'random_move':
            # Переміщення в випадкову віддалену точку
            center = (self.bounds[0] + self.bounds[1]) / 2
            max_distance = (self.bounds[1] - self.bounds[0]) * 0.8
            random_direction = np.random.uniform(-1, 1, self.dimensions)
            random_direction = random_direction / np.linalg.norm(random_direction)
            self.position = center + random_direction * max_distance


class ObjectiveFunction:
    """Тестові функції оптимізації"""

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Функція Растрігіна - багатомодальна з багатьма локальними мінімумами"""
        A = 10
        n = len(x)
        return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Сферична функція - простий глобальний мінімум у (0,0)"""
        return np.sum(x ** 2)

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Функція Аклі"""
        a, b, c = 20, 0.2, 2 * np.pi
        n = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e


class PSO:
    """Простий стандартний алгоритм PSO з підтримкою викидів та візуалізацією"""

    def __init__(self, num_particles: int, bounds: Tuple[float, float],
                 objective_func, dimensions: int = 2, random_seed: int = None):

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.num_particles = num_particles
        self.bounds = bounds
        self.objective_func = objective_func
        self.dimensions = dimensions

        # Ініціалізація частинок
        self.particles = [Particle(bounds, dimensions) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')

        # Для збору статистики та візуалізації
        self.fitness_history = []
        self.outliers_count_history = []
        self.particle_positions_history = []  # Для візуалізації
        self.global_best_history = []

    def evaluate_fitness(self):
        """Оцінка фітнес-функції для всіх частинок"""
        active_particles = [p for p in self.particles if not (p.is_outlier and p.outlier_type == 'remove')]

        for particle in active_particles:
            fitness = self.objective_func(particle.position)

            # Оновлення персонального найкращого
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # Погані позиції викидів можуть "зіпсувати" глобальне рішення
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

    def introduce_outliers(self, probability: float, outlier_type: str):
        """Введення викидів з заданою ймовірністю"""
        new_outliers = 0
        for particle in self.particles:
            if not particle.is_outlier and np.random.random() < probability:
                particle.make_outlier(outlier_type)
                new_outliers += 1
        return new_outliers

    def step(self, iteration: int):
        """Один крок стандартного алгоритму PSO"""
        # Оцінка фітнесу
        self.evaluate_fitness()

        # Оновлення швидкостей та позицій для всіх активних частинок
        active_particles = [p for p in self.particles if not (p.is_outlier and p.outlier_type == 'remove')]

        if self.global_best_position is not None:
            for particle in active_particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

        # Збір статистики
        outliers_count = len([p for p in self.particles if p.is_outlier])

        self.fitness_history.append(self.global_best_fitness)
        self.outliers_count_history.append(outliers_count)

        # Збереження позицій для візуалізації
        positions = []
        for particle in self.particles:
            if not (particle.is_outlier and particle.outlier_type == 'remove'):
                positions.append({
                    'position': particle.position.copy(),
                    'is_outlier': particle.is_outlier,
                    'outlier_type': particle.outlier_type,
                    'fitness': self.objective_func(particle.position)
                })

        self.particle_positions_history.append(positions)
        if self.global_best_position is not None:
            self.global_best_history.append(self.global_best_position.copy())
        else:
            self.global_best_history.append(np.array([0, 0]))


class PSOVisualizer:
    """Візуалізація PSO з викидами"""

    def __init__(self, pso: PSO):
        self.pso = pso
        self.bounds = pso.bounds
        self.objective_func = pso.objective_func

    def create_function_contour(self, resolution: int = 100):
        """Створення контурної карти функції"""
        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.objective_func(np.array([X[i, j], Y[i, j]]))

        return X, Y, Z

    def plot_step(self, iteration: int, ax, X, Y, Z):
        """Малювання одного кроку оптимізації"""
        ax.clear()

        # Контурна карта функції
        contour = ax.contour(X, Y, Z, levels=20, alpha=0.6, colors='gray', linewidths=0.5)
        ax.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')

        if iteration < len(self.pso.particle_positions_history):
            positions_data = self.pso.particle_positions_history[iteration]

            # Розділення частинок за типами
            normal_particles = []
            freeze_outliers = []
            random_outliers = []

            for p_data in positions_data:
                if not p_data['is_outlier']:
                    normal_particles.append(p_data['position'])
                elif p_data['outlier_type'] == 'freeze':
                    freeze_outliers.append(p_data['position'])
                elif p_data['outlier_type'] == 'random_move':
                    random_outliers.append(p_data['position'])

            # Малювання частинок різними кольорами
            if normal_particles:
                normal_particles = np.array(normal_particles)
                ax.scatter(normal_particles[:, 0], normal_particles[:, 1],
                           c='blue', s=50, alpha=0.8, label='Звичайні частинки', marker='o')

            if freeze_outliers:
                freeze_outliers = np.array(freeze_outliers)
                ax.scatter(freeze_outliers[:, 0], freeze_outliers[:, 1],
                           c='red', s=80, alpha=0.9, label='Заморожені викиди', marker='s')

            if random_outliers:
                random_outliers = np.array(random_outliers)
                ax.scatter(random_outliers[:, 0], random_outliers[:, 1],
                           c='orange', s=80, alpha=0.9, label='Випадкові викиди', marker='^')

            # Глобальний найкращий
            if iteration < len(self.pso.global_best_history):
                global_best = self.pso.global_best_history[iteration]
                ax.scatter(global_best[0], global_best[1],
                           c='gold', s=150, marker='*', edgecolors='black', linewidth=2,
                           label='Глобальний найкращий', zorder=5)

        # Оптимум функції (для Растрігіна - (0,0))
        ax.scatter(0, 0, c='lime', s=100, marker='x', linewidth=3,
                   label='Глобальний оптимум', zorder=5)

        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(f'PSO з викидами - Ітерація {iteration}\n'
                     f'Фітнес: {self.pso.fitness_history[iteration]:.3f}, '
                     f'Викиди: {self.pso.outliers_count_history[iteration]}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def create_static_visualization(self, step_interval: int = 3):
        """Створення статичної візуалізації кроків"""
        if not self.pso.particle_positions_history:
            print("Немає даних для візуалізації. Запустіть спочатку оптимізацію.")
            return

        # Створення контурної карти
        X, Y, Z = self.create_function_contour()

        # Вибір кроків для відображення
        total_iterations = len(self.pso.particle_positions_history)
        selected_iterations = list(range(0, total_iterations, step_interval))

        # Розрахунок сітки підплотів
        n_plots = len(selected_iterations)
        cols = min(4, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, iteration in enumerate(selected_iterations):
            row = idx // cols
            col = idx % cols

            if rows == 1:
                ax = axes[col] if cols > 1 else axes[0]
            else:
                ax = axes[row, col]

            self.plot_step(iteration, ax, X, Y, Z)

        # Приховати зайві підплоти
        if n_plots < rows * cols:
            for idx in range(n_plots, rows * cols):
                row = idx // cols
                col = idx % cols
                if rows == 1:
                    ax = axes[col] if cols > 1 else axes[0]
                else:
                    ax = axes[row, col]
                ax.set_visible(False)

        plt.tight_layout()

        # Збереження
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'pso_visualization_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Візуалізація збережена: {filename}")
        plt.show()

    def create_animation(self, interval: int = 200, step_interval: int = 3):
        """Створення анімації PSO"""
        if not self.pso.particle_positions_history:
            print("Немає даних для анімації. Запустіть спочатку оптимізацію.")
            return

        # Створення контурної карти
        X, Y, Z = self.create_function_contour()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Вибір кроків для анімації
        selected_iterations = list(range(0, len(self.pso.particle_positions_history), step_interval))

        def animate(frame):
            iteration = selected_iterations[frame]
            self.plot_step(iteration, ax, X, Y, Z)

        anim = animation.FuncAnimation(fig, animate, frames=len(selected_iterations),
                                       interval=interval, repeat=True, blit=False)

        # Збереження анімації
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'pso_animation_{timestamp}.gif'
        try:
            anim.save(filename, writer='pillow', fps=3)
            print(f"Анімація збережена: {filename}")
        except Exception as e:
            print(f"Не вдалося зберегти анімацію: {e}")

        plt.show()
        return anim


def run_single_pso_experiment():
    """Запуск одного експерименту PSO з візуалізацією"""

    print("=== ВІЗУАЛІЗАЦІЯ PSO З ВИКИДАМИ ===\n")

    # Конфігурація експерименту
    config = ExperimentConfig(
        num_particles=15,
        outlier_probability=0.0036,  # Більша ймовірність для кращої демонстрації
        outlier_type='freeze',  # Можна змінити на 'freeze' або 'remove'
        max_iterations=60
    )

    # Параметри функції
    bounds = (-5.12, 5.12)
    objective_func = ObjectiveFunction.rastrigin

    print(f"Конфігурація:")
    print(f"- Кількість частинок: {config.num_particles}")
    print(f"- Ймовірність викидів: {config.outlier_probability * 100:.1f}% на ітерацію")
    print(f"- Тип викидів: {config.outlier_type}")
    print(f"- Максимум ітерацій: {config.max_iterations}")
    print(f"- Функція: Растрігіна")
    print(f"- Межі: {bounds}\n")

    # Створення та запуск PSO (без фіксованого seed для випадкової ініціалізації)
    # Для відтворюваних результатів замініть None на конкретне число, наприклад 42
    pso = PSO(config.num_particles, bounds, objective_func, random_seed=None)

    # Показати початкові позиції для підтвердження
    print("Початкові позиції частинок:")
    for i, particle in enumerate(pso.particles[:5]):  # Показати перші 5 частинок
        print(f"  Частинка {i + 1}: ({particle.position[0]:.2f}, {particle.position[1]:.2f})")
    if len(pso.particles) > 5:
        print(f"  ... та ще {len(pso.particles) - 5} частинок")
    print()

    print("Запуск оптимізації...")

    for iteration in range(config.max_iterations):
        # Введення викидів (починаючи з 10-ї ітерації)
        if iteration >= 10 and config.outlier_probability > 0:
            new_outliers = pso.introduce_outliers(config.outlier_probability, config.outlier_type)
            if new_outliers > 0:
                print(f"  Ітерація {iteration}: додано {new_outliers} викидів типу '{config.outlier_type}'")

        pso.step(iteration)

        # Виведення прогресу кожні 10 ітерацій
        if iteration % 10 == 0:
            outliers_count = pso.outliers_count_history[-1]
            fitness = pso.fitness_history[-1]
            print(f"  Ітерація {iteration}: фітнес = {fitness:.3f}, викидів = {outliers_count}")

    final_fitness = pso.fitness_history[-1]
    final_outliers = pso.outliers_count_history[-1]
    print(f"\nРезультат:")
    print(f"- Фінальний фітнес: {final_fitness:.3f}")
    print(f"- Максимальна кількість викидів: {max(pso.outliers_count_history)}")
    print(f"- Фінальна кількість викидів: {final_outliers}")

    # Створення візуалізації
    visualizer = PSOVisualizer(pso)

    print(f"\nСтворення візуалізації...")
    print("1. Статичні кадри кожні 3 кроки...")
    visualizer.create_static_visualization(step_interval=3)

    # Створення графіку сходимості
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(pso.fitness_history, 'b-', linewidth=2, label='Найкращий фітнес')
    plt.xlabel('Ітерація')
    plt.ylabel('Фітнес')
    plt.title('Сходимість алгоритму')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.plot(pso.outliers_count_history, 'r-', linewidth=2, label='Кількість викидів')
    plt.xlabel('Ітерація')
    plt.ylabel('Кількість викидів')
    plt.title('Еволюція кількості викидів')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    convergence_filename = f'pso_convergence_{timestamp}.png'
    plt.savefig(convergence_filename, dpi=300, bbox_inches='tight')
    print(f"Графік сходимості збережений: {convergence_filename}")
    plt.show()

    print(f"\n2. Створення анімації...")
    visualizer.create_animation(interval=300, step_interval=2)

    return pso, visualizer


def compare_outlier_types():
    """Порівняння різних типів викидів"""
    print("=== ПОРІВНЯННЯ ТИПІВ ВИКИДІВ ===\n")

    outlier_types = ['freeze', 'random_move', 'remove']
    bounds = (-5.12, 5.12)

    results = {}

    for outlier_type in outlier_types:
        print(f"Тестування викидів типу: {outlier_type}")

        config = ExperimentConfig(
            num_particles=15,
            outlier_probability=0.03,
            outlier_type=outlier_type,
            max_iterations=50
        )

        # Використовуємо однаковий seed для справедливого порівняння
        pso = PSO(config.num_particles, bounds, ObjectiveFunction.rastrigin) #, random_seed=42)

        for iteration in range(config.max_iterations):
            if iteration >= 10:
                pso.introduce_outliers(config.outlier_probability, config.outlier_type)
            pso.step(iteration)

        results[outlier_type] = {
            'final_fitness': pso.fitness_history[-1],
            'max_outliers': max(pso.outliers_count_history),
            'pso': pso
        }

        print(f"  Фінальний фітнес: {pso.fitness_history[-1]:.3f}")
        print(f"  Макс. викидів: {max(pso.outliers_count_history)}\n")

    # Візуалізація порівняння
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (outlier_type, result) in enumerate(results.items()):
        pso = result['pso']
        visualizer = PSOVisualizer(pso)
        X, Y, Z = visualizer.create_function_contour()

        # Початковий стан
        ax = axes[0, idx]
        visualizer.plot_step(0, ax, X, Y, Z)
        ax.set_title(f'{outlier_type.title()} - Початок')

        # Фінальний стан
        ax = axes[1, idx]
        final_iteration = len(pso.particle_positions_history) - 1
        visualizer.plot_step(final_iteration, ax, X, Y, Z)
        ax.set_title(f'{outlier_type.title()} - Кінець')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = f'outlier_types_comparison_{timestamp}.png'
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    print(f"Порівняння збережено: {comparison_filename}")
    plt.show()


if __name__ == "__main__":
    print("Виберіть режим:")
    print("1. Один експеримент з візуалізацією")
    print("2. Порівняння типів викидів")

    choice = input("Введіть номер (1 або 2): ").strip()

    if choice == "1":
        pso, visualizer = run_single_pso_experiment()
    elif choice == "2":
        compare_outlier_types()
    else:
        print("Запуск за замовчуванням...")
        pso, visualizer = run_single_pso_experiment()