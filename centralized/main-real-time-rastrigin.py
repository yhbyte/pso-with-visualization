"""
Реалізація алгоритму рою частинок (PSO) для оптимізації функції Растригіна
з візуалізацією в реальному часі.

Автор: [Ваше ім'я]
Дата: 2025-06-24
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from typing import Tuple, List
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Налаштування matplotlib для інтерактивного режиму
plt.ion()


class Particle:
    """Клас, що представляє окрему частинку в рої."""

    def __init__(self, bounds: List[Tuple[float, float]], dim: int):
        """
        Ініціалізація частинки.

        Args:
            bounds: Межі простору пошуку для кожного виміру
            dim: Розмірність простору пошуку
        """
        self.position = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            dim
        )
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')


class PSO:
    """Клас реалізації алгоритму рою частинок."""

    def __init__(
        self,
        objective_func,
        bounds: List[Tuple[float, float]],
        n_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5
    ):
        """
        Ініціалізація алгоритму PSO.

        Args:
            objective_func: Цільова функція для оптимізації
            bounds: Межі простору пошуку
            n_particles: Кількість частинок у рої
            max_iter: Максимальна кількість ітерацій
            w: Інерційна вага
            c1: Когнітивний коефіцієнт
            c2: Соціальний коефіцієнт
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.dim = len(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Ініціалізація рою
        self.swarm = [Particle(bounds, self.dim) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')

        # Історія для візуалізації
        self.history = []

        logger.info(f"Ініціалізовано PSO з {n_particles} частинками")

    def evaluate_fitness(self, particle: Particle) -> float:
        """Обчислення значення цільової функції для частинки."""
        return self.objective_func(particle.position)

    def update_velocity(self, particle: Particle) -> None:
        """Оновлення швидкості частинки згідно з формулою PSO."""
        r1, r2 = np.random.rand(2)

        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (self.global_best_position - particle.position)

        particle.velocity = self.w * particle.velocity + cognitive + social

    def update_position(self, particle: Particle) -> None:
        """Оновлення позиції частинки з урахуванням меж."""
        particle.position = particle.position + particle.velocity

        # Обмеження позиції межами простору пошуку
        for i in range(self.dim):
            particle.position[i] = np.clip(
                particle.position[i],
                self.bounds[i][0],
                self.bounds[i][1]
            )

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Виконання оптимізації.

        Returns:
            Кортеж з найкращою знайденою позицією та її значенням
        """
        for iteration in range(self.max_iter):
            positions = []

            for particle in self.swarm:
                # Обчислення придатності
                particle.fitness = self.evaluate_fitness(particle)

                # Оновлення локального найкращого
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()

                # Оновлення глобального найкращого
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()

                positions.append(particle.position.copy())

            # Збереження історії
            self.history.append({
                'iteration': iteration,
                'positions': np.array(positions),
                'global_best': self.global_best_position.copy(),
                'global_best_fitness': self.global_best_fitness
            })

            # Оновлення швидкостей та позицій
            for particle in self.swarm:
                self.update_velocity(particle)
                self.update_position(particle)

            if iteration % 10 == 0:
                logger.info(
                    f"Ітерація {iteration}: "
                    f"найкраще значення = {self.global_best_fitness:.6f}"
                )

        return self.global_best_position, self.global_best_fitness


def rastrigin(x: np.ndarray) -> float:
    """
    Функція Растригіна.

    f(x) = 10n + Σ[xi² - 10cos(2πxi)]

    Args:
        x: Вектор вхідних значень

    Returns:
        Значення функції
    """
    n = len(x)
    A = 10
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


class PSOVisualizer:
    """Клас для візуалізації роботи алгоритму PSO."""

    def __init__(self, pso: PSO, func, bounds: List[Tuple[float, float]]):
        """
        Ініціалізація візуалізатора.

        Args:
            pso: Об'єкт алгоритму PSO
            func: Цільова функція
            bounds: Межі простору пошуку
        """
        self.pso = pso
        self.func = func
        self.bounds = bounds

        # Створення сітки для візуалізації
        x = np.linspace(bounds[0][0], bounds[0][1], 100)
        y = np.linspace(bounds[1][0], bounds[1][1], 100)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.zeros_like(self.X)

        for i in range(len(x)):
            for j in range(len(y)):
                self.Z[j, i] = func(np.array([self.X[j, i], self.Y[j, i]]))

    def animate(self):
        """Створення анімації оптимізації."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Налаштування першого графіка (контурний)
        # Використовуємо filled contour для кращої візуалізації
        contourf = ax1.contourf(self.X, self.Y, self.Z, levels=30, cmap='viridis', alpha=0.7)
        contour = ax1.contour(self.X, self.Y, self.Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)

        # Ініціалізація scatter plots
        particles_scatter = ax1.scatter([], [], c='red', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        best_scatter = ax1.scatter([], [], c='yellow', s=300, marker='*', edgecolors='black', linewidth=2)

        ax1.set_xlabel('x₁', fontsize=12)
        ax1.set_ylabel('x₂', fontsize=12)
        ax1.set_title('Простір пошуку функції Растригіна', fontsize=14)
        ax1.set_xlim(self.bounds[0])
        ax1.set_ylim(self.bounds[1])
        ax1.grid(True, alpha=0.3)

        # Додавання colorbar
        cbar = plt.colorbar(contourf, ax=ax1)
        cbar.set_label('f(x₁, x₂)', fontsize=10)

        # Налаштування другого графіка (збіжність)
        line_best, = ax2.plot([], [], 'b-', linewidth=2, label='Найкраще значення')
        ax2.set_xlabel('Ітерація', fontsize=12)
        ax2.set_ylabel('Значення функції', fontsize=12)
        ax2.set_title('Збіжність алгоритму PSO', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Текст для відображення поточної ітерації
        iteration_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                                 verticalalignment='top', fontsize=12,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        def init():
            """Ініціалізація анімації."""
            particles_scatter.set_offsets(np.empty((0, 2)))
            best_scatter.set_offsets(np.empty((0, 2)))
            line_best.set_data([], [])
            iteration_text.set_text('')
            return particles_scatter, best_scatter, line_best, iteration_text

        def update(frame):
            """Оновлення кадру анімації."""
            if frame < len(self.pso.history):
                data = self.pso.history[frame]

                # Оновлення позицій частинок
                positions = data['positions']
                particles_scatter.set_offsets(positions)

                # Оновлення найкращої позиції
                best_pos = data['global_best'].reshape(1, -1)
                best_scatter.set_offsets(best_pos)

                # Оновлення графіка збіжності
                iterations = np.arange(frame + 1)
                best_values = [h['global_best_fitness'] for h in self.pso.history[:frame+1]]
                line_best.set_data(iterations, best_values)

                # Автомасштабування осей для графіка збіжності
                if iterations.size > 0:
                    ax2.set_xlim(0, self.pso.max_iter)
                    if best_values:
                        y_margin = max(best_values) * 0.1
                        ax2.set_ylim(min(best_values) - y_margin, max(best_values) + y_margin)

                # Оновлення тексту
                iteration_text.set_text(
                    f'Ітерація: {frame}\n'
                    f'Найкраще: {data["global_best_fitness"]:.4f}\n'
                    f'Позиція: [{data["global_best"][0]:.3f}, {data["global_best"][1]:.3f}]'
                )

            return particles_scatter, best_scatter, line_best, iteration_text

        # Створення анімації
        anim = FuncAnimation(
            fig, update, init_func=init,
            frames=len(self.pso.history),
            interval=200,  # Збільшено інтервал для кращого сприйняття
            blit=True,
            repeat=True
        )

        plt.tight_layout()
        return fig, anim


def main():
    """Головна функція для демонстрації роботи алгоритму."""
    # Параметри задачі
    dim = 2  # Двовимірний простір для візуалізації
    bounds = [(-5.12, 5.12)] * dim  # Стандартні межі для функції Растригіна

    # Параметри алгоритму
    n_particles = 30
    max_iter = 50

    # Створення та запуск оптимізації
    logger.info("Початок оптимізації функції Растригіна")
    pso = PSO(
        objective_func=rastrigin,
        bounds=bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        w=0.7,
        c1=1.5,
        c2=1.5
    )

    best_position, best_fitness = pso.optimize()

    logger.info(f"\nРезультати оптимізації:")
    logger.info(f"Найкраща позиція: {best_position}")
    logger.info(f"Найкраще значення функції: {best_fitness:.6f}")
    logger.info(f"Теоретичний мінімум: 0.0 at (0, 0)")

    # Візуалізація
    visualizer = PSOVisualizer(pso, rastrigin, bounds)
    fig, anim = visualizer.animate()

    # Збереження анімації (опціонально)
    anim.save('pso_optimization.gif', writer='pillow', fps=10)

    plt.show()


if __name__ == "__main__":
    main()