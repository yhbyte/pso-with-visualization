import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from statistics import mean, stdev
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
    num_runs: int = 10000  # Кількість прогонів для статистики


class Particle:
    """Частинка рою з можливістю перетворення на викид"""

    def __init__(self, bounds: Tuple[float, float], dimensions: int = 2):
        self.dimensions = dimensions
        self.bounds = bounds
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.is_outlier = False
        self.outlier_type = None

    def update_velocity(self, global_best: np.ndarray, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """Оновлення швидкості частинки згідно з класичним PSO"""
        if self.is_outlier and self.outlier_type == 'freeze':
            return

        r1, r2 = np.random.random(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

        # Обмеження швидкості
        max_velocity = (self.bounds[1] - self.bounds[0]) * 0.2
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

    def update_position(self):
        """Оновлення позиції частинки"""
        if self.is_outlier:
            if self.outlier_type == 'freeze':
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
            # Переміщення в випадкову ДУЖЕ віддалену точку
            range_size = self.bounds[1] - self.bounds[0]
            # Переміщуємо на край або за межі області пошуку
            extreme_position = np.random.choice([self.bounds[0] - range_size * 0.2,
                                                 self.bounds[1] + range_size * 0.2],
                                                size=self.dimensions)
            self.position = extreme_position
            # Зробити швидкість дуже великою і хаотичною
            self.velocity = np.random.uniform(-range_size, range_size, self.dimensions)


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
    def rosenbrock(x: np.ndarray) -> float:
        """Функція Розенброка - має форму банана з глобальним мінімумом у (1,1)"""
        if len(x) < 2:
            return np.sum(x ** 2)
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Функція Аклі - дуже складна з глибоким глобальним мінімумом в центрі
        Особливо чутлива до викидів через свою структуру"""
        a, b, c = 20, 0.2, 2 * np.pi
        n = len(x)

        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(c * x))

        return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e

    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Функція Грівенка - має глобальну квадратичну структуру з локальними осциляціями
        Викиди можуть серйозно заплутати алгоритм через подвійну структуру"""
        sum_term = np.sum(x ** 2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_term - prod_term + 1

    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """Функція Швефеля - найобманливіша функція!
        Глобальний мінімум далеко від центру, викиди можуть повністю збити з пантелику"""
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    @staticmethod
    def levy(x: np.ndarray) -> float:
        """Функція Леві - дуже шорстка поверхня з багатьма локальними мінімумами
        Викиди створюють хаос у навігації"""
        n = len(x)
        w = 1 + (x - 1) / 4

        term1 = np.sin(np.pi * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

        sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))

        return term1 + sum_term + term3

    @staticmethod
    def composite_function(x: np.ndarray) -> float:
        """Композитна функція - комбінація кількох складних функцій
        Створює найскладніший ландшафт, де викиди максимально шкідливі"""
        # Нормалізуємо вхід для різних функцій
        rastrigin_part = ObjectiveFunction.rastrigin(x * 0.5) * 0.3
        ackley_part = ObjectiveFunction.ackley(x * 0.8) * 0.4
        griewank_part = ObjectiveFunction.griewank(x * 2.0) * 0.3

        # Додаємо "перешкоди" - локальні атрактори
        obstacles = 0
        obstacle_centers = [np.array([2.0, 2.0]), np.array([-2.0, -2.0]),
                            np.array([2.0, -2.0]), np.array([-2.0, 2.0])]

        for center in obstacle_centers:
            if len(x) >= len(center):
                dist = np.linalg.norm(x[:len(center)] - center)
                obstacles += 50 * np.exp(-dist ** 2 / 0.5)

        return rastrigin_part + ackley_part + griewank_part + obstacles

    @staticmethod
    def get_function_info(func_name: str) -> dict:
        """Отримати інформацію про функцію"""
        info = {
            'rastrigin': {
                'bounds': (-5.12, 5.12),
                'optimum': 0.0,
                'success_threshold': 80,
                'difficulty': 'medium',
                'description': 'Багато локальних мінімумів'
            },
            'ackley': {
                'bounds': (-32.768, 32.768),
                'optimum': 0.0,
                'success_threshold': 5,
                'difficulty': 'hard',
                'description': 'Глибока яма в центрі з багатьма пастками'
            },
            'griewank': {
                'bounds': (-600, 600),
                'optimum': 0.0,
                'success_threshold': 10,
                'difficulty': 'hard',
                'description': 'Подвійна структура: глобальна + локальні осциляції'
            },
            'schwefel': {
                'bounds': (-500, 500),
                'optimum': 0.0,
                'success_threshold': 1000,
                'difficulty': 'very_hard',
                'description': 'Найобманливіша - оптимум далеко від центру'
            },
            'levy': {
                'bounds': (-10, 10),
                'optimum': 0.0,
                'success_threshold': 20,
                'difficulty': 'hard',
                'description': 'Дуже шорстка поверхня'
            },
            'composite_function': {
                'bounds': (-5, 5),
                'optimum': 0.0,
                'success_threshold': 100,
                'difficulty': 'extreme',
                'description': 'Композитна функція з перешкодами'
            }
        }
        return info.get(func_name, info['rastrigin'])


class PSO:
    """Реалізація алгоритму Particle Swarm Optimization з підтримкою викидів"""

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
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.global_best_fitness = float('inf')

        # Для збору статистики
        self.fitness_history = []
        self.diversity_history = []
        self.active_particles_history = []
        self.outliers_count_history = []

    def evaluate_fitness(self):
        """Оцінка фітнес-функції для всіх частинок"""
        active_particles = [p for p in self.particles if not (p.is_outlier and p.outlier_type == 'remove')]

        for particle in active_particles:
            fitness = self.objective_func(particle.position)

            # Оновлення персонального найкращого
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # Оновлення глобального найкращого (тільки для не-викидів)
            if not particle.is_outlier and fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

    def calculate_diversity(self) -> float:
        """Розрахунок різноманітності рою (середня відстань від центру)"""
        active_particles = [p for p in self.particles if not (p.is_outlier and p.outlier_type == 'remove')]
        if len(active_particles) < 2:
            return 0.0

        positions = np.array([p.position for p in active_particles])
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions]
        return np.mean(distances)

    def introduce_outliers(self, probability: float, outlier_type: str):
        """Введення викидів з заданою ймовірністю"""
        for particle in self.particles:
            if not particle.is_outlier and np.random.random() < probability:
                particle.make_outlier(outlier_type)

    def step(self):
        """Один крок оптимізації"""
        # Оцінка фітнесу
        self.evaluate_fitness()

        # Оновлення швидкостей та позицій
        active_particles = [p for p in self.particles if not (p.is_outlier and p.outlier_type == 'remove')]

        for particle in active_particles:
            particle.update_velocity(self.global_best_position)
            particle.update_position()

        # Збір статистики
        outliers_count = len([p for p in self.particles if p.is_outlier])

        self.fitness_history.append(self.global_best_fitness)
        self.diversity_history.append(self.calculate_diversity())
        self.active_particles_history.append(len(active_particles))
        self.outliers_count_history.append(outliers_count)


def run_single_experiment(args):
    """Функція для запуску одного експерименту (для паралелізації)"""
    config, objective_func, bounds, run_id = args

    # PSO з викидами
    pso = PSO(config.num_particles, bounds, objective_func, random_seed=run_id)

    for iteration in range(config.max_iterations):
        if config.outlier_probability > 0 and iteration > 10:
            pso.introduce_outliers(config.outlier_probability, config.outlier_type)
        pso.step()

    return {
        'final_fitness': pso.global_best_fitness,
        'convergence_rate': len([f for f in pso.fitness_history if f > pso.global_best_fitness * 1.1]),
        'final_diversity': pso.diversity_history[-1] if pso.diversity_history else 0,
        'avg_active_particles': np.mean(pso.active_particles_history),
        'max_outliers': max(pso.outliers_count_history) if pso.outliers_count_history else 0,
        'fitness_history': pso.fitness_history
    }


class StatisticalExperimentRunner:
    """Виконання статистично значущих експериментів"""

    def __init__(self, num_processes: int = None):
        self.num_processes = num_processes or mp.cpu_count() - 1

    def run_statistical_experiment(self, config: ExperimentConfig, objective_func,
                                   bounds: Tuple[float, float] = (-5, 5),
                                   success_threshold: float = 80) -> Dict:
        """Виконання статистичного експерименту з багатьма прогонами"""

        print(f"Запуск {config.num_runs} прогонів для конфігурації: "
              f"{config.num_particles} частинок, викиди: {config.outlier_type}")
        print(f"  Очікувана ймовірність викидів: {config.outlier_probability * 100:.3f}% на ітерацію")

        # Розрахунок очікуваної кількості викидів
        iterations_with_outliers = config.max_iterations - 5  # Викиди з 6-ої ітерації
        prob_become_outlier = 1 - (1 - config.outlier_probability) ** iterations_with_outliers
        expected_outliers = config.num_particles * prob_become_outlier
        print(f"  Очікувана кількість викидів за експеримент: {expected_outliers:.1f}")

        # Контрольна група (без викидів)
        control_config = ExperimentConfig(
            config.num_particles, 0.0, 'none',
            config.max_iterations, config.num_runs
        )

        # Паралельний запуск контрольних експериментів
        control_args = [(control_config, objective_func, bounds, i)
                        for i in range(config.num_runs)]

        print("  Виконання контрольних експериментів (без викидів)...")
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            control_results = list(executor.map(run_single_experiment, control_args))

        # Паралельний запуск експериментів з викидами
        if config.outlier_probability > 0:
            print("  Виконання експериментів з викидами...")
            outlier_args = [(config, objective_func, bounds, i)
                            for i in range(config.num_runs)]

            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                outlier_results = list(executor.map(run_single_experiment, outlier_args))
        else:
            outlier_results = control_results

        return {
            'control': self._aggregate_results(control_results, success_threshold),
            'outliers': self._aggregate_results(outlier_results, success_threshold),
            'config': config,
            'raw_control': control_results,
            'raw_outliers': outlier_results
        }

    def _aggregate_results(self, results: List[Dict], success_threshold: float = 80) -> Dict:
        """Агрегація результатів з багатьох прогонів"""
        final_fitness = [r['final_fitness'] for r in results]
        convergence_rates = [r['convergence_rate'] for r in results]
        diversities = [r['final_diversity'] for r in results]
        active_particles = [r['avg_active_particles'] for r in results]
        max_outliers = [r['max_outliers'] for r in results]

        return {
            'final_fitness_mean': np.mean(final_fitness),
            'final_fitness_std': np.std(final_fitness),
            'final_fitness_median': np.median(final_fitness),
            'convergence_rate_mean': np.mean(convergence_rates),
            'diversity_mean': np.mean(diversities),
            'diversity_std': np.std(diversities),
            'active_particles_mean': np.mean(active_particles),
            'max_outliers_mean': np.mean(max_outliers),
            'success_rate': len([f for f in final_fitness if f < success_threshold]) / len(final_fitness),
            'all_fitness': final_fitness
        }


class StatisticsVisualizer:
    """Візуалізація статистичних результатів"""

    def __init__(self):
        self.results_data = []

    def add_experiment_result(self, result: Dict, experiment_name: str):
        """Додавання результату експерименту до візуалізації"""
        self.results_data.append({
            'name': experiment_name,
            'result': result
        })

    def create_comparison_plots(self):
        """Створення порівняльних графіків"""
        if len(self.results_data) < 2:
            print("Недостатньо даних для порівняння")
            return

        fig = plt.figure(figsize=(20, 15))

        # 1. Порівняння середніх значень фітнесу
        ax1 = plt.subplot(3, 3, 1)
        self._plot_fitness_comparison(ax1)

        # 2. Розподіл фітнес-значень
        ax2 = plt.subplot(3, 3, 2)
        self._plot_fitness_distributions(ax2)

        # 3. Відсоток погіршення
        ax3 = plt.subplot(3, 3, 3)
        self._plot_degradation_percentage(ax3)

        # 4. Успішність сходимості
        ax4 = plt.subplot(3, 3, 4)
        self._plot_success_rates(ax4)

        # 5. Різноманітність рою
        ax5 = plt.subplot(3, 3, 5)
        self._plot_diversity_comparison(ax5)

        # 6. Кількість активних частинок
        ax6 = plt.subplot(3, 3, 6)
        self._plot_active_particles(ax6)

        # 7. Box plots для фітнесу
        ax7 = plt.subplot(3, 3, 7)
        self._plot_fitness_boxplots(ax7)

        # 8. Статистична значущість (t-test)
        ax8 = plt.subplot(3, 3, 8)
        self._plot_statistical_significance(ax8)

        # 9. Середня кількість викидів
        ax9 = plt.subplot(3, 3, 9)
        self._plot_outliers_count(ax9)

        plt.tight_layout()

        # Збереження
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'pso_statistical_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Статистичний аналіз збережено: {filename}")

        plt.show()

    def _plot_fitness_comparison(self, ax):
        """Порівняння середніх значень фітнесу"""
        experiments = []
        control_means = []
        outlier_means = []
        control_stds = []
        outlier_stds = []

        for data in self.results_data:
            experiments.append(data['name'])
            control_means.append(data['result']['control']['final_fitness_mean'])
            outlier_means.append(data['result']['outliers']['final_fitness_mean'])
            control_stds.append(data['result']['control']['final_fitness_std'])
            outlier_stds.append(data['result']['outliers']['final_fitness_std'])

        x = np.arange(len(experiments))
        width = 0.35

        ax.bar(x - width / 2, control_means, width, yerr=control_stds,
               label='Без викидів', alpha=0.8, capsize=5)
        ax.bar(x + width / 2, outlier_means, width, yerr=outlier_stds,
               label='З викидами', alpha=0.8, capsize=5)

        ax.set_xlabel('Конфігурації експериментів')
        ax.set_ylabel('Середнє значення фітнесу')
        ax.set_title('Порівняння ефективності оптимізації')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_fitness_distributions(self, ax):
        """Розподіл фітнес-значень"""
        for i, data in enumerate(self.results_data):
            control_fitness = data['result']['control']['all_fitness']
            outlier_fitness = data['result']['outliers']['all_fitness']

            ax.hist(control_fitness, bins=50, alpha=0.6,
                    label=f'{data["name"]} - Контроль', density=True)
            ax.hist(outlier_fitness, bins=50, alpha=0.6,
                    label=f'{data["name"]} - Викиди', density=True)

        ax.set_xlabel('Фінальне значення фітнесу')
        ax.set_ylabel('Щільність ймовірності')
        ax.set_title('Розподіл результатів оптимізації')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_degradation_percentage(self, ax):
        """Відсоток погіршення через викиди"""
        experiments = []
        degradations = []

        for data in self.results_data:
            control_mean = data['result']['control']['final_fitness_mean']
            outlier_mean = data['result']['outliers']['final_fitness_mean']

            if control_mean > 0:
                degradation = ((outlier_mean - control_mean) / control_mean) * 100
            else:
                degradation = 0

            experiments.append(data['name'])
            degradations.append(degradation)

        colors = ['red' if d > 0 else 'green' for d in degradations]
        bars = ax.bar(experiments, degradations, color=colors, alpha=0.7)

        # Додавання значень на стовпці
        for bar, deg in zip(bars, degradations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + (1 if height > 0 else -1),
                    f'{deg:.1f}%', ha='center', va='bottom' if height > 0 else 'top')

        ax.set_xlabel('Конфігурації експериментів')
        ax.set_ylabel('Погіршення ефективності (%)')
        ax.set_title('Вплив викидів на ефективність')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)

    def _plot_success_rates(self, ax):
        """Частота успішної сходимості"""
        experiments = []
        control_success = []
        outlier_success = []

        for data in self.results_data:
            experiments.append(data['name'])
            control_success.append(data['result']['control']['success_rate'] * 100)
            outlier_success.append(data['result']['outliers']['success_rate'] * 100)

        x = np.arange(len(experiments))
        width = 0.35

        ax.bar(x - width / 2, control_success, width, label='Без викидів', alpha=0.8)
        ax.bar(x + width / 2, outlier_success, width, label='З викидами', alpha=0.8)

        ax.set_xlabel('Конфігурації експериментів')
        ax.set_ylabel('Відсоток успішних прогонів')
        ax.set_title('Надійність сходимості')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_diversity_comparison(self, ax):
        """Порівняння різноманітності рою"""
        experiments = []
        control_div = []
        outlier_div = []

        for data in self.results_data:
            experiments.append(data['name'])
            control_div.append(data['result']['control']['diversity_mean'])
            outlier_div.append(data['result']['outliers']['diversity_mean'])

        x = np.arange(len(experiments))
        width = 0.35

        ax.bar(x - width / 2, control_div, width, label='Без викидів', alpha=0.8)
        ax.bar(x + width / 2, outlier_div, width, label='З викидами', alpha=0.8)

        ax.set_xlabel('Конфігурації експериментів')
        ax.set_ylabel('Середня різноманітність')
        ax.set_title('Різноманітність рою')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_active_particles(self, ax):
        """Кількість активних частинок"""
        experiments = []
        control_active = []
        outlier_active = []

        for data in self.results_data:
            experiments.append(data['name'])
            control_active.append(data['result']['control']['active_particles_mean'])
            outlier_active.append(data['result']['outliers']['active_particles_mean'])

        x = np.arange(len(experiments))
        width = 0.35

        ax.bar(x - width / 2, control_active, width, label='Без викидів', alpha=0.8)
        ax.bar(x + width / 2, outlier_active, width, label='З викидами', alpha=0.8)

        ax.set_xlabel('Конфігурації експериментів')
        ax.set_ylabel('Середня кількість активних частинок')
        ax.set_title('Активність рою')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_fitness_boxplots(self, ax):
        """Box plots для детального розподілу фітнесу"""
        all_data = []
        labels = []

        for data in self.results_data:
            all_data.append(data['result']['control']['all_fitness'])
            labels.append(f"{data['name']}\nБез викидів")
            all_data.append(data['result']['outliers']['all_fitness'])
            labels.append(f"{data['name']}\nЗ викидами")

        box_plot = ax.boxplot(all_data, labels=labels, patch_artist=True)

        # Кольорування
        colors = ['lightblue', 'lightcoral'] * len(self.results_data)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Конфігурації')
        ax.set_ylabel('Фінальне значення фітнесу')
        ax.set_title('Детальний розподіл результатів')
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_statistical_significance(self, ax):
        """Статистична значущість різниці (t-test)"""
        from scipy import stats

        experiments = []
        p_values = []

        for data in self.results_data:
            control_fitness = data['result']['control']['all_fitness']
            outlier_fitness = data['result']['outliers']['all_fitness']

            # Двосторонній t-test
            t_stat, p_val = stats.ttest_ind(control_fitness, outlier_fitness)

            experiments.append(data['name'])
            p_values.append(p_val)

        colors = ['red' if p < 0.05 else 'green' for p in p_values]
        bars = ax.bar(experiments, p_values, color=colors, alpha=0.7)

        # Лінія значущості
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='p = 0.05')

        # Додавання p-values на стовпці
        for bar, p_val in zip(bars, p_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{p_val:.3f}', ha='center', va='bottom')

        ax.set_xlabel('Конфігурації експериментів')
        ax.set_ylabel('p-value')
        ax.set_title('Статистична значущість різниці')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)

    def _plot_outliers_count(self, ax):
        """Середня кількість викидів"""
        experiments = []
        outliers_counts = []

        for data in self.results_data:
            experiments.append(data['name'])
            outliers_counts.append(data['result']['outliers']['max_outliers_mean'])

        bars = ax.bar(experiments, outliers_counts, alpha=0.7, color='orange')

        # Додавання значень
        for bar, count in zip(bars, outliers_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{count:.1f}', ha='center', va='bottom')

        ax.set_xlabel('Конфігурації експериментів')
        ax.set_ylabel('Середня максимальна кількість викидів')
        ax.set_title('Кількість викидів у рої')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)


def main():
    """Головна функція для проведення статистичних експериментів"""

    print("=== СТАТИСТИЧНЕ ДОСЛІДЖЕННЯ ВПЛИВУ ВИКИДІВ НА АЛГОРИТМ PSO ===\n")

    # Конфігурації експериментів (налаштовані для значущого впливу викидів)
    configs = [
        # 30 частинок (очікується ~12 викидів за експеримент = 40% рою)
        ExperimentConfig(20, 0.0036, 'freeze', 150, 10000),  # 0.36% ймовірність заморожування
        ExperimentConfig(20, 0.0036, 'random_move', 150, 10000),  # 0.36% ймовірність випадкових рухів
        ExperimentConfig(20, 0.0036, 'remove', 150, 10000),  # 0.36% ймовірність видалення

        # 15 частинок (очікується ~6 викидів за експеримент = 40% рою)
        ExperimentConfig(10, 0.0036, 'freeze', 150, 10000),  # 0.36% ймовірність заморожування
        ExperimentConfig(10, 0.0036, 'random_move', 150, 10000),  # 0.36% ймовірність випадкових рухів
        ExperimentConfig(10, 0.0036, 'remove', 150, 10000),  # 0.36% ймовірність видалення
    ]

    runner = StatisticalExperimentRunner()
    visualizer = StatisticsVisualizer()

    # Виконання експериментів
    print("Виконання статистичних експериментів з функцією Растрігіна...")
    print("Це може зайняти 10-15 хвилин (10000 прогонів на конфігурацію)...\n")

    for i, config in enumerate(configs):
        exp_name = f"{config.num_particles}p_{config.outlier_type}"
        print(f"\n[{i + 1}/{len(configs)}] {exp_name}")

        result = runner.run_statistical_experiment(config, ObjectiveFunction.schwefel)
        visualizer.add_experiment_result(result, exp_name)

        # Виведення проміжних результатів
        control_mean = result['control']['final_fitness_mean']
        outlier_mean = result['outliers']['final_fitness_mean']
        degradation = ((outlier_mean - control_mean) / control_mean) * 100 if control_mean > 0 else 0

        print(f"  Контроль: {control_mean:.2f} ± {result['control']['final_fitness_std']:.2f}")
        print(f"  З викидами: {outlier_mean:.2f} ± {result['outliers']['final_fitness_std']:.2f}")
        print(f"  Погіршення: {degradation:+.1f}%")
        print(
            f"  Успішність: {result['control']['success_rate'] * 100:.1f}% → {result['outliers']['success_rate'] * 100:.1f}%")

    # Створення підсумкових графіків
    print("\n=== СТВОРЕННЯ СТАТИСТИЧНИХ ГРАФІКІВ ===")
    visualizer.create_comparison_plots()

    # Додатковий аналіз
    print("\n=== ПІДСУМКОВИЙ АНАЛІЗ ===")
    print_summary_analysis(visualizer.results_data)


def print_summary_analysis(results_data):
    """Виведення підсумкового аналізу"""
    print("\nОсновні висновки:")

    max_degradation = 0
    worst_config = ""

    for data in results_data:
        control_mean = data['result']['control']['final_fitness_mean']
        outlier_mean = data['result']['outliers']['final_fitness_mean']
        degradation = ((outlier_mean - control_mean) / control_mean) * 100 if control_mean > 0 else 0

        if degradation > max_degradation:
            max_degradation = degradation
            worst_config = data['name']

    print(f"1. Найбільший негативний вплив: {worst_config} ({max_degradation:.1f}% погіршення)")

    # Аналіз за розміром рою
    small_swarm_avg = []
    large_swarm_avg = []

    for data in results_data:
        if '15p_' in data['name']:
            control_mean = data['result']['control']['final_fitness_mean']
            outlier_mean = data['result']['outliers']['final_fitness_mean']
            degradation = ((outlier_mean - control_mean) / control_mean) * 100 if control_mean > 0 else 0
            small_swarm_avg.append(degradation)
        elif '30p_' in data['name']:
            control_mean = data['result']['control']['final_fitness_mean']
            outlier_mean = data['result']['outliers']['final_fitness_mean']
            degradation = ((outlier_mean - control_mean) / control_mean) * 100 if control_mean > 0 else 0
            large_swarm_avg.append(degradation)

    if small_swarm_avg and large_swarm_avg:
        print(f"2. Середнє погіршення для малих роїв (15 частинок): {np.mean(small_swarm_avg):.1f}%")
        print(f"3. Середнє погіршення для великих роїв (30 частинок): {np.mean(large_swarm_avg):.1f}%")

        if np.mean(small_swarm_avg) > np.mean(large_swarm_avg):
            print("4. Малі рої (15 частинок) більш чутливі до викидів")
        else:
            print("4. Великі рої (30 частинок) більш чутливі до викидів")


if __name__ == "__main__":
    main()