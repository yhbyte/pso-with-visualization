import logging
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.WARNING)


@dataclass
class ParticleInfo:
    id: int
    position: List[float]
    velocity: List[float]
    current_fitness: float
    personal_best_position: List[float]
    personal_best_fitness: float
    timestamp: float
    is_outlier: bool = False
    outlier_score: float = 0.0

    def to_dict(self):
        """Конвертація до словника з правильними типами для JSON"""
        return {
            'id': int(self.id),
            'position': [float(x) for x in self.position],
            'velocity': [float(v) for v in self.velocity],
            'current_fitness': float(self.current_fitness),
            'personal_best_position': [float(x) for x in self.personal_best_position],
            'personal_best_fitness': float(self.personal_best_fitness),
            'timestamp': float(self.timestamp),
            'is_outlier': bool(self.is_outlier),
            'outlier_score': float(self.outlier_score)
        }


class OutlierDetector:
    """Клас для виявлення викидів у рої"""

    @staticmethod
    def calculate_outlier_score(particle_fitness: float, neighbor_fitnesses: List[float]) -> float:
        """Розрахунок показника викиду на основі відхилення від сусідів"""
        if not neighbor_fitnesses:
            return 0.0

        # Конвертуємо до стандартних Python типів
        particle_fitness = float(particle_fitness)
        neighbor_fitnesses = [float(f) for f in neighbor_fitnesses]

        mean_neighbor_fitness = np.mean(neighbor_fitnesses)
        std_neighbor_fitness = np.std(neighbor_fitnesses) + 1e-10  # Уникнення ділення на нуль

        # Z-score відносно сусідів
        z_score = abs(particle_fitness - mean_neighbor_fitness) / std_neighbor_fitness
        return float(min(z_score, 10.0))  # Обмежуємо максимальне значення

    @staticmethod
    def is_outlier(outlier_score: float, threshold: float = 2.0) -> bool:
        """Визначення чи є частинка викидом"""
        return outlier_score > threshold


class AutonomousParticle:

    def __init__(self, particle_id: int, dimensions: int, bounds: Tuple[float, float],
                 base_port: int = 5000, fitness_function_name: str = 'rastrigin'):
        self.id = particle_id
        self.dimensions = dimensions
        self.bounds = bounds
        self.port = base_port + particle_id
        self.fitness_function_name = fitness_function_name

        # PSO параметри (збільшені для більшої експлорації)
        self.w = 0.9  # Збільшена інерція
        self.c1 = 2.0  # Збільшений когнітивний компонент
        self.c2 = 2.0  # Збільшений соціальний компонент

        # Ініціалізація позиції та швидкості з більшою варіативністю
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1.0, 1.0, dimensions)  # Збільшена початкова швидкість

        # Особисті знання
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('inf')
        self.current_fitness = float('inf')

        # Знання про сусідів
        self.neighbors: Set[int] = set()
        self.neighbor_info: Dict[int, ParticleInfo] = {}
        self.local_best_position = self.position.copy()
        self.local_best_fitness = float('inf')

        # Детекція викидів
        self.outlier_detector = OutlierDetector()
        self.is_outlier = False
        self.outlier_score = 0.0
        self.outlier_history = []

        # Статистика для візуалізації
        self.iteration_count = 0
        self.fitness_history = []
        self.position_history = []

        # Керування потоками
        self.running = True
        self.flask_app = None
        self.pso_thread = None
        self.server_thread = None

        # Налаштувати Flask сервер
        self.setup_flask_server()

    def setup_flask_server(self):
        self.flask_app = Flask(f'particle_{self.id}')
        self.flask_app.logger.setLevel(logging.ERROR)  # Зменшуємо логування

        @self.flask_app.route('/info', methods=['GET'])
        def get_info():
            try:
                return jsonify(self.get_particle_info().to_dict())
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.flask_app.route('/exchange', methods=['POST'])
        def exchange_info():
            try:
                neighbor_data = request.get_json()
                if neighbor_data:
                    self.receive_neighbor_info(neighbor_data)
                return jsonify(self.get_particle_info().to_dict())
            except Exception as e:
                print(f"Particle {self.id} exchange error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.flask_app.route('/add_neighbor', methods=['POST'])
        def add_neighbor():
            try:
                data = request.get_json()
                if data and 'neighbor_id' in data:
                    neighbor_id = int(data['neighbor_id'])
                    self.neighbors.add(neighbor_id)
                return jsonify({'status': 'ok'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.flask_app.route('/status', methods=['GET'])
        def get_status():
            try:
                return jsonify({
                    'id': int(self.id),
                    'iteration': int(self.iteration_count),
                    'fitness': float(self.current_fitness),
                    'personal_best': float(self.personal_best_fitness),
                    'neighbors': list(self.neighbors),
                    'running': bool(self.running),
                    'is_outlier': bool(self.is_outlier),
                    'outlier_score': float(self.outlier_score),
                    'position': [float(x) for x in self.position]
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def get_particle_info(self) -> ParticleInfo:
        return ParticleInfo(
            id=int(self.id),
            position=[float(x) for x in self.position],
            velocity=[float(v) for v in self.velocity],
            current_fitness=float(self.current_fitness),
            personal_best_position=[float(x) for x in self.personal_best_position],
            personal_best_fitness=float(self.personal_best_fitness),
            timestamp=float(time.time()),
            is_outlier=bool(self.is_outlier),
            outlier_score=float(self.outlier_score)
        )

    def receive_neighbor_info(self, neighbor_data: Dict):
        try:
            # Створюємо ParticleInfo з отриманих даних
            neighbor_info = ParticleInfo(
                id=int(neighbor_data.get('id', 0)),
                position=[float(x) for x in neighbor_data.get('position', [])],
                velocity=[float(v) for v in neighbor_data.get('velocity', [])],
                current_fitness=float(neighbor_data.get('current_fitness', float('inf'))),
                personal_best_position=[float(x) for x in neighbor_data.get('personal_best_position', [])],
                personal_best_fitness=float(neighbor_data.get('personal_best_fitness', float('inf'))),
                timestamp=float(neighbor_data.get('timestamp', time.time())),
                is_outlier=bool(neighbor_data.get('is_outlier', False)),
                outlier_score=float(neighbor_data.get('outlier_score', 0.0))
            )

            self.neighbor_info[neighbor_info.id] = neighbor_info
            self.update_local_best()
            self.detect_outliers()
        except Exception as e:
            print(f"Particle {self.id} error processing neighbor info: {e}")

    def detect_outliers(self):
        """Виявлення викидів на основі порівняння з сусідами"""
        try:
            neighbor_fitnesses = [float(info.current_fitness) for info in self.neighbor_info.values()]

            if neighbor_fitnesses:
                self.outlier_score = self.outlier_detector.calculate_outlier_score(
                    float(self.current_fitness), neighbor_fitnesses
                )
                self.is_outlier = self.outlier_detector.is_outlier(self.outlier_score)
                self.outlier_history.append(float(self.outlier_score))
        except Exception as e:
            print(f"Particle {self.id} outlier detection error: {e}")
            self.outlier_score = 0.0
            self.is_outlier = False

    def update_local_best(self):
        best_fitness = self.personal_best_fitness
        best_position = self.personal_best_position.copy()

        for neighbor_id in self.neighbors:
            if neighbor_id in self.neighbor_info:
                neighbor = self.neighbor_info[neighbor_id]
                if neighbor.personal_best_fitness < best_fitness:
                    best_fitness = neighbor.personal_best_fitness
                    best_position = np.array(neighbor.personal_best_position)

        self.local_best_fitness = best_fitness
        self.local_best_position = best_position

    def evaluate_fitness(self):
        try:
            if self.fitness_function_name == 'rastrigin':
                A = 10
                n = len(self.position)
                self.current_fitness = float(A * n + np.sum(self.position ** 2 - A * np.cos(2 * np.pi * self.position)))

            elif self.fitness_function_name == 'ackley':
                # Ackley function - багато локальних мінімумів
                a, b, c = 20, 0.2, 2 * np.pi
                sum_sq = np.sum(self.position ** 2)
                sum_cos = np.sum(np.cos(c * self.position))
                n = len(self.position)

                self.current_fitness = float(-a * np.exp(-b * np.sqrt(sum_sq / n)) -
                                             np.exp(sum_cos / n) + a + np.exp(1))

            elif self.fitness_function_name == 'schwefel':
                # Schwefel function - дуже складна топографія
                n = len(self.position)
                self.current_fitness = float(
                    418.9829 * n - np.sum(self.position * np.sin(np.sqrt(np.abs(self.position)))))

            elif self.fitness_function_name == 'griewank':
                # Griewank function - комбінація глобальних та локальних властивостей
                sum_sq = np.sum(self.position ** 2)
                indices = np.arange(1, len(self.position) + 1)
                prod_cos = np.prod(np.cos(self.position / np.sqrt(indices)))
                self.current_fitness = float(1 + sum_sq / 4000 - prod_cos)

            # Переконуємося, що fitness є валідним числом
            if np.isnan(self.current_fitness) or np.isinf(self.current_fitness):
                self.current_fitness = float('inf')

            # Оновити особистий best
            if self.current_fitness < self.personal_best_fitness:
                self.personal_best_fitness = float(self.current_fitness)
                self.personal_best_position = self.position.copy()

            # Зберегти історію
            self.fitness_history.append(float(self.current_fitness))
            self.position_history.append(self.position.copy())

        except Exception as e:
            print(f"Particle {self.id} fitness evaluation error: {e}")
            self.current_fitness = float('inf')

    def communicate_with_neighbors(self):
        my_info = self.get_particle_info().to_dict()

        def contact_neighbor(neighbor_id):
            try:
                port = 5000 + neighbor_id
                response = requests.post(
                    f'http://localhost:{port}/exchange',
                    json=my_info,
                    timeout=0.2
                )
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                # Сусід недоступний - видаляємо з списку
                self.neighbors.discard(neighbor_id)
            return None

        # Паралельно спілкуємося з усіма сусідами
        if self.neighbors:
            with ThreadPoolExecutor(max_workers=min(len(self.neighbors), 5)) as executor:
                futures = {executor.submit(contact_neighbor, nid): nid
                           for nid in self.neighbors.copy()}

                for future in as_completed(futures, timeout=1.0):
                    try:
                        response = future.result()
                        if response and 'error' not in response:
                            self.receive_neighbor_info(response)
                    except Exception as e:
                        pass

    def update_velocity_and_position(self):
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)

        cognitive_component = self.c1 * r1 * (self.personal_best_position - self.position)
        social_component = self.c2 * r2 * (self.local_best_position - self.position)

        # Модифікація для викидів - зменшена соціальна складова
        if self.is_outlier:
            social_component *= 0.5  # Зменшуємо вплив соціального компоненту для викидів

        self.velocity = (self.w * self.velocity +
                         cognitive_component +
                         social_component)

        # Обмеження швидкості (збільшене для більшої експлорації)
        max_velocity = 0.5 * (self.bounds[1] - self.bounds[0])
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

        # Оновлення позиції
        self.position += self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

        # Додавання шуму для викидів (симуляція проблемної поведінки)
        if self.iteration_count % 50 == 0 and random.random() < 0.1:  # 10% ймовірність шуму кожні 50 ітерацій
            noise = np.random.normal(0, 0.5, self.dimensions)
            self.position += noise
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def pso_main_loop(self):
        while self.running:
            try:
                self.evaluate_fitness()

                # Комунікація з сусідами (не кожну ітерацію для стабільності)
                if self.iteration_count % 2 == 0:  # Кожну другу ітерацію
                    self.communicate_with_neighbors()

                self.update_velocity_and_position()
                self.iteration_count += 1

                # Збільшена затримка для стабільності
                time.sleep(0.2)

            except Exception as e:
                if self.running:
                    print(f"Particle {self.id} error in iteration {self.iteration_count}: {e}")
                time.sleep(0.5)  # Затримка при помилці

    def start(self):
        try:
            # Запустити Flask сервер в окремому потоці
            self.server_thread = threading.Thread(
                target=lambda: self.flask_app.run(
                    host='localhost',
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                ),
                daemon=True
            )
            self.server_thread.start()

            # Дати більше часу серверу запуститися
            time.sleep(1.0)

            # Запустити PSO цикл
            self.pso_thread = threading.Thread(target=self.pso_main_loop, daemon=True)
            self.pso_thread.start()

        except Exception as e:
            print(f"Error starting particle {self.id}: {e}")
            self.running = False

    def stop(self):
        self.running = False
        if self.pso_thread and self.pso_thread.is_alive():
            self.pso_thread.join(timeout=1)


class DecentralizedSwarmManager:

    def __init__(self, num_particles: int, dimensions: int, bounds: Tuple[float, float]):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.particles: List[AutonomousParticle] = []
        self.running = False

    def create_particles(self, fitness_function_name: str = 'ra'):
        self.particles = []
        for i in range(self.num_particles):
            particle = AutonomousParticle(
                particle_id=i,
                dimensions=self.dimensions,
                bounds=self.bounds,
                fitness_function_name=fitness_function_name
            )
            self.particles.append(particle)

    def setup_topology(self, topology_type: str = 'ring'):
        if topology_type == 'ring':
            self.setup_ring_topology()
        elif topology_type == 'star':
            self.setup_star_topology()
        elif topology_type == 'random':
            self.setup_random_topology()

    def setup_ring_topology(self):
        for i in range(self.num_particles):
            left = (i - 1) % self.num_particles
            right = (i + 1) % self.num_particles
            self.particles[i].neighbors.add(left)
            self.particles[i].neighbors.add(right)

    def setup_star_topology(self):
        for i in range(1, self.num_particles):
            self.particles[0].neighbors.add(i)
            self.particles[i].neighbors.add(0)

    def setup_random_topology(self, connections_per_particle: int = 3):
        for i in range(self.num_particles):
            possible_neighbors = list(range(self.num_particles))
            possible_neighbors.remove(i)

            num_connections = min(connections_per_particle, len(possible_neighbors))
            neighbors = random.sample(possible_neighbors, num_connections)

            for neighbor in neighbors:
                self.particles[i].neighbors.add(neighbor)
                self.particles[neighbor].neighbors.add(i)

    def start_swarm(self):
        print(f"Запуск {self.num_particles} автономних частинок...")

        # Запускаємо частинки послідовно з затримкою
        for i, particle in enumerate(self.particles):
            print(f"  Запуск частинки {i}...")
            particle.start()
            time.sleep(0.5)  # Затримка між запусками

        self.running = True
        print("Всі частинки запущені! Очікування стабілізації...")
        time.sleep(2)  # Додатковий час для стабілізації

    def stop_swarm(self):
        print("Зупинка частинок...")

        for particle in self.particles:
            particle.stop()

        self.running = False
        print("Всі частинки зупинені!")

    def collect_statistics(self):
        stats = {
            'particles': [],
            'global_best_fitness': float('inf'),
            'global_best_position': None,
            'iteration_counts': [],
            'convergence_history': [],
            'outlier_counts': [],
            'outlier_history': []
        }

        max_iterations = 0
        outlier_count = 0

        for particle in self.particles:
            try:
                response = requests.get(f'http://localhost:{particle.port}/status', timeout=0.5)
                if response.status_code == 200:
                    particle_status = response.json()
                    stats['particles'].append(particle_status)

                    if particle.personal_best_fitness < stats['global_best_fitness']:
                        stats['global_best_fitness'] = particle.personal_best_fitness
                        stats['global_best_position'] = particle.personal_best_position.tolist()

                    stats['iteration_counts'].append(particle.iteration_count)
                    max_iterations = max(max_iterations, len(particle.fitness_history))

                    if particle_status['is_outlier']:
                        outlier_count += 1

            except:
                continue

        stats['outlier_counts'] = outlier_count

        # Генерація історії конвергенції
        if max_iterations > 0:
            global_fitness_history = []
            outlier_count_history = []

            for i in range(max_iterations):
                best_fitness_at_iteration = float('inf')
                outliers_at_iteration = 0

                for particle in self.particles:
                    if i < len(particle.fitness_history):
                        personal_best_at_i = min(particle.fitness_history[:i + 1])
                        best_fitness_at_iteration = min(best_fitness_at_iteration, personal_best_at_i)

                    if i < len(particle.outlier_history):
                        if particle.outlier_history[i] > 2.0:  # Поріг викиду
                            outliers_at_iteration += 1

                global_fitness_history.append(best_fitness_at_iteration)
                outlier_count_history.append(outliers_at_iteration)

            stats['convergence_history'] = global_fitness_history
            stats['outlier_history'] = outlier_count_history

        return stats


def run_experiment(num_particles: int, topology: str, function: str, duration: int = 20):
    print(f"\n🔄 Експеримент: {topology} топологія, {function} функція")

    # Створити менеджер
    manager = DecentralizedSwarmManager(
        num_particles=num_particles,
        dimensions=2,
        bounds=(-10.0, 10.0)  # Збільшений діапазон
    )

    # Створити і налаштувати частинки
    manager.create_particles(function)
    manager.setup_topology(topology)

    try:
        # Запустити рій
        manager.start_swarm()

        # Дати час на роботу
        time.sleep(duration)

        # Зібрати статистику
        stats = manager.collect_statistics()

        return {
            'final_fitness': stats['global_best_fitness'],
            'convergence': stats['convergence_history'],
            'outlier_count': stats['outlier_counts'],
            'outlier_history': stats['outlier_history']
        }

    finally:
        # Завжди зупинити частинки
        manager.stop_swarm()
        time.sleep(1)


def start_decentralized_pso():
    print("🚀 Демонстрація децентралізованого PSO з детекцією викидів")
    print("=" * 70)

    num_particles = 12  # Збільшена кількість для кращої демонстрації
    duration = 20  # Збільшений час для демонстрації викидів

    # Функції, що провокують викиди
    functions = ['ackley', 'schwefel', 'griewank', 'rastrigin']
    topologies = ['ring', 'random']

    results = {}

    for function in functions:
        print(f"\n📊 Тестування функції: {function}")
        results[function] = {}

        for topology in topologies:
            print(f"  Топологія: {topology}...")

            try:
                result = run_experiment(
                    num_particles=num_particles,
                    topology=topology,
                    function=function,
                    duration=duration
                )
                results[function][topology] = result
                print(f"    Результат: {result['final_fitness']:.6f}")
                print(f"    Викиди: {result['outlier_count']}")

            except Exception as e:
                print(f"    Помилка: {e}")
                results[function][topology] = {
                    'final_fitness': float('inf'),
                    'convergence': [],
                    'outlier_count': 0,
                    'outlier_history': []
                }

    return results


def visualize_decentralized_results_with_outliers(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Децентралізований PSO з детекцією викидів', fontsize=16, fontweight='bold')

    topology_colors = {'ring': 'blue', 'random': 'green'}
    topology_styles = {'ring': '-', 'random': '--'}

    # 1. Конвергенція для різних функцій
    functions = list(results.keys())[:2]  # Беремо перші 2 функції

    for idx, function in enumerate(functions):
        ax = axes[0, idx]
        for topology, data in results.get(function, {}).items():
            if data['convergence']:
                ax.plot(data['convergence'], label=f'{topology.title()}',
                        color=topology_colors[topology],
                        linestyle=topology_styles[topology], linewidth=2)

        ax.set_title(f'Конвергенція: {function.title()}', fontweight='bold')
        ax.set_xlabel('Ітерація')
        ax.set_ylabel('Найкращий fitness')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Кількість викидів в часі
    ax2 = axes[0, 2]
    for function in functions:
        for topology, data in results.get(function, {}).items():
            if data.get('outlier_history'):
                ax2.plot(data['outlier_history'],
                         label=f'{function}-{topology}',
                         linewidth=2)

    ax2.set_title('Динаміка викидів', fontweight='bold')
    ax2.set_xlabel('Ітерація')
    ax2.set_ylabel('Кількість викидів')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Порівняння фінальних результатів
    ax3 = axes[1, 0]
    if functions:
        topologies = list(results[functions[0]].keys())
        x = np.arange(len(functions))
        width = 0.35

        for i, topology in enumerate(topologies):
            values = []
            for func in functions:
                if func in results and topology in results[func]:
                    val = results[func][topology]['final_fitness']
                    if val == float('inf'):
                        val = 1000
                    values.append(val)
                else:
                    values.append(1000)

            bars = ax3.bar(x + i * width, values, width, label=topology.title(),
                           color=topology_colors[topology], alpha=0.7)

        ax3.set_title('Фінальні результати', fontweight='bold')
        ax3.set_xlabel('Функція')
        ax3.set_ylabel('Фінальний fitness')
        ax3.set_yscale('log')
        ax3.set_xticks(x + width / 2)
        ax3.set_xticklabels([f.title() for f in functions])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Статистика викидів
    ax4 = axes[1, 1]
    outlier_data = {}
    for function in functions:
        outlier_data[function] = {}
        for topology in results[function].keys():
            outlier_data[function][topology] = results[function][topology]['outlier_count']

    if outlier_data:
        x = np.arange(len(functions))
        width = 0.35

        for i, topology in enumerate(topologies):
            values = [outlier_data[func].get(topology, 0) for func in functions]
            ax4.bar(x + i * width, values, width, label=topology.title(),
                    color=topology_colors[topology], alpha=0.7)

        ax4.set_title('Кількість викидів', fontweight='bold')
        ax4.set_xlabel('Функція')
        ax4.set_ylabel('Викиди')
        ax4.set_xticks(x + width / 2)
        ax4.set_xticklabels([f.title() for f in functions])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Методологія
    ax5 = axes[1, 2]
    ax5.text(0.5, 0.9, '🎯 Детекція викидів',
             ha='center', va='center', fontsize=14, fontweight='bold', transform=ax5.transAxes)
    ax5.text(0.5, 0.7,
             '• Z-score відносно сусідів\n• Поріг: σ > 2.0\n• Реальний час\n• Модифікована поведінка',
             ha='center', va='center', fontsize=12, transform=ax5.transAxes)
    ax5.text(0.5, 0.4,
             'if |fitness - mean_neighbors| / std > 2:\n    particle.is_outlier = True\n    reduce_social_influence()',
             ha='center', va='center', fontsize=10, family='monospace', transform=ax5.transAxes)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')

    plt.tight_layout()
    plt.show()

    # Детальна статистика
    print("\n📊 Результати експериментів з детекцією викидів:")
    print("=" * 60)

    for func_name in results.keys():
        print(f"\n🎯 Функція {func_name.title()}:")
        for topo_name in results[func_name].keys():
            final_fitness = results[func_name][topo_name]['final_fitness']
            outlier_count = results[func_name][topo_name]['outlier_count']

            if final_fitness == float('inf'):
                print(f"  {topo_name.title():8}: Не завершено")
            else:
                print(f"  {topo_name.title():8}: Fitness={final_fitness:.6f}, Викиди={outlier_count}")


if __name__ == "__main__":
    try:
        results = start_decentralized_pso()
        visualize_decentralized_results_with_outliers(results)
        print("\n✅ Демонстрація завершена!")

    except KeyboardInterrupt:
        print("\n🛑 Зупинено користувачем")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")