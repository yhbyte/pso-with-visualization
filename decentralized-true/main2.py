import logging
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple

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


class AutonomousParticle:

    def __init__(self, particle_id: int, dimensions: int, bounds: Tuple[float, float],
                 base_port: int = 5000, fitness_function_name: str = 'sphere'):
        self.id = particle_id
        self.dimensions = dimensions
        self.bounds = bounds
        self.port = base_port + particle_id
        self.fitness_function_name = fitness_function_name

        # PSO параметри
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

        # Ініціалізація позиції та швидкості
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-0.5, 0.5, dimensions)

        # Особисті знання
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('inf')
        self.current_fitness = float('inf')

        # Знання про сусідів
        self.neighbors: Set[int] = set()
        self.neighbor_info: Dict[int, ParticleInfo] = {}
        self.local_best_position = self.position.copy()
        self.local_best_fitness = float('inf')

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
        self.flask_app.logger.setLevel(logging.INFO)

        @self.flask_app.route('/info', methods=['GET'])
        def get_info():
            return jsonify(asdict(self.get_particle_info()))

        @self.flask_app.route('/exchange', methods=['POST'])
        def exchange_info():
            neighbor_data = request.get_json()
            self.receive_neighbor_info(neighbor_data)
            return jsonify(asdict(self.get_particle_info()))

        @self.flask_app.route('/add_neighbor', methods=['POST'])
        def add_neighbor():
            neighbor_id = request.get_json().get('neighbor_id')
            self.neighbors.add(neighbor_id)
            return jsonify({'status': 'ok'})

        @self.flask_app.route('/status', methods=['GET'])
        def get_status():
            return jsonify({
                'id': self.id,
                'iteration': self.iteration_count,
                'fitness': self.current_fitness,
                'personal_best': self.personal_best_fitness,
                'neighbors': list(self.neighbors),
                'running': self.running
            })

    def get_particle_info(self) -> ParticleInfo:
        return ParticleInfo(
            id=self.id,
            position=self.position.tolist(),
            velocity=self.velocity.tolist(),
            current_fitness=self.current_fitness,
            personal_best_position=self.personal_best_position.tolist(),
            personal_best_fitness=self.personal_best_fitness,
            timestamp=time.time()
        )

    def receive_neighbor_info(self, neighbor_data: Dict):
        neighbor_info = ParticleInfo(**neighbor_data)
        self.neighbor_info[neighbor_info.id] = neighbor_info
        self.update_local_best()

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
        if self.fitness_function_name == 'sphere':
            self.current_fitness = np.sum(self.position ** 2)
        elif self.fitness_function_name == 'rastrigin':
            A = 10
            n = len(self.position)
            self.current_fitness = A * n + np.sum(self.position ** 2 - A * np.cos(2 * np.pi * self.position))

        # Оновити особистий best
        if self.current_fitness < self.personal_best_fitness:
            self.personal_best_fitness = self.current_fitness
            self.personal_best_position = self.position.copy()

        # Зберегти історію
        self.fitness_history.append(self.current_fitness)
        self.position_history.append(self.position.copy())

    def communicate_with_neighbors(self):
        my_info = asdict(self.get_particle_info())

        def contact_neighbor(neighbor_id):
            try:
                port = 5000 + neighbor_id
                response = requests.post(
                    f'http://localhost:{port}/exchange',
                    json=my_info,
                    timeout=0.1
                )
                if response.status_code == 200:
                    return response.json()
            except:
                # Сусід недоступний - видаляємо з списку
                self.neighbors.discard(neighbor_id)
            return None

        # Паралельно спілкуємося з усіма сусідами
        with ThreadPoolExecutor(max_workers=len(self.neighbors) or 1) as executor:
            futures = {executor.submit(contact_neighbor, nid): nid
                       for nid in self.neighbors.copy()}

            for future in as_completed(futures, timeout=0.5):
                try:
                    response = future.result()
                    if response:
                        self.receive_neighbor_info(response)
                except:
                    pass

    def update_velocity_and_position(self):
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)

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
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def pso_main_loop(self):
        while self.running:
            try:
                self.evaluate_fitness()
                self.communicate_with_neighbors()
                self.update_velocity_and_position()

                self.iteration_count += 1

            except Exception as e:
                if self.running:
                    print(f"Particle {self.id} error: {e}")
                break

    def start(self):
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

        # Дати час серверу запуститися
        time.sleep(0.5)

        # Запустити PSO цикл
        self.pso_thread = threading.Thread(target=self.pso_main_loop, daemon=True)
        self.pso_thread.start()

    def stop(self):
        self.running = False
        if self.pso_thread and self.pso_thread.is_alive():
            self.pso_thread.join(timeout=1)


"""Менеджер для координації запуску і зупинки автономних частинок"""
class DecentralizedSwarmManager:

    def __init__(self, num_particles: int, dimensions: int, bounds: Tuple[float, float]):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.particles: List[AutonomousParticle] = []
        self.running = False

    def create_particles(self, fitness_function_name: str = 'sphere'):
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

        for particle in self.particles:
            particle.start()

        self.running = True
        print("Всі частинки запущені!")

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
            'convergence_history': []
        }

        max_iterations = 0

        for particle in self.particles:
            try:
                max_iterations = self.update_status(max_iterations, particle, stats)
            except:
                continue

        self.generate_general_convergence_history(max_iterations, stats)
        return stats

    def generate_general_convergence_history(self, max_iterations, stats):
        if max_iterations > 0:
            global_fitness_history = []
            for i in range(max_iterations):
                best_fitness_at_iteration = float('inf')
                for particle in self.particles:
                    if i < len(particle.fitness_history):
                        personal_best_at_i = min(particle.fitness_history[:i + 1])
                        best_fitness_at_iteration = min(best_fitness_at_iteration, personal_best_at_i)
                global_fitness_history.append(best_fitness_at_iteration)

            stats['convergence_history'] = global_fitness_history

    def update_status(self, max_iterations, particle, stats):
        response = requests.get(f'http://localhost:{particle.port}/status', timeout=0.5)
        if response.status_code == 200:
            particle_status = response.json()
            stats['particles'].append(particle_status)

            if particle.personal_best_fitness < stats['global_best_fitness']:
                stats['global_best_fitness'] = particle.personal_best_fitness
                stats['global_best_position'] = particle.personal_best_position.tolist()

            stats['iteration_counts'].append(particle.iteration_count)
            max_iterations = max(max_iterations, len(particle.fitness_history))
        return max_iterations


def run_experiment(num_particles: int, topology: str, function: str, duration: int = 10):
    print(f"\n🔄 Експеримент: {topology} топологія, {function} функція")

    # Створити менеджер
    manager = DecentralizedSwarmManager(
        num_particles=num_particles,
        dimensions=2,
        bounds=(-5.0, 5.0)
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
            'convergence': stats['convergence_history']
        }

    finally:
        # Завжди зупинити частинки
        manager.stop_swarm()
        time.sleep(1)  # Дати час на cleanup


def start_decentralized_pso():
    print("🚀 Демонстрація децентралізованого PSO")
    print("=" * 60)
    print("Кожна частинка - це окремий HTTP сервер!")

    num_particles = 8  # Менше частинок для швидкості
    duration = 8  # секунд на експеримент

    # Різні конфігурації для тестування
    topologies = ['ring', 'star', 'random']
    functions = ['sphere', 'rastrigin']

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

            except Exception as e:
                print(f"    Помилка: {e}")
                results[function][topology] = {
                    'final_fitness': float('inf'),
                    'convergence': []
                }

    return results


def visualize_decentralized_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Децентралізований PSO (HTTP + Threading)', fontsize=16, fontweight='bold')

    topology_colors = {'ring': 'blue', 'star': 'red', 'random': 'green'}
    topology_styles = {'ring': '-', 'star': '--', 'random': '-.'}

    # 1. Конвергенція Sphere
    ax1 = axes[0, 0]
    for topology, data in results.get('sphere', {}).items():
        if data['convergence']:
            ax1.plot(data['convergence'], label=topology.title(),
                     color=topology_colors[topology],
                     linestyle=topology_styles[topology], linewidth=2)

    ax1.set_title('Конвергенція: Sphere функція', fontweight='bold')
    ax1.set_xlabel('Ітерація')
    ax1.set_ylabel('Найкращий fitness')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Конвергенція Rastrigin
    ax2 = axes[0, 1]
    for topology, data in results.get('rastrigin', {}).items():
        if data['convergence']:
            ax2.plot(data['convergence'], label=topology.title(),
                     color=topology_colors[topology],
                     linestyle=topology_styles[topology], linewidth=2)

    ax2.set_title('Конвергенція: Rastrigin функція', fontweight='bold')
    ax2.set_xlabel('Ітерація')
    ax2.set_ylabel('Найкращий fitness')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Порівняння фінальних результатів
    ax3 = axes[1, 0]
    functions = list(results.keys())
    if functions:
        topologies = list(results[functions[0]].keys())

        x = np.arange(len(functions))
        width = 0.25

        for i, topology in enumerate(topologies):
            values = []
            for func in functions:
                if func in results and topology in results[func]:
                    val = results[func][topology]['final_fitness']
                    if val == float('inf'):
                        val = 1000  # Для візуалізації
                    values.append(val)
                else:
                    values.append(1000)

            bars = ax3.bar(x + i * width, values, width, label=topology.title(),
                           color=topology_colors[topology], alpha=0.7)

            # Додати значення на стовпчики
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if height < 1000:
                    ax3.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        ax3.set_title('Фінальні результати', fontweight='bold')
        ax3.set_xlabel('Функція')
        ax3.set_ylabel('Фінальний fitness')
        ax3.set_yscale('log')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([f.title() for f in functions])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Архітектурна діаграма
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.8, '🌐 Децентралізація',
             ha='center', va='center', fontsize=14, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.6,
             '• Кожна частинка = HTTP сервер\n• Асинхронна комунікація\n• Без центрального координатора\n• Відмовостійкість',
             ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.5, 0.3,
             'Particle 0:5000 ↔ Particle 1:5001\n     ↕                    ↕\nParticle 7:5007 ↔ Particle 2:5002',
             ha='center', va='center', fontsize=10, family='monospace', transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

    # Статистика
    print("\n📊 Результати експериментів:")
    print("=" * 50)

    for func_name in results.keys():
        print(f"\n🎯 Функція {func_name.title()}:")
        for topo_name in results[func_name].keys():
            final_fitness = results[func_name][topo_name]['final_fitness']
            if final_fitness == float('inf'):
                print(f"  {topo_name.title():8}: Не завершено")
            else:
                print(f"  {topo_name.title():8}: {final_fitness:.6f}")


def signal_handler(sig, frame):
    print("\n🛑 Отримано сигнал зупинки...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    try:
        results = start_decentralized_pso()
        visualize_decentralized_results(results)

        print("\n✅ Демонстрація завершена!")

    except KeyboardInterrupt:
        print("\n🛑 Зупинено користувачем")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")