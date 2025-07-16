import matplotlib.pyplot as plt
import numpy as np

from Swarm import Swarm


# Функція сфери - проста унімодальна функція
def sphere_function(x):
    return np.sum(x ** 2)


# Функція Растригіна - складна мультимодальна функція
def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


# Функція Аклі - складна мультимодальна функція
def ackley_function(x):
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)


def main():
    print("🔄 Демонстрація децентралізованого PSO")
    print("=" * 50)

    # Параметри
    num_particles = 20
    dimensions = 2
    bounds = (-5.0, 5.0)
    max_iterations = 100

    # Створити рій
    swarm = Swarm(num_particles, dimensions, bounds)

    # Тестування різних топологій
    topologies = [
        ("Ring", lambda: swarm.setup_ring_topology()),
        ("Star", lambda: swarm.setup_star_topology()),
        ("Random", lambda: swarm.setup_random_topology(3))
    ]

    functions = [
        ("Sphere", sphere_function),
        ("Rastrigin", rastrigin_function)
    ]

    results = {}

    for func_name, fitness_func in functions:
        print(f"\n📊 Тестування функції: {func_name}")
        print("-" * 30)

        results[func_name] = {}

        for topo_name, topo_setup in topologies:
            print(f"  Топологія: {topo_name}")

            # Скинути рій
            swarm = Swarm(num_particles, dimensions, bounds)
            topo_setup()

            # Виконати оптимізацію
            for iteration in range(max_iterations):
                swarm.perform_iteration(fitness_func)

                if iteration % 20 == 0:
                    status = swarm.get_swarm_status()
                    print(f"    Ітерація {iteration:3d}: Best = {status['global_best_fitness']:.6f}")

            final_status = swarm.get_swarm_status()
            results[func_name][topo_name] = {
                'final_fitness': final_status['global_best_fitness'],
                'convergence': final_status['convergence_history']
            }

            print(f"    Фінальний результат: {final_status['global_best_fitness']:.6f}")
            print(f"    Позиція: {final_status['global_best_position']}")

    return results


def visualize_swarm_behavior(results):
    """Візуалізація результатів демонстрації"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Результати децентралізованого PSO', fontsize=16, fontweight='bold')

    # Кольори та стилі для топологій
    topology_colors = {'Ring': 'blue', 'Star': 'red', 'Random': 'green'}
    topology_styles = {'Ring': '-', 'Star': '--', 'Random': '-.'}

    # 1. Конвергенція для функції Sphere
    ax1 = axes[0, 0]
    for topology, data in results['Sphere'].items():
        convergence = data['convergence']
        ax1.plot(convergence, label=topology,
                 color=topology_colors[topology],
                 linestyle=topology_styles[topology],
                 linewidth=2)

    ax1.set_title('Конвергенція: Функція Sphere', fontweight='bold')
    ax1.set_xlabel('Ітерація')
    ax1.set_ylabel('Найкращий fitness')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Конвергенція для функції Rastrigin
    ax2 = axes[0, 1]
    for topology, data in results['Rastrigin'].items():
        convergence = data['convergence']
        ax2.plot(convergence, label=topology,
                 color=topology_colors[topology],
                 linestyle=topology_styles[topology],
                 linewidth=2)

    ax2.set_title('Конвергенція: Функція Rastrigin', fontweight='bold')
    ax2.set_xlabel('Ітерація')
    ax2.set_ylabel('Найкращий fitness')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Порівняння фінальних результатів
    ax3 = axes[1, 0]
    functions = list(results.keys())
    topologies = list(results[functions[0]].keys())

    x = np.arange(len(functions))
    width = 0.25

    for i, topology in enumerate(topologies):
        values = [results[func][topology]['final_fitness'] for func in functions]
        bars = ax3.bar(x + i * width, values, width, label=topology,
                       color=topology_colors[topology], alpha=0.7)

        # Додати значення на стовпчики
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    ax3.set_title('Фінальні результати', fontweight='bold')
    ax3.set_xlabel('Функція')
    ax3.set_ylabel('Фінальний fitness')
    ax3.set_yscale('log')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(functions)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Швидкість конвергенції (скільки ітерацій до досягнення певного рівня)
    ax4 = axes[1, 1]

    # Для кожної топології та функції знайти ітерацію, коли досягли 95% від фінального результату
    convergence_speeds = {}
    threshold_factor = 1.05  # 5% від фінального значення

    for func_name in results.keys():
        convergence_speeds[func_name] = {}
        for topology in results[func_name].keys():
            convergence = results[func_name][topology]['convergence']
            final_fitness = results[func_name][topology]['final_fitness']
            threshold = final_fitness * threshold_factor

            # Знайти першу ітерацію, коли досягли порогу
            converged_at = len(convergence)  # За замовчуванням - остання ітерація
            for i, fitness in enumerate(convergence):
                if fitness <= threshold:
                    converged_at = i
                    break

            convergence_speeds[func_name][topology] = converged_at

    # Створити стовпчикову діаграму швидкості конвергенції
    x = np.arange(len(functions))
    width = 0.25

    for i, topology in enumerate(topologies):
        values = [convergence_speeds[func][topology] for func in functions]
        bars = ax4.bar(x + i * width, values, width, label=topology,
                       color=topology_colors[topology], alpha=0.7)

        # Додати значення на стовпчики
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{value}', ha='center', va='bottom', fontsize=9)

    ax4.set_title('Швидкість конвергенції (ітерації до 95% результату)', fontweight='bold')
    ax4.set_xlabel('Функція')
    ax4.set_ylabel('Кількість ітерацій')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(functions)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Статистика результатів
    print("\n📊 Детальна статистика результатів:")
    print("=" * 60)

    for func_name in results.keys():
        print(f"\n🎯 Функція {func_name}:")
        print("-" * 40)

        # Сортувати топології за фінальним результатом
        sorted_topologies = sorted(results[func_name].keys(),
                                   key=lambda t: results[func_name][t]['final_fitness'])

        for i, topology in enumerate(sorted_topologies):
            final_fitness = results[func_name][topology]['final_fitness']
            converged_at = convergence_speeds[func_name][topology]
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"

            print(f"  {medal} {topology:8}: "
                  f"Фінальний = {final_fitness:.6f}, "
                  f"Конвергенція = {converged_at:3d} ітерацій")

    # Загальний висновок
    print(f"\n🏆 Загальні висновки:")
    print("-" * 40)

    overall_scores = {topology: 0 for topology in topologies}

    for func_name in results.keys():
        sorted_topologies = sorted(results[func_name].keys(),
                                   key=lambda t: results[func_name][t]['final_fitness'])
        # Додати бали: 1-е місце = 3 бали, 2-е = 2 бали, 3-є = 1 бал
        for i, topology in enumerate(sorted_topologies):
            overall_scores[topology] += (len(sorted_topologies) - i)

    # Вивести загальний рейтинг
    sorted_overall = sorted(overall_scores.keys(), key=lambda t: overall_scores[t], reverse=True)
    for i, topology in enumerate(sorted_overall):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"  {medal} {topology}: {overall_scores[topology]} балів")


if __name__ == "__main__":
    # Запустити демонстрацію
    results = main()

    # Візуалізація (розкоментуйте якщо потрібна)
    visualize_swarm_behavior(results)

    print("\n✅ Демонстрація завершена!")
    print("Децентралізований PSO успішно працює без центрального координатора.")