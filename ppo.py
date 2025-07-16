import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ParticleSwarmRL:
    def __init__(self, n_particles, n_dims, objective_function):
        # Параметри PSO
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.f = objective_function

        # Ініціалізація рою
        self.positions = np.random.uniform(-10, 10, (n_particles, n_dims))
        self.velocities = np.random.uniform(-1, 1, (n_particles, n_dims))
        self.p_best = self.positions.copy()
        self.p_best_scores = np.array([self.f(p) for p in self.positions])
        self.g_best = self.p_best[np.argmin(self.p_best_scores)]

        # Лічильники стагнації
        self.stagnation_counters = np.zeros(n_particles)

        # Параметри викидів
        self.outlier_threshold_dist = 3.0  # 3 стандартних відхилення
        self.outlier_threshold_vel = 2.0
        self.stagnation_window = 10

        # Статус частинок (активні/деактивовані)
        self.active = np.ones(n_particles, dtype=bool)


class PSOEnvironment(gym.Env):
    def __init__(self, swarm, max_steps=100):
        super().__init__()
        self.swarm = swarm
        self.max_steps = max_steps
        self.current_step = 0

        # Визначаємо простори
        state_dim = swarm.n_dims + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,)
        )
        self.action_space = spaces.Discrete(3)  # 3 можливі дії

    def step(self, action):
        # Знаходимо викидів
        outliers = self.swarm.detect_outliers()

        # Застосовуємо дію до першого викиду (спрощено)
        if outliers:
            self.swarm.apply_action(outliers[0], action)

        # Оновлюємо PSO
        self.swarm.update_pso()

        # Обчислюємо винагороду
        reward = self.calculate_reward()

        # Перевіряємо завершення
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.swarm.get_state(), reward, done, False, {}