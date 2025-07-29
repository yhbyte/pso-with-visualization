import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class ParticleSwarmRL:
    def __init__(self, n_particles, n_dims, objective_function, bounds=(-10, 10)):
        # PSO parameters
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.f = objective_function
        self.bounds = bounds

        # PSO coefficients
        self.w = 0.7298  # inertia weight
        self.c1 = 1.49618  # cognitive parameter
        self.c2 = 1.49618  # social parameter

        # Initialize swarm
        self.positions = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dims))
        self.velocities = np.random.uniform(-1, 1, (n_particles, n_dims))
        self.p_best = self.positions.copy()
        self.p_best_scores = np.array([self.f(p) for p in self.positions])
        self.g_best_idx = np.argmin(self.p_best_scores)
        self.g_best = self.p_best[self.g_best_idx].copy()
        self.g_best_score = self.p_best_scores[self.g_best_idx]

        # Stagnation counters
        self.stagnation_counters = np.zeros(n_particles)

        # Outlier parameters
        self.outlier_threshold_dist = 3.0  # 3 standard deviations
        self.outlier_threshold_vel = 2.0
        self.stagnation_window = 10

        # Particle status (active/deactivated)
        self.active = np.ones(n_particles, dtype=bool)
        self.deactivation_timers = np.zeros(n_particles)

        # History for visualization
        self.history = {
            'positions': [self.positions.copy()],
            'g_best_scores': [self.g_best_score],
            'n_outliers': [0],
            'outlier_indices': [[]]
        }

    def detect_outliers(self):
        """Detect outliers based on three criteria"""
        outliers = []

        if np.sum(self.active) < 2:  # Need at least 2 active particles
            return outliers

        # Center of mass and statistics
        active_positions = self.positions[self.active]
        center_of_mass = np.mean(active_positions, axis=0)
        distances = np.linalg.norm(self.positions - center_of_mass, axis=1)

        # Calculate standard deviation only for active particles
        if len(active_positions) > 1:
            std_dist = np.std(distances[self.active])
        else:
            std_dist = 1.0

        active_velocities = self.velocities[self.active]
        if len(active_velocities) > 0:
            avg_velocity = np.mean(np.linalg.norm(active_velocities, axis=1))
        else:
            avg_velocity = 1.0

        for i in range(self.n_particles):
            if not self.active[i]:
                continue

            # Criterion 1: distance
            if std_dist > 0 and distances[i] > self.outlier_threshold_dist * std_dist:
                outliers.append(i)
                continue

            # Criterion 2: velocity
            if avg_velocity > 0 and np.linalg.norm(self.velocities[i]) > self.outlier_threshold_vel * avg_velocity:
                outliers.append(i)
                continue

            # Criterion 3: stagnation
            if self.stagnation_counters[i] > self.stagnation_window:
                outliers.append(i)

        return outliers

    def get_state(self):
        """Get global state of the swarm for RL agent"""
        active_mask = self.active
        n_active = np.sum(active_mask)

        if n_active == 0:
            # Return zero state if no active particles
            return np.zeros(self.n_dims + 4)

        active_positions = self.positions[active_mask]
        center_of_mass = np.mean(active_positions, axis=0)

        if n_active > 1:
            position_std = np.std(active_positions)
        else:
            position_std = 0.0

        active_velocities = self.velocities[active_mask]
        avg_velocity = np.mean(np.linalg.norm(active_velocities, axis=1))

        best_score = self.g_best_score
        outlier_ratio = len(self.detect_outliers()) / n_active if n_active > 0 else 0

        # Form state vector
        state = np.concatenate([
            center_of_mass,  # D values
            [position_std],  # 1 value
            [avg_velocity],  # 1 value
            [best_score],  # 1 value
            [outlier_ratio]  # 1 value
        ])

        return state.astype(np.float32)

    def apply_action(self, particle_id, action):
        """Apply RL agent's action to a particle"""
        if action == 0:  # Ignore
            pass
        elif action == 1:  # Return to center
            active_positions = self.positions[self.active]
            if len(active_positions) > 0:
                center = np.mean(active_positions, axis=0)
                self.positions[particle_id] = center
                self.velocities[particle_id] = np.zeros(self.n_dims)
                self.stagnation_counters[particle_id] = 0
        elif action == 2:  # Deactivate
            self.active[particle_id] = False
            self.deactivation_timers[particle_id] = 5  # Deactivate for 5 iterations

    def update_pso(self):
        """Standard PSO update for active particles"""
        for i in range(self.n_particles):
            # Handle deactivation timers
            if self.deactivation_timers[i] > 0:
                self.deactivation_timers[i] -= 1
                if self.deactivation_timers[i] == 0:
                    self.active[i] = True

            if not self.active[i]:
                continue

            # Random coefficients
            r1 = np.random.rand(self.n_dims)
            r2 = np.random.rand(self.n_dims)

            # Update velocity
            cognitive = self.c1 * r1 * (self.p_best[i] - self.positions[i])
            social = self.c2 * r2 * (self.g_best - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

            # Update position
            self.positions[i] += self.velocities[i]

            # Boundary handling
            self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

            # Evaluate
            score = self.f(self.positions[i])

            # Update personal best
            if score < self.p_best_scores[i]:
                self.p_best[i] = self.positions[i].copy()
                self.p_best_scores[i] = score
                self.stagnation_counters[i] = 0
            else:
                self.stagnation_counters[i] += 1

            # Update global best
            if score < self.g_best_score:
                self.g_best = self.positions[i].copy()
                self.g_best_score = score
                self.g_best_idx = i

    def step(self):
        """Single step of the algorithm"""
        self.update_pso()

        # Update history
        self.history['positions'].append(self.positions.copy())
        self.history['g_best_scores'].append(self.g_best_score)
        outliers = self.detect_outliers()
        self.history['n_outliers'].append(len(outliers))
        self.history['outlier_indices'].append(outliers)


class PSO_Environment(gym.Env):
    """Gymnasium environment for PSO with RL"""

    def __init__(self, n_particles=30, n_dims=2, max_steps=100, objective_function=None):
        super().__init__()

        # Default objective function (Rastrigin)
        if objective_function is None:
            def rastrigin(x):
                A = 10
                n = len(x)
                return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

            objective_function = rastrigin

        self.n_particles = n_particles
        self.n_dims = n_dims
        self.max_steps = max_steps
        self.objective_function = objective_function

        # Initialize swarm
        self.swarm = ParticleSwarmRL(n_particles, n_dims, objective_function)
        self.current_step = 0

        # Previous state for reward calculation
        self.prev_best_score = self.swarm.g_best_score
        self.prev_position_std = self._get_position_std()
        self.prev_n_outliers = 0

        # Define spaces
        state_dim = n_dims + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 3 possible actions

    def _get_position_std(self):
        """Calculate standard deviation of active particles"""
        active_positions = self.swarm.positions[self.swarm.active]
        if len(active_positions) > 1:
            return np.std(active_positions)
        return 0.0

    def calculate_reward(self):
        """Calculate reward based on multiple criteria"""
        # Current metrics
        current_best_score = self.swarm.g_best_score
        current_position_std = self._get_position_std()
        current_n_outliers = len(self.swarm.detect_outliers())
        n_active = np.sum(self.swarm.active)

        # Reward components
        r_improve = 0.0
        r_cohesion = 0.0
        r_efficiency = 0.0

        # 1. Improvement reward (most important)
        if current_best_score < self.prev_best_score:
            improvement = self.prev_best_score - current_best_score
            r_improve = 10.0 * improvement  # Scale appropriately

        # 2. Cohesion reward
        if current_position_std < self.prev_position_std:
            r_cohesion = 1.0 * (self.prev_position_std - current_position_std)

        # 3. Efficiency penalty
        r_efficiency = -0.5 * current_n_outliers

        # Total reward
        reward = r_improve + r_cohesion + r_efficiency

        # Update previous values
        self.prev_best_score = current_best_score
        self.prev_position_std = current_position_std
        self.prev_n_outliers = current_n_outliers

        return reward

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        # Reinitialize swarm
        self.swarm = ParticleSwarmRL(
            self.n_particles, self.n_dims, self.objective_function
        )
        self.current_step = 0

        # Reset previous values
        self.prev_best_score = self.swarm.g_best_score
        self.prev_position_std = self._get_position_std()
        self.prev_n_outliers = 0

        return self.swarm.get_state(), {}

    def step(self, action):
        """Execute one step"""
        # Find outliers
        outliers = self.swarm.detect_outliers()

        # Apply action to first outlier (simplified)
        if outliers:
            self.swarm.apply_action(outliers[0], action)

        # Update PSO
        self.swarm.step()

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get new state
        state = self.swarm.get_state()

        # Additional info
        info = {
            'best_score': self.swarm.g_best_score,
            'n_outliers': len(outliers),
            'n_active': np.sum(self.swarm.active)
        }

        return state, reward, terminated, truncated, info


class PSO_RL_Visualizer:
    """Real-time visualization for PSO-RL"""

    def __init__(self, env):
        self.env = env
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # For 2D visualization
        if env.n_dims == 2:
            self.scatter = self.ax1.scatter([], [], c='blue', s=50)
            self.outlier_scatter = self.ax1.scatter([], [], c='red', s=100, marker='x')
            self.best_scatter = self.ax1.scatter([], [], c='green', s=200, marker='*')

            bounds = env.swarm.bounds
            self.ax1.set_xlim(bounds[0], bounds[1])
            self.ax1.set_ylim(bounds[0], bounds[1])
            self.ax1.set_xlabel('X')
            self.ax1.set_ylabel('Y')
            self.ax1.set_title('Particle Positions')

        # Convergence plot
        self.line_best, = self.ax2.plot([], [], 'b-', label='Best Score')
        self.line_outliers, = self.ax2.plot([], [], 'r--', label='# Outliers')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Value')
        self.ax2.set_title('Convergence')
        self.ax2.legend()
        self.ax2.grid(True)

    def update(self, frame):
        """Update visualization"""
        swarm = self.env.swarm

        # Update particle positions (only for 2D)
        if self.env.n_dims == 2:
            active_mask = swarm.active
            active_positions = swarm.positions[active_mask]

            # Regular particles
            self.scatter.set_offsets(active_positions)

            # Outliers
            outlier_indices = swarm.detect_outliers()
            if outlier_indices:
                outlier_positions = swarm.positions[outlier_indices]
                self.outlier_scatter.set_offsets(outlier_positions)
            else:
                self.outlier_scatter.set_offsets(np.empty((0, 2)))

            # Best position
            self.best_scatter.set_offsets([swarm.g_best])

        # Update convergence plot
        iterations = range(len(swarm.history['g_best_scores']))
        self.line_best.set_data(iterations, swarm.history['g_best_scores'])

        # Normalize outlier count for visualization
        max_outliers = max(swarm.history['n_outliers']) if swarm.history['n_outliers'] else 1
        if max_outliers > 0:
            normalized_outliers = [n / max_outliers * max(swarm.history['g_best_scores'])
                                   for n in swarm.history['n_outliers']]
        else:
            normalized_outliers = swarm.history['n_outliers']

        self.line_outliers.set_data(iterations, normalized_outliers)

        # Adjust axes
        if len(iterations) > 1:
            self.ax2.set_xlim(0, len(iterations))
            y_min = min(min(swarm.history['g_best_scores']), 0)
            y_max = max(swarm.history['g_best_scores'])
            self.ax2.set_ylim(y_min - 0.1 * abs(y_max - y_min),
                              y_max + 0.1 * abs(y_max - y_min))

        return self.scatter, self.outlier_scatter, self.best_scatter, self.line_best, self.line_outliers


def train_pso_rl(n_episodes=100, visualize=False):
    """Train PSO-RL agent"""
    # Create environment
    env = PSO_Environment(n_particles=30, n_dims=2, max_steps=100)

    # Wrap environment for stable-baselines3
    env = DummyVecEnv([lambda: env])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    # Train
    print("Training PPO agent...")
    model.learn(total_timesteps=10000)

    # Save model
    model.save("pso_rl_model")
    print("Model saved!")

    return model


def test_pso_rl(model=None, visualize=True):
    """Test PSO-RL with visualization"""
    # Create test environment
    env = PSO_Environment(n_particles=20, n_dims=2, max_steps=200)

    if visualize:
        visualizer = PSO_RL_Visualizer(env)
        plt.ion()

    # Reset environment
    obs, _ = env.reset()

    for step in range(env.max_steps):
        # Get action
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            # Random action for comparison
            action = env.action_space.sample()

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Update visualization
        if visualize:
            visualizer.update(step)
            plt.pause(0.01)

        if terminated:
            break

    print(f"\nFinal Results:")
    print(f"Best score: {env.swarm.g_best_score:.6f}")
    print(f"Best position: {env.swarm.g_best}")
    print(f"Active particles: {np.sum(env.swarm.active)}/{env.n_particles}")

    if visualize:
        plt.ioff()
        plt.show()

    return env.swarm


if __name__ == "__main__":
    # Option 1: Train and test
    model = train_pso_rl(n_episodes=10000)
    test_pso_rl(model, visualize=True)

    # Option 2: Just test with random actions (no training)
    # print("Testing PSO with random outlier management...")
    # test_pso_rl(model=None, visualize=True)

    # Option 3: Train first, then test with visualization
    # print("Training PSO-RL agent...")
    # model = train_pso_rl()
    # print("\nTesting trained agent...")
    # test_pso_rl(model, visualize=True)