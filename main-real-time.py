"""
Particle Swarm Optimization (PSO) for 2D Pathfinding with Real-time Visualization
================================================================================

This module implements a PSO algorithm to find the shortest path to a target point in 2D space.
The swarm converges when particles reach within an epsilon threshold of the target.
Includes real-time visualization using Pygame.
"""

import logging
import time
from dataclasses import dataclass
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pygame
from numpy import floating

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PSOConfig:
    n_particles: int = 30
    n_iterations: int = 100
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive coefficient
    c2: float = 1.5  # Social coefficient
    epsilon: float = 0.1  # Swarm convergence threshold (average distance to target)
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-10, 10), (-10, 10))
    v_max: float = 2.0  # Maximum velocity
    delay: float = 0.5  # Delay between iterations in seconds


@dataclass
class VisualizationConfig:
    window_width: int = 800
    window_height: int = 800
    margin: int = 50
    particle_radius: int = 5
    target_radius: int = 10
    fps: int = 60
    colors = {
        'background': (20, 20, 30),
        'particle': (100, 149, 237),  # Cornflower blue
        'particle_best': (255, 215, 0),  # Gold
        'target': (255, 69, 69),  # Red
        'target_area': (255, 69, 69, 50),  # Red with alpha
        'global_best': (50, 205, 50),  # Lime green
        'grid': (50, 50, 60),
        'text': (200, 200, 200),
        'velocity': (150, 150, 255)  # Light blue
    }


class Particle:

    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        self.position = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1])
        ])
        self.velocity = np.random.uniform(-1, 1, 2)
        self.best_position = self.position.copy()
        self.best_distance = float('inf')
        self.trail = []  # Store recent positions for trail effect

    def update_velocity(self, global_best_position: np.ndarray, config: PSOConfig) -> None:
        r1, r2 = np.random.random(2)

        cognitive = config.c1 * r1 * (self.best_position - self.position)
        social = config.c2 * r2 * (global_best_position - self.position)

        self.velocity = config.w * self.velocity + cognitive + social

        # Limit velocity
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > config.v_max:
            self.velocity = (self.velocity / velocity_magnitude) * config.v_max

    def update_position(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        # Store previous position for trail
        self.trail.append(self.position.copy())
        if len(self.trail) > 10:  # Keep only recent positions
            self.trail.pop(0)

        self.position += self.velocity

        # Enforce boundaries
        for i in range(2):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
                self.velocity[i] *= -0.5  # Bounce with damping
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]
                self.velocity[i] *= -0.5


class PygameVisualizer:

    def __init__(self, pso_config: PSOConfig, vis_config: VisualizationConfig):
        self.pso_config = pso_config
        self.vis_config = vis_config

        pygame.init()
        self.screen = pygame.display.set_mode((vis_config.window_width, vis_config.window_height))
        pygame.display.set_caption("PSO Pathfinding - Real-time Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Calculate scaling factors
        self.scale_x = (vis_config.window_width - 2 * vis_config.margin) / (pso_config.bounds[0][1] - pso_config.bounds[0][0])
        self.scale_y = (vis_config.window_height - 2 * vis_config.margin) / (pso_config.bounds[1][1] - pso_config.bounds[1][0])

    def world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        x = int((pos[0] - self.pso_config.bounds[0][0]) * self.scale_x + self.vis_config.margin)
        y = int(self.vis_config.window_height - ((pos[1] - self.pso_config.bounds[1][0]) * self.scale_y + self.vis_config.margin))
        return (x, y)

    def draw_grid(self):
        # Vertical lines
        for i in range(11):
            x = self.vis_config.margin + i * (self.vis_config.window_width - 2 * self.vis_config.margin) // 10
            pygame.draw.line(self.screen, self.vis_config.colors['grid'],
                           (x, self.vis_config.margin),
                           (x, self.vis_config.window_height - self.vis_config.margin), 1)

        # Horizontal lines
        for i in range(11):
            y = self.vis_config.margin + i * (self.vis_config.window_height - 2 * self.vis_config.margin) // 10
            pygame.draw.line(self.screen, self.vis_config.colors['grid'],
                           (self.vis_config.margin, y),
                           (self.vis_config.window_width - self.vis_config.margin, y), 1)

    def draw_particle(self, particle: Particle, is_best: bool = False):
        # Draw trail
        if len(particle.trail) > 1:
            for i in range(len(particle.trail) - 1):
                alpha = int(255 * (i + 1) / len(particle.trail))
                start_pos = self.world_to_screen(particle.trail[i])
                end_pos = self.world_to_screen(particle.trail[i + 1])
                pygame.draw.line(self.screen, (*self.vis_config.colors['particle'][:3], alpha),
                               start_pos, end_pos, 2)

        # Draw velocity vector
        screen_pos = self.world_to_screen(particle.position)
        velocity_end = particle.position + particle.velocity * 0.5
        velocity_screen = self.world_to_screen(velocity_end)
        pygame.draw.line(self.screen, self.vis_config.colors['velocity'],
                        screen_pos, velocity_screen, 2)

        # Draw particle
        color = self.vis_config.colors['particle_best'] if is_best else self.vis_config.colors['particle']
        pygame.draw.circle(self.screen, color, screen_pos, self.vis_config.particle_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, self.vis_config.particle_radius, 1)

    def draw_target(self, target: np.ndarray):
        """Draw target point."""
        target_screen = self.world_to_screen(target)

        # Draw target
        pygame.draw.circle(self.screen, self.vis_config.colors['target'],
                         target_screen, self.vis_config.target_radius)
        pygame.draw.circle(self.screen, (255, 255, 255),
                         target_screen, self.vis_config.target_radius, 2)

    def draw_info(self, iteration: int, best_distance: float, converged: bool, average_distance_to_target: float):
        """Draw information text."""
        texts = [
            f"Iteration: {iteration}",
            f"Best Distance: {best_distance:.4f}",
            f"Swarm Average to Target: {average_distance_to_target:.4f}",
            f"Epsilon: {self.pso_config.epsilon:.4f}",
            f"Status: {'CONVERGED!' if converged else 'Searching...'}"
        ]

        y_offset = 10
        for text in texts:
            color = (50, 255, 50) if converged and "Status" in text else self.vis_config.colors['text']
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 30

    def draw_controls(self):
        """Draw control instructions."""
        controls = [
            "SPACE: Pause/Resume",
            "R: Restart",
            "ESC: Exit"
        ]

        y_offset = self.vis_config.window_height - 80
        for control in controls:
            text_surface = self.small_font.render(control, True, self.vis_config.colors['text'])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 20


class ParticleSwarmOptimizer:

    def __init__(self, target: Tuple[float, float], config: PSOConfig = PSOConfig(),
                 vis_config: VisualizationConfig = VisualizationConfig()):
        self.target = np.array(target)
        self.config = config
        self.vis_config = vis_config
        self.swarm: List[Particle] = []
        self.global_best_position = None
        self.global_best_distance = float('inf')
        self.history = {'iterations': [], 'best_distances': [], 'positions': [], 'average_distance_to_target': []}
        self.visualizer = PygameVisualizer(config, vis_config)
        self.paused = False
        self.converged = False
        self.current_average_to_target = float('inf')

    def _calculate_distance(self, position: np.ndarray) -> floating:
        return np.linalg.norm(position - self.target)

    def _calculate_swarm_centroid(self):
        positions = np.array([particle.position for particle in self.swarm])
        return np.mean(positions, axis=0)

    def _calculate_swarm_average_distance_to_target(self) -> float | floating:
        if not self.swarm:
            return 0.0

        distances = [np.linalg.norm(particle.position - self.target) for particle in self.swarm]
        return np.mean(distances)

    def _initialize_swarm(self) -> None:
        self.swarm = [Particle(self.config.bounds) for _ in range(self.config.n_particles)]

        # Initialize global best
        for particle in self.swarm:
            distance = self._calculate_distance(particle.position)
            particle.best_distance = distance

            if distance < self.global_best_distance:
                self.global_best_distance = distance
                self.global_best_position = particle.position.copy()

        self.current_average_to_target = self._calculate_swarm_average_distance_to_target()
        logger.info(f"Swarm initialized with {self.config.n_particles} particles")
        logger.info(f"Initial average to target: {self.current_average_to_target:.4f}")

    def _update_particle_best(self, particle: Particle) -> None:
        distance = self._calculate_distance(particle.position)

        if distance < particle.best_distance:
            particle.best_distance = distance
            particle.best_position = particle.position.copy()

    def _update_global_best(self, particle: Particle) -> None:
        if particle.best_distance < self.global_best_distance:
            self.global_best_distance = particle.best_distance
            self.global_best_position = particle.best_position.copy()

    def _check_convergence(self) -> bool:
        self.current_average_to_target = self._calculate_swarm_average_distance_to_target()
        return self.current_average_to_target < self.config.epsilon

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.__init__(self.target.tolist(), self.config, self.vis_config)
                    return True
        return True

    def _draw_frame(self, iteration: int):
        self.visualizer.screen.fill(self.vis_config.colors['background'])
        self.visualizer.draw_grid()

        # Draw target
        self.visualizer.draw_target(self.target)

        # Draw swarm centroid
        if self.swarm:
            positions = np.array([particle.position for particle in self.swarm])
            centroid = np.mean(positions, axis=0)
            centroid_screen = self.visualizer.world_to_screen(centroid)
            pygame.draw.circle(self.visualizer.screen, (255, 165, 0), centroid_screen, 5, 2)  # Orange circle

        # Draw global best position
        if self.global_best_position is not None:
            gb_screen = self.visualizer.world_to_screen(self.global_best_position)
            pygame.draw.circle(self.visualizer.screen, self.vis_config.colors['global_best'],
                             gb_screen, 8, 2)

        # Draw particles
        for particle in self.swarm:
            is_best = np.array_equal(particle.position, self.global_best_position)
            self.visualizer.draw_particle(particle, is_best)

        # Draw info and controls
        self.visualizer.draw_info(iteration, self.global_best_distance, self.converged, self.current_average_to_target)
        self.visualizer.draw_controls()

        pygame.display.flip()

    def optimize_with_visualization(self) -> Tuple[np.ndarray, float, int]:
        self._initialize_swarm()
        iteration = 0
        last_update_time = time.time()

        running = True
        while running and iteration < self.config.n_iterations and not self.converged:
            # Handle events
            if not self._handle_events():
                break

            # Draw current state
            self._draw_frame(iteration)
            self.visualizer.clock.tick(self.vis_config.fps)

            # Update particles only if not paused and after delay
            current_time = time.time()
            if not self.paused and (current_time - last_update_time) >= self.config.delay:
                # Store current positions for history
                current_positions = [p.position.copy() for p in self.swarm]
                self.history['positions'].append(current_positions)
                self.history['iterations'].append(iteration)
                self.history['best_distances'].append(self.global_best_distance)
                self.history['average_distance_to_target'].append(self.current_average_to_target)

                # Update each particle
                for particle in self.swarm:
                    particle.update_velocity(self.global_best_position, self.config)
                    particle.update_position(self.config.bounds)
                    self._update_particle_best(particle)
                    self._update_global_best(particle)

                # Check convergence
                if self._check_convergence():
                    self.converged = True
                    logger.info(f"Swarm converged at iteration {iteration + 1}")
                    logger.info(f"Final average distance to target: {self.current_average_to_target:.4f}")
                    logger.info(f"Best distance to target: {self.global_best_distance:.4f}")

                # Log progress
                if (iteration + 1) % 10 == 0:
                    logger.info(f"Iteration {iteration + 1}: Best distance = {self.global_best_distance:.4f}, Average_distance_to_target = {self.current_average_to_target:.4f}")

                iteration += 1
                last_update_time = current_time

        # Keep window open after convergence
        if self.converged:
            while running:
                if not self._handle_events():
                    break
                self._draw_frame(iteration - 1)
                self.visualizer.clock.tick(self.vis_config.fps)

        pygame.quit()

        if not self.converged:
            logger.warning(f"Maximum iterations reached. Best distance: {self.global_best_distance:.4f}")

        return self.global_best_position, self.global_best_distance, iteration

    def visualize_convergence(self) -> None:
        if not self.history['iterations']:
            logger.warning("No history data to visualize")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot best distance
        ax1.plot(self.history['iterations'], self.history['best_distances'], 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Distance to Target')
        ax1.set_title('PSO Best Distance History')
        ax1.grid(True, alpha=0.3)

        # Plot average distance to target
        ax2.plot(self.history['iterations'], self.history['average_distance_to_target'], 'g-', linewidth=2)
        ax2.axhline(y=self.config.epsilon, color='r', linestyle='--',
                    label=f'Epsilon = {self.config.epsilon}')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Swarm Average Distance to Target')
        ax2.set_title('PSO Swarm Average Distance to Target History')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define target point
    target = (7.5, 8.0)

    # Create custom configuration
    pso_config = PSOConfig(
        n_particles=20,
        n_iterations=200,
        w=0.8,
        c1=1.8,
        c2=1.8,
        epsilon=0.5,  # Swarm converges when particles are within 0.5 units on average from centroid
        bounds=((-10, 10), (-10, 10)),
        v_max=2.5,
        delay=0.05  # Half second delay for better visualization
    )

    # Create visualization configuration
    vis_config = VisualizationConfig(
        window_width=800,
        window_height=800,
        particle_radius=6
    )

    # Create and run optimizer with visualization
    optimizer = ParticleSwarmOptimizer(target, pso_config, vis_config)
    best_position, best_distance, iterations_used = optimizer.optimize_with_visualization()

    # Print results
    print(f"\nOptimization Results:")
    print(f"Target: {target}")
    print(f"Best position found: ({best_position[0]:.4f}, {best_position[1]:.4f})")
    print(f"Distance to target: {best_distance:.4f}")
    print(f"Iterations used: {iterations_used}")
    print(f"Swarm converged: {optimizer.converged} (average distance to target < epsilon)")

    # Show convergence plot
    optimizer.visualize_convergence()


if __name__ == "__main__":
    main()