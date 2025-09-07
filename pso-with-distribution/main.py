# pso_coverage.py
# ----------------
# Модель покриття території сенсорами + пошук розташування методом рою часток (PSO)
# Мінімальні залежності: numpy, matplotlib
# Запуск: python pso_coverage.py
# (опційно додай параметри CLI: --sensors 25 --radius 120 --swarm 40 --iters 120)

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# ============================== Конфігурація задачі ==============================

@dataclass
class AreaConfig:
    """Розміри прямокутної області (локальні координати/метри)."""
    xmin: float = 0.0
    ymin: float = 0.0
    xmax: float = 1000.0
    ymax: float = 1000.0

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass
class ObstacleRect:
    """Прямокутна перешкода: тут сенсор ставити не можна (але покриття допускається)."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)


@dataclass
class ProblemConfig:
    """Параметри задачі покриття."""
    area: AreaConfig = field(default_factory=AreaConfig)
    obstacles: List[ObstacleRect] = field(default_factory=lambda: [ObstacleRect(400, 450, 600, 650)])
    num_sensors: int = 22                # кількість сенсорів
    sensor_radius: float = 120.0         # радіус покриття сенсора
    grid_res: int = 85                   # щільність сітки для оцінки покриття (баланс точність/швидкість)
    min_separation: float = 0.6          # частка від радіуса для "мінімальної відстані" між сенсорами


# ================================ Параметри PSO =================================

@dataclass
class PSOConfig:
    swarm_size: int = 28                 # кількість частинок (рішень) у рої
    iterations: int = 90                 # кількість ітерацій
    inertia: float = 0.72                # інерція швидкості (w)
    c1: float = 1.6                      # когнітивна вага
    c2: float = 1.6                      # соціальна вага
    vmax: float = 50.0                   # обмеження модуля швидкості по координаті


# ============================== Обчислення якості ===============================

class CoverageEvaluator:
    """Оцінює якість розташування: частку покриття + штрафи за межі/перешкоди/надмірне скупчення."""

    def __init__(self, pcfg: ProblemConfig):
        self.pcfg = pcfg
        gx = np.linspace(pcfg.area.xmin, pcfg.area.xmax, pcfg.grid_res)
        gy = np.linspace(pcfg.area.ymin, pcfg.area.ymax, pcfg.grid_res)
        self.xx, self.yy = np.meshgrid(gx, gy)
        self.grid_points = np.stack([self.xx.ravel(), self.yy.ravel()], axis=1)

    def coverage_fraction(self, sensors_xy: np.ndarray) -> float:
        """Частка площі (за сіткою), що потрапляє в радіус хоча б одного сенсора."""
        if sensors_xy.size == 0:
            return 0.0
        gp = self.grid_points  # (M,2)
        d2 = (gp[:, None, 0] - sensors_xy[None, :, 0]) ** 2 + (gp[:, None, 1] - sensors_xy[None, :, 1]) ** 2
        covered = (d2.min(axis=1) <= self.pcfg.sensor_radius ** 2)
        return float(covered.mean())

    def boundary_penalty(self, sensors_xy: np.ndarray) -> float:
        """Штраф за сенсори поза межами області або всередині перешкод."""
        a = self.pcfg.area
        x, y = sensors_xy[:, 0], sensors_xy[:, 1]
        out = (x < a.xmin) | (x > a.xmax) | (y < a.ymin) | (y > a.ymax)
        penalty = out.sum() * 3.0
        if self.pcfg.obstacles:
            in_obs = np.zeros_like(out)
            for ob in self.pcfg.obstacles:
                in_obs |= ob.contains(x, y)
            penalty += in_obs.sum() * 3.0
        return float(penalty)

    def spacing_penalty(self, sensors_xy: np.ndarray) -> float:
        """Штраф за надто близькі сенсори — стимулює рівномірний розподіл."""
        n = sensors_xy.shape[0]
        if n < 2:
            return 0.0
        diff = sensors_xy[None, :, :] - sensors_xy[:, None, :]
        d = np.sqrt((diff ** 2).sum(axis=2) + 1e-9)
        mask = ~np.eye(n, dtype=bool)
        d = d[mask]
        min_sep = self.pcfg.sensor_radius * self.pcfg.min_separation
        close = d < min_sep
        if not np.any(close):
            return 0.0
        # чим ближче до "заборонено близько", тим більший штраф
        return float(((min_sep - d[close]) / max(min_sep, 1e-6)).sum())

    def fitness(self, sensors_xy: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Мінімізуємо: F = -coverage + w1*spacing + w2*boundary"""
        cov = self.coverage_fraction(sensors_xy)
        sp = self.spacing_penalty(sensors_xy)
        bp = self.boundary_penalty(sensors_xy)
        w1, w2 = 0.10, 2.5
        score = -cov + w1 * sp + w2 * bp
        return score, {"coverage": cov, "spacing_penalty": sp, "boundary_penalty": bp}


# ===================================== PSO =====================================

class PSO:
    """Класична схема PSO над вектором рішень (x1,y1,...,xN,yN)."""

    def __init__(self, pcfg: ProblemConfig, pso_cfg: PSOConfig, evaluator: CoverageEvaluator, seed: int = 7):
        self.pcfg = pcfg
        self.cfg = pso_cfg
        self.eval = evaluator
        self.dim = pcfg.num_sensors * 2
        self.rng = np.random.default_rng(seed)
        self.lb = np.array([pcfg.area.xmin, pcfg.area.ymin] * pcfg.num_sensors)
        self.ub = np.array([pcfg.area.xmax, pcfg.area.ymax] * pcfg.num_sensors)

    def _vec_to_xy(self, vec: np.ndarray) -> np.ndarray:
        return vec.reshape(self.pcfg.num_sensors, 2)

    def _push_out_of_obstacles(self, vec: np.ndarray) -> np.ndarray:
        """М’яко виштовхує сенсори, що випадково потрапили в перешкоди, до найближчого ребра."""
        xy = self._vec_to_xy(vec)
        for i in range(xy.shape[0]):
            x, y = xy[i, 0], xy[i, 1]
            for ob in self.pcfg.obstacles:
                if ob.contains(np.array([x]), np.array([y]))[0]:
                    dx = min(abs(x - ob.xmin), abs(ob.xmax - x))
                    dy = min(abs(y - ob.ymin), abs(ob.ymax - y))
                    if dx < dy:
                        xy[i, 0] = ob.xmin - 1.0 if abs(x - ob.xmin) < abs(ob.xmax - x) else ob.xmax + 1.0
                    else:
                        xy[i, 1] = ob.ymin - 1.0 if abs(y - ob.ymin) < abs(ob.ymax - y) else ob.ymax + 1.0
                    xy[i, 0] = float(np.clip(xy[i, 0], self.pcfg.area.xmin, self.pcfg.area.xmax))
                    xy[i, 1] = float(np.clip(xy[i, 1], self.pcfg.area.ymin, self.pcfg.area.ymax))
        return xy.ravel()

    def _clip_to_bounds(self, vec: np.ndarray) -> np.ndarray:
        vec = np.clip(vec, self.lb, self.ub)
        return self._push_out_of_obstacles(vec)

    def solve(self) -> Dict[str, object]:
        # 1) Ініціалізація рою
        swarm_pos = self.lb + self.rng.random(self.dim * self.cfg.swarm_size).reshape(self.cfg.swarm_size, self.dim) * (self.ub - self.lb)
        for i in range(self.cfg.swarm_size):
            swarm_pos[i] = self._clip_to_bounds(swarm_pos[i])
        swarm_vel = self.rng.uniform(-10.0, 10.0, size=(self.cfg.swarm_size, self.dim))

        # (для звіту/порівняння) збережемо випадкове початкове рішення
        initial_vec = swarm_pos[0].copy()
        initial_xy = self._vec_to_xy(initial_vec)
        initial_cov = self.eval.coverage_fraction(initial_xy)

        # 2) Персональні та глобальна найкращі точки
        pbest_pos = swarm_pos.copy()
        pbest_val = np.empty(self.cfg.swarm_size)
        for i in range(self.cfg.swarm_size):
            f, _ = self.eval.fitness(self._vec_to_xy(pbest_pos[i]))
            pbest_val[i] = f
        gbest_idx = int(np.argmin(pbest_val))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = float(pbest_val[gbest_idx])

        # 3) Основний цикл
        history = []
        for it in range(1, self.cfg.iterations + 1):
            for i in range(self.cfg.swarm_size):
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                cognitive = self.cfg.c1 * r1 * (pbest_pos[i] - swarm_pos[i])
                social = self.cfg.c2 * r2 * (gbest_pos - swarm_pos[i])
                swarm_vel[i] = self.cfg.inertia * swarm_vel[i] + cognitive + social
                swarm_vel[i] = np.clip(swarm_vel[i], -self.cfg.vmax, self.cfg.vmax)
                swarm_pos[i] = self._clip_to_bounds(swarm_pos[i] + swarm_vel[i])

                f, _ = self.eval.fitness(self._vec_to_xy(swarm_pos[i]))
                if f < pbest_val[i]:
                    pbest_val[i] = f
                    pbest_pos[i] = swarm_pos[i].copy()

            best_idx = int(np.argmin(pbest_val))
            if pbest_val[best_idx] < gbest_val:
                gbest_val = float(pbest_val[best_idx])
                gbest_pos = pbest_pos[best_idx].copy()

            cov = self.eval.coverage_fraction(self._vec_to_xy(gbest_pos))
            history.append({"iteration": it, "gbest_f": gbest_val, "coverage": cov})

        return {
            "initial_xy": initial_xy,
            "initial_cov": initial_cov,
            "best_vec": gbest_pos,
            "best_xy": self._vec_to_xy(gbest_pos),
            "best_f": gbest_val,
            "history": history,
        }


# ================================= Візуалізація =================================

def plot_layout(pcfg: ProblemConfig, sensors_xy: np.ndarray, title: str, out_path: str) -> None:
    """Малює межі, перешкоди, диски покриття та точки сенсорів; зберігає PNG і показує вікно."""
    fig, ax = plt.subplots(figsize=(7.8, 7.8))
    # Кордон області
    ax.add_patch(Rectangle((pcfg.area.xmin, pcfg.area.ymin), pcfg.area.width, pcfg.area.height, fill=False, linewidth=2))
    # Перешкоди
    for ob in pcfg.obstacles:
        ax.add_patch(Rectangle((ob.xmin, ob.ymin), ob.xmax - ob.xmin, ob.ymax - ob.ymin, alpha=0.25))
    # Диски покриття
    for (x, y) in sensors_xy:
        ax.add_patch(Circle((x, y), pcfg.sensor_radius, alpha=0.12))
    # Точки сенсорів
    ax.scatter(sensors_xy[:, 0], sensors_xy[:, 1], s=28)

    ax.set_xlim(pcfg.area.xmin - 5, pcfg.area.xmax + 5)
    ax.set_ylim(pcfg.area.ymin - 5, pcfg.area.ymax + 5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.show()


def save_coordinates_csv(path: str, sensors_xy: np.ndarray) -> None:
    """Зберігає список сенсорів і їх координати у CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sensor_id", "x", "y"])
        for i, (x, y) in enumerate(sensors_xy, start=1):
            w.writerow([i, f"{x:.3f}", f"{y:.3f}"])


# ==================================== CLI / main ====================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSO-оптимізація розміщення сенсорів для максимального покриття.")
    p.add_argument("--sensors", type=int, default=22, help="Кількість сенсорів (default: 22)")
    p.add_argument("--radius", type=float, default=120.0, help="Радіус покриття сенсора (default: 120)")
    p.add_argument("--swarm", type=int, default=28, help="Розмір рою (default: 28)")
    p.add_argument("--iters", type=int, default=200, help="Кількість ітерацій PSO (default: 90)")
    p.add_argument("--grid", type=int, default=85, help="Роздільна здатність сітки оцінки (default: 85)")
    p.add_argument("--seed", type=int, default=9, help="Зерно генератора випадковостей (default: 7)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Налаштування задачі
    pcfg = ProblemConfig(
        area=AreaConfig(0, 0, 1000, 1000),
        obstacles=[ObstacleRect(400, 450, 600, 650)],
        num_sensors=args.sensors,
        sensor_radius=args.radius,
        grid_res=args.grid,
        min_separation=0.6,
    )

    # 2) Оцінювач та PSO
    evaluator = CoverageEvaluator(pcfg)
    pso_cfg = PSOConfig(
        swarm_size=args.swarm,
        iterations=args.iters,
        inertia=0.72,
        c1=1.6,
        c2=1.6,
        vmax=50.0,
    )
    solver = PSO(pcfg, pso_cfg, evaluator, seed=args.seed)

    # 3) Пошук
    result = solver.solve()
    init_xy = result["initial_xy"]
    best_xy = result["best_xy"]

    # 4) Метрики
    init_cov = evaluator.coverage_fraction(init_xy)
    best_score, best_parts = evaluator.fitness(best_xy)

    print("=== SUMMARY ===")
    print(f"Initial coverage:  {init_cov*100:.2f}%")
    print(f"Final coverage:    {best_parts['coverage']*100:.2f}%")
    print(f"Spacing penalty:   {best_parts['spacing_penalty']:.3f}")
    print(f"Boundary penalty:  {best_parts['boundary_penalty']:.3f}")
    print(f"Final fitness:     {best_score:.6f}")

    # 5) Візуалізації
    initial_img = "initial_layout.png"
    optimized_img = "optimized_layout.png"
    plot_layout(pcfg, init_xy, f"Initial layout (coverage ≈ {init_cov*100:.1f}%)", initial_img)
    plot_layout(pcfg, best_xy, f"Optimized layout (coverage ≈ {best_parts['coverage']*100:.1f}%)", optimized_img)

    # 6) CSV-и
    coords_csv = "sensors_optimized.csv"
    hist_csv = "pso_history.csv"
    save_coordinates_csv(coords_csv, best_xy)
    # історію теж збережемо
    import csv as _csv
    with open(hist_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["iteration", "gbest_f", "coverage"])
        w.writeheader()
        for row in result["history"]:
            w.writerow({"iteration": row["iteration"],
                        "gbest_f": f"{row['gbest_f']:.6f}",
                        "coverage": f"{row['coverage']:.6f}"})

    print("\nArtifacts saved:")
    print(f"- {initial_img}")
    print(f"- {optimized_img}")
    print(f"- {coords_csv}")
    print(f"- {hist_csv}")


if __name__ == "__main__":
    main()
