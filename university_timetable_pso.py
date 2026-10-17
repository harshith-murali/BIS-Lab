"""Reproducible university timetable optimization using discrete-decoded PSO."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COURSES = ['CS101', 'CS102', 'CS103', 'EE101', 'EE102', 'MATH101']
FACULTY = ['Dr. Sharma', 'Dr. Patel', 'Dr. Sharma', 'Dr. Iyer', 'Dr. Iyer', 'Dr. Nair']
ROOMS = ['Room 101', 'Room 102', 'Lab A']
ROOM_TYPES = ['Lecture', 'Lecture', 'Lab']
REQUIRED_TYPES = ['Lecture', 'Lab', 'Lecture', 'Lecture', 'Lab', 'Lecture']
TIMESLOTS = ['Mon 9:00', 'Mon 11:00', 'Wed 9:00', 'Wed 11:00', 'Fri 9:00']
DAYS = ['Mon', 'Wed', 'Fri']
HARD_WEIGHT = 1000


@dataclass(frozen=True)
class Score:
    faculty_clashes: int
    room_clashes: int
    room_type_errors: int
    faculty_gaps: int
    day_imbalance: int

    @property
    def hard(self) -> int:
        return self.faculty_clashes + self.room_clashes + self.room_type_errors

    @property
    def cost(self) -> int:
        # For this fixed instance the soft penalties are always below 1000,
        # so resolving a hard constraint always takes priority.
        return HARD_WEIGHT * self.hard + self.faculty_gaps + self.day_imbalance


def decode(position: np.ndarray) -> np.ndarray:
    """Round continuous coordinates into (room, timeslot) indices per course."""
    upper = np.array([len(ROOMS) - 1, len(TIMESLOTS) - 1])
    return np.rint(np.clip(position.reshape(len(COURSES), 2), 0, upper)).astype(int)


def evaluate(schedule: np.ndarray) -> Score:
    """Count pairwise clashes, incorrect room types and compactness penalties."""
    faculty_clashes = room_clashes = type_errors = 0
    faculty_periods: dict[tuple[str, str], set[int]] = {}
    daily_load = Counter({day: 0 for day in DAYS})
    for i, (room, slot) in enumerate(schedule):
        type_errors += ROOM_TYPES[room] != REQUIRED_TYPES[i]
        day = TIMESLOTS[slot].split()[0]
        daily_load[day] += 1
        # Slot order within each day, not the global slot index, defines gaps.
        period = sum(t.split()[0] == day for t in TIMESLOTS[:slot])
        faculty_periods.setdefault((FACULTY[i], day), set()).add(period)
        for j in range(i):
            other_room, other_slot = schedule[j]
            if slot == other_slot:
                faculty_clashes += FACULTY[i] == FACULTY[j]
                room_clashes += room == other_room
    gaps = sum(max(p) - min(p) + 1 - len(p) for p in faculty_periods.values())
    # Three days, six classes: the target is two classes per day.
    imbalance = sum((load - len(COURSES) / len(DAYS)) ** 2 for load in daily_load.values())
    return Score(int(faculty_clashes), int(room_clashes), int(type_errors), gaps, int(imbalance))


def optimize(particles: int = 60, iterations: int = 250, seed: int = 42):
    """PSO with personal/global bests and bounded continuous velocities.

    Each particle has two coordinates per course. Fitness is evaluated after
    discrete decoding; positions and velocities themselves remain continuous.
    """
    if particles < 1 or iterations < 1:
        raise ValueError('particles and iterations must be positive')
    rng = np.random.default_rng(seed)
    upper = np.tile([len(ROOMS) - 1, len(TIMESLOTS) - 1], len(COURSES))
    positions = rng.uniform(0, upper, size=(particles, len(upper)))
    velocity_limit = upper * 0.5
    velocities = rng.uniform(-velocity_limit, velocity_limit, size=positions.shape)
    personal_best = positions.copy()
    personal_cost = np.array([evaluate(decode(p)).cost for p in positions])
    best_index = int(np.argmin(personal_cost))
    global_best = personal_best[best_index].copy()
    global_cost = int(personal_cost[best_index])
    history = [global_cost]
    for step in range(iterations):
        # Linearly reduce inertia to shift from exploration to refinement.
        inertia = 0.9 - 0.5 * step / max(1, iterations - 1)
        r1, r2 = rng.random(positions.shape), rng.random(positions.shape)
        velocities = (inertia * velocities
                      + 1.7 * r1 * (personal_best - positions)
                      + 1.7 * r2 * (global_best - positions))
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)
        positions = np.clip(positions + velocities, 0, upper)
        costs = np.array([evaluate(decode(p)).cost for p in positions])
        improved = costs < personal_cost
        personal_best[improved] = positions[improved]
        personal_cost[improved] = costs[improved]
        best_index = int(np.argmin(personal_cost))
        if personal_cost[best_index] < global_cost:
            global_cost = int(personal_cost[best_index])
            global_best = personal_best[best_index].copy()
        history.append(global_cost)
        # Zero is a proven lower bound for this nonnegative objective.
        if global_cost == 0:
            break
    return decode(global_best), history


def save_output(schedule: np.ndarray, history: list[int], output: Path, seed: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    score = evaluate(schedule)
    fig = plt.figure(figsize=(15, 10), facecolor='#f3f6fc')
    grid = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.1, 0.5], hspace=0.48)
    fig.suptitle('University Timetable Optimization | PSO', fontsize=23,
                 fontweight='bold', color='#123259', y=0.97)
    fig.text(0.5, 0.915,
             f'Hard violations: {score.hard}    |    Soft penalty: {score.faculty_gaps + score.day_imbalance}'
             f'    |    Best cost: {score.cost}    |    Iterations: {len(history) - 1}    |    Seed: {seed}',
             ha='center', fontsize=12, color='#34506f')
    ax = fig.add_subplot(grid[0]); ax.axis('off')
    rows = []
    for slot in range(len(TIMESLOTS)):
        row = [TIMESLOTS[slot]]
        for room in range(len(ROOMS)):
            entries = [f'{COURSES[i]}\n{FACULTY[i]}' for i, pair in enumerate(schedule)
                       if tuple(pair) == (room, slot)]
            row.append('\n'.join(entries) or '—')
        rows.append(row)
    table = ax.table(cellText=rows, colLabels=['Timeslot'] + ROOMS,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d3dfed')
        cell.set_facecolor('#163d68' if row == 0 else ('#ffffff' if row % 2 else '#eaf0f8'))
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
    ax = fig.add_subplot(grid[1])
    ax.plot(range(len(history)), history, color='#147d92', linewidth=2.5, marker='o', markersize=4)
    ax.set(title='Convergence of the global best', xlabel='Iteration (0 = initialization)',
           ylabel='Penalty cost (lower is better)')
    ax.set_yscale('symlog', linthresh=1); ax.grid(alpha=0.2)
    ax.set_xlim(0, max(1, len(history) - 1))
    ax = fig.add_subplot(grid[2]); ax.axis('off')
    ax.text(0, 1, 'Objective = 1000 × (faculty clashes + room clashes + room-type errors) + faculty gaps + day imbalance\n'
            f'Faculty clashes: {score.faculty_clashes}    Room clashes: {score.room_clashes}    '
            f'Room-type errors: {score.room_type_errors}\n'
            f'Faculty gaps: {score.faculty_gaps}    Day imbalance: {score.day_imbalance}    '
            'Room 101 / 102: Lecture    Lab A: Lab', va='top', fontsize=10, linespacing=1.9, color='#34506f')
    fig.subplots_adjust(top=0.86, bottom=0.05, left=0.08, right=0.96)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, default=60)
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('output.png'))
    args = parser.parse_args()
    schedule, history = optimize(args.particles, args.iterations, args.seed)
    score = evaluate(schedule)
    print(f'Best cost: {score.cost} | Hard violations: {score.hard} | Iterations: {len(history) - 1}')
    print(f'{"Course":<10} {"Faculty":<14} {"Room":<12} Timeslot')
    for i, (room, slot) in enumerate(schedule):
        print(f'{COURSES[i]:<10} {FACULTY[i]:<14} {ROOMS[room]:<12} {TIMESLOTS[slot]}')
    save_output(schedule, history, args.output, args.seed)
    print(f'Output saved to {args.output.resolve()}')
    if score.hard:
        print('Search ended with unresolved hard constraints; this timetable is not feasible.')


if __name__ == '__main__':
    main()
