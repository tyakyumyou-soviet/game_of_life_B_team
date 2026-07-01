#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

import random
import numpy as np
from alifebook_lib.visualizers import MatrixVisualizer


# ===== 世界のパラメータ =====
WIDTH = 160
HEIGHT = 160

INITIAL_INDIVIDUALS = 8
INITIAL_FOOD = 640
INITIAL_FOOD_NEAR_INDIVIDUAL = 12

FOOD_SPAWN_RATE = 0.015
FOOD_ENERGY = 35
MAX_FOOD = 1040

INITIAL_ENERGY = 220.0
BASE_ENERGY_COST = 0.35
MOVE_ENERGY_COST = 1.2
GROW_ENERGY_COST = 18.0
GROW_MIN_ENERGY = 150.0
NEW_CELL_ENERGY = 8.0

COMBAT_RATE = 0.45
COMBAT_ENERGY_COST = 6.0
COMBAT_WIN_ENERGY = 25.0
COMBAT_DAMAGE = 35.0

REPRODUCTION_ENERGY = 180.0
REPRODUCTION_COST = 80.0
MUTATION_RATE = 0.08
MUTATION_STRENGTH = 0.15

MAX_INDIVIDUALS = 40
MAX_CELLS_PER_INDIVIDUAL = 22
MIN_CELLS_PER_INDIVIDUAL = 2

# 個体は単独セルではなく、連結した複数セルからなる体を持つ。
# ここでは最初の体を十字型の5細胞として作る。
INITIAL_BODY_SHAPE = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]

# 環境。値が大きいほど過酷で、消費エネルギーが増える。
ENVIRONMENT_CHANGE_RATE = 0.003
ENVIRONMENT_DAMAGE = 0.45


NAME_PARTS = [
    "Aqua", "Beta", "Cera", "Dino", "Echo", "Faro", "Giga", "Helio",
    "Iris", "Juno", "Kilo", "Luma", "Mira", "Nova", "Ortho", "Pyro",
]


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def random_name(individual_id):
    return "%s-%02d" % (random.choice(NAME_PARTS), individual_id)


def random_genes():
    return {
        "move_rate": random.uniform(0.35, 0.9),
        "growth_rate": random.uniform(0.15, 0.5),
        "food_sense": random.randint(3, 12),
        "energy_efficiency": random.uniform(0.7, 1.3),
        "body_stability": random.uniform(0.65, 1.0),
        "attack": random.uniform(0.5, 1.8),
        "defense": random.uniform(0.5, 1.8),
        "aggression": random.uniform(0.1, 0.8),
    }


def mutate_genes(parent_genes):
    child = {}
    for key, value in parent_genes.items():
        if random.random() < MUTATION_RATE:
            if isinstance(value, int):
                value += random.choice((-1, 1))
            else:
                value *= random.uniform(1.0 - MUTATION_STRENGTH, 1.0 + MUTATION_STRENGTH)
        child[key] = value

    child["move_rate"] = clamp(child["move_rate"], 0.05, 1.0)
    child["growth_rate"] = clamp(child["growth_rate"], 0.02, 0.9)
    child["food_sense"] = int(clamp(child["food_sense"], 1, 20))
    child["energy_efficiency"] = clamp(child["energy_efficiency"], 0.45, 1.8)
    child["body_stability"] = clamp(child["body_stability"], 0.25, 1.0)
    child["attack"] = clamp(child["attack"], 0.1, 3.0)
    child["defense"] = clamp(child["defense"], 0.1, 3.0)
    child["aggression"] = clamp(child["aggression"], 0.0, 1.0)
    return child


def make_individual(individual_id, parent_id=None, genes=None):
    if genes is None:
        genes = random_genes()
    return {
        "id": individual_id,
        "name": random_name(individual_id),
        "parent_id": parent_id,
        "age": 0,
        "energy": INITIAL_ENERGY,
        "kills": 0,
        "genes": genes,
    }


def in_bounds(y, x):
    return 0 <= y < HEIGHT and 0 <= x < WIDTH


def place_cells(owner, individual_id, cells):
    for y, x in cells:
        if in_bounds(y, x):
            owner[y, x] = individual_id


def cells_of(owner, individual_id):
    positions = np.argwhere(owner == individual_id)
    return [(int(y), int(x)) for y, x in positions]


def find_empty_near(owner, y, x, radius=4):
    candidates = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny = y + dy
            nx = x + dx
            if in_bounds(ny, nx) and owner[ny, nx] == -1:
                candidates.append((ny, nx))
    if not candidates:
        return None
    return random.choice(candidates)


def create_initial_body(owner, individual_id, y, x):
    cells = [(y + dy, x + dx) for dy, dx in INITIAL_BODY_SHAPE]
    if all(in_bounds(cy, cx) and owner[cy, cx] == -1 for cy, cx in cells):
        place_cells(owner, individual_id, cells)
        return True
    return False


def connected_body_components(cells):
    cell_set = set(cells)
    components = []

    while cell_set:
        start = cell_set.pop()
        component = [start]
        stack = [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                neighbor = (y + dy, x + dx)
                if neighbor in cell_set:
                    cell_set.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(component)

    return components


def keep_body_connected(owner, individual):
    cells = cells_of(owner, individual["id"])
    if len(cells) <= 1:
        return

    components = connected_body_components(cells)
    if len(components) <= 1:
        return

    main_body = max(components, key=len)
    lost_cells = 0
    for component in components:
        if component is main_body:
            continue
        lost_cells += len(component)
        for y, x in component:
            owner[y, x] = -1

    individual["energy"] -= lost_cells * NEW_CELL_ENERGY


def spawn_food(food, owner, count):
    empty = np.argwhere((food == 0) & (owner == -1))
    if len(empty) == 0:
        return
    count = min(count, len(empty))
    indices = np.random.choice(len(empty), size=count, replace=False)
    for index in indices:
        y, x = empty[index]
        food[y, x] = 1


def spawn_food_near(food, owner, y, x, count, radius=10):
    candidates = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny = y + dy
            nx = x + dx
            if in_bounds(ny, nx) and food[ny, nx] == 0 and owner[ny, nx] == -1:
                candidates.append((ny, nx))

    random.shuffle(candidates)
    for ny, nx in candidates[:count]:
        food[ny, nx] = 1


def nearest_food_direction(food, cells, sense_range):
    food_positions = np.argwhere(food == 1)
    if len(food_positions) == 0:
        return random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    best_distance = None
    best_target = None
    for y, x in cells:
        for fy, fx in food_positions:
            distance = abs(y - fy) + abs(x - fx)
            if distance <= sense_range and (best_distance is None or distance < best_distance):
                best_distance = distance
                best_target = (int(fy), int(fx), y, x)

    if best_target is None:
        return random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    fy, fx, y, x = best_target
    dy = fy - y
    dx = fx - x
    if abs(dx) > abs(dy):
        return (0, 1 if dx > 0 else -1)
    return (1 if dy > 0 else -1, 0)


def can_move(owner, individual_id, cells, dy, dx):
    for y, x in cells:
        ny = y + dy
        nx = x + dx
        if not in_bounds(ny, nx):
            return False
        if owner[ny, nx] not in (-1, individual_id):
            return False
    return True


def move_individual(owner, food, individual, dy, dx):
    individual_id = individual["id"]
    cells = cells_of(owner, individual_id)
    if not cells or not can_move(owner, individual_id, cells, dy, dx):
        return

    for y, x in cells:
        owner[y, x] = -1

    eaten = 0
    moved_cells = []
    for y, x in cells:
        ny = y + dy
        nx = x + dx
        moved_cells.append((ny, nx))
        if food[ny, nx] == 1:
            eaten += 1
            food[ny, nx] = 0

    place_cells(owner, individual_id, moved_cells)
    individual["energy"] += eaten * FOOD_ENERGY
    individual["energy"] -= MOVE_ENERGY_COST / individual["genes"]["energy_efficiency"]


def neighboring_enemy_ids(owner, individual_id):
    enemy_ids = set()
    for y, x in cells_of(owner, individual_id):
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny = y + dy
            nx = x + dx
            if not in_bounds(ny, nx):
                continue
            enemy_id = owner[ny, nx]
            if enemy_id >= 0 and enemy_id != individual_id:
                enemy_ids.add(int(enemy_id))
    return list(enemy_ids)


def combat_score(owner, individual):
    cells = cells_of(owner, individual["id"])
    body_bonus = len(cells) * 0.18
    energy_bonus = max(0.0, individual["energy"]) * 0.01
    return individual["genes"]["attack"] + body_bonus + energy_bonus + random.uniform(0.0, 1.0)


def defense_score(owner, individual):
    cells = cells_of(owner, individual["id"])
    body_bonus = len(cells) * 0.14
    energy_bonus = max(0.0, individual["energy"]) * 0.008
    return individual["genes"]["defense"] + body_bonus + energy_bonus + random.uniform(0.0, 1.0)


def remove_one_body_cell(owner, individual):
    cells = cells_of(owner, individual["id"])
    if len(cells) <= MIN_CELLS_PER_INDIVIDUAL:
        individual["energy"] = 0
        return
    y, x = random.choice(cells)
    owner[y, x] = -1
    individual["energy"] -= NEW_CELL_ENERGY


def fight(owner, individuals, attacker):
    if random.random() > attacker["genes"]["aggression"] * COMBAT_RATE:
        return

    enemies = [enemy_id for enemy_id in neighboring_enemy_ids(owner, attacker["id"]) if enemy_id in individuals]
    if not enemies:
        return

    defender = individuals[random.choice(enemies)]
    attacker["energy"] -= COMBAT_ENERGY_COST / attacker["genes"]["energy_efficiency"]

    attack = combat_score(owner, attacker)
    defense = defense_score(owner, defender)
    if attack > defense:
        defender["energy"] -= COMBAT_DAMAGE * attacker["genes"]["attack"] / defender["genes"]["defense"]
        remove_one_body_cell(owner, defender)
        attacker["energy"] += COMBAT_WIN_ENERGY
        attacker["kills"] += 1 if defender["energy"] <= 0 else 0
        print("fight: %s attacked %s win" % (attacker["name"], defender["name"]))
    else:
        attacker["energy"] -= COMBAT_DAMAGE * defender["genes"]["defense"] / attacker["genes"]["attack"]
        remove_one_body_cell(owner, attacker)
        print("fight: %s attacked %s lose" % (attacker["name"], defender["name"]))


def grow_individual(owner, food, individual):
    individual_id = individual["id"]
    cells = cells_of(owner, individual_id)
    if len(cells) >= MAX_CELLS_PER_INDIVIDUAL:
        return
    if individual["energy"] < GROW_MIN_ENERGY:
        return
    if random.random() > individual["genes"]["growth_rate"]:
        return

    y, x = random.choice(cells)
    new_cell = find_empty_near(owner, y, x, radius=1)
    if new_cell is None:
        return

    ny, nx = new_cell
    owner[ny, nx] = individual_id
    if food[ny, nx] == 1:
        food[ny, nx] = 0
        individual["energy"] += FOOD_ENERGY
    individual["energy"] -= GROW_ENERGY_COST


def lose_unstable_cells(owner, individual):
    if random.random() < individual["genes"]["body_stability"]:
        return

    cells = cells_of(owner, individual["id"])
    if len(cells) <= MIN_CELLS_PER_INDIVIDUAL:
        return
    y, x = random.choice(cells)
    owner[y, x] = -1
    individual["energy"] -= NEW_CELL_ENERGY


def reproduce(owner, individuals, lineage, next_id, parent):
    if len(individuals) >= MAX_INDIVIDUALS:
        return next_id
    if parent["energy"] < REPRODUCTION_ENERGY:
        return next_id

    parent_cells = cells_of(owner, parent["id"])
    random.shuffle(parent_cells)
    for y, x in parent_cells:
        start = find_empty_near(owner, y, x, radius=4)
        if start is None:
            continue

        child = make_individual(next_id, parent_id=parent["id"], genes=mutate_genes(parent["genes"]))
        if create_initial_body(owner, child["id"], start[0], start[1]):
            parent["energy"] -= REPRODUCTION_COST
            individuals[child["id"]] = child
            lineage.append((child["id"], parent["id"], child["name"], parent["name"]))
            print("born: %s <- %s genes=%s" % (child["name"], parent["name"], child["genes"]))
            return next_id + 1
    return next_id


def remove_dead(owner, individuals):
    for individual_id in list(individuals.keys()):
        individual = individuals[individual_id]
        cell_count = len(cells_of(owner, individual_id))
        if individual["energy"] <= 0 or cell_count < MIN_CELLS_PER_INDIVIDUAL:
            print("dead: %s age=%d cells=%d kills=%d" % (
                individual["name"], individual["age"], cell_count, individual["kills"]))
            owner[owner == individual_id] = -1
            del individuals[individual_id]


def print_population_summary(owner, step, individuals, food_count, lineage_count):
    print("step=%d alive=%d food=%d lineage=%d" % (step, len(individuals), food_count, lineage_count))
    ranked = sorted(
        individuals.values(),
        key=lambda individual: (individual["energy"], individual["age"], individual["kills"]),
        reverse=True,
    )
    for individual in ranked[:5]:
        genes = individual["genes"]
        print(
            "  %s parent=%s age=%d cells=%d energy=%.1f kills=%d "
            "move=%.2f grow=%.2f sense=%d atk=%.2f def=%.2f aggr=%.2f" % (
                individual["name"],
                individual["parent_id"],
                individual["age"],
                len(cells_of(owner, individual["id"])),
                individual["energy"],
                individual["kills"],
                genes["move_rate"],
                genes["growth_rate"],
                genes["food_sense"],
                genes["attack"],
                genes["defense"],
                genes["aggression"],
            )
        )


def update_environment(environment):
    change_count = int(WIDTH * HEIGHT * ENVIRONMENT_CHANGE_RATE)
    for _ in range(change_count):
        y = random.randrange(HEIGHT)
        x = random.randrange(WIDTH)
        environment[y, x] = clamp(environment[y, x] + random.uniform(-0.15, 0.15), 0.0, 1.0)


def apply_living_cost(owner, environment, individual):
    cells = cells_of(owner, individual["id"])
    if not cells:
        individual["energy"] = 0
        return
    env_cost = 0.0
    for y, x in cells:
        env_cost += environment[y, x] * ENVIRONMENT_DAMAGE
    cell_cost = len(cells) * BASE_ENERGY_COST
    individual["energy"] -= (cell_cost + env_cost) / individual["genes"]["energy_efficiency"]
    individual["age"] += 1


def render_matrix(owner, food, environment):
    # 0.0 = 黒, 1.0 = 白。環境は薄い背景、餌と個体を濃く表示する。
    image = 0.85 - environment * 0.25
    image[food == 1] = 0.35

    ids = np.unique(owner[owner >= 0])
    for individual_id in ids:
        shade = 0.05 + (individual_id % 9) * 0.06
        image[owner == individual_id] = shade
    return image


def main():
    visualizer = MatrixVisualizer(value_range_min=0.0, value_range_max=1.0)

    owner = np.full((HEIGHT, WIDTH), -1, dtype=np.int16)
    food = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
    environment = np.random.uniform(0.0, 0.35, size=(HEIGHT, WIDTH))

    individuals = {}
    lineage = []
    next_id = 0

    for _ in range(INITIAL_INDIVIDUALS):
        for _ in range(100):
            y = random.randrange(HEIGHT)
            x = random.randrange(WIDTH)
            if create_initial_body(owner, next_id, y, x):
                individual = make_individual(next_id)
                individuals[next_id] = individual
                spawn_food_near(food, owner, y, x, INITIAL_FOOD_NEAR_INDIVIDUAL)
                print("spawn: %s genes=%s" % (individual["name"], individual["genes"]))
                next_id += 1
                break

    spawn_food(food, owner, INITIAL_FOOD)

    step = 0
    while visualizer:
        step += 1

        if np.sum(food) < MAX_FOOD:
            spawn_food(food, owner, int(WIDTH * HEIGHT * FOOD_SPAWN_RATE))

        update_environment(environment)

        for individual_id in list(individuals.keys()):
            if individual_id not in individuals:
                continue
            individual = individuals[individual_id]
            if individual["energy"] <= 0:
                continue

            cells = cells_of(owner, individual_id)
            if not cells:
                individual["energy"] = 0
                continue

            if random.random() < individual["genes"]["move_rate"]:
                dy, dx = nearest_food_direction(food, cells, individual["genes"]["food_sense"])
                move_individual(owner, food, individual, dy, dx)

            grow_individual(owner, food, individual)
            fight(owner, individuals, individual)
            if individual["energy"] <= 0:
                continue

            lose_unstable_cells(owner, individual)
            keep_body_connected(owner, individual)
            apply_living_cost(owner, environment, individual)
            next_id = reproduce(owner, individuals, lineage, next_id, individual)

        remove_dead(owner, individuals)

        if step % 100 == 0:
            food_count = int(np.sum(food))
            print_population_summary(owner, step, individuals, food_count, len(lineage))

        visualizer.update(render_matrix(owner, food, environment))


if __name__ == '__main__':
    main()
