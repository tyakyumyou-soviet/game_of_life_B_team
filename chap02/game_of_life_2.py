#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

import random
import time
import numpy as np
from alifebook_lib.visualizers import MatrixVisualizer


# ===== 世界のパラメータ =====
WIDTH = 160
HEIGHT = 160

INITIAL_INDIVIDUALS = 10
INITIAL_FOOD = 720
INITIAL_FOOD_NEAR_INDIVIDUAL = 24

FOOD_SPAWN_RATE = 0.015
FOOD_ENERGY = 55
MAX_FOOD = 1200

STEPS_PER_SECOND = 30
FIGHT_STEPS = 2 * STEPS_PER_SECOND
MATING_STEPS = 1 * STEPS_PER_SECOND
FIGHT_SECONDS = 2.0
MATING_SECONDS = 1.0

INITIAL_ENERGY = 420.0
BASE_ENERGY_COST = 0.12
MOVE_ENERGY_COST = 0.55
GROW_ENERGY_COST = 18.0
GROW_MIN_ENERGY = 260.0
NEW_CELL_ENERGY = 8.0
MEAT_ENERGY_PER_CELL = 18.0

FIGHT_ENERGY_COST = 10.0
FIGHT_WIN_ENERGY = 35.0
FIGHT_DAMAGE = 42.0

REPRODUCTION_MIN_ENERGY = 170.0
MATING_ENERGY_COST = 45.0
REPRODUCTION_COOLDOWN_STEPS = 450
MUTATION_RATE = 0.08
MUTATION_STRENGTH = 0.15

MAX_INDIVIDUALS = 60
BASE_MAX_CELLS_PER_INDIVIDUAL = 22
MIN_CELLS_PER_INDIVIDUAL = 2

MALE_BODY_SHAPE = [
    (-2, 0),
    (-1, 0),
    (0, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]
FEMALE_BODY_SHAPE = [
    (-1, 0),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

ENVIRONMENT_CHANGE_RATE = 0.003
ENVIRONMENT_DAMAGE = 0.12

NAME_PARTS = [
    "Aqua", "Beta", "Cera", "Dino", "Echo", "Faro", "Giga", "Helio",
    "Iris", "Juno", "Kilo", "Luma", "Mira", "Nova", "Ortho", "Pyro",
]

NUMERIC_GENE_KEYS = [
    "move_rate", "growth_rate", "food_sense", "energy_efficiency",
    "body_stability", "attack", "defense", "aggression", "body_size",
    "muscle", "armor", "speed", "move_speed", "stamina", "fertility",
    "reproduction_power", "appetite", "libido", "predation", "fear",
    "parental_care", "lifespan",
]

TRAITS = {
    "normal": {"attack": 1.00, "defense": 1.00, "speed": 1.00, "cost": 1.00},
    "spikes": {"attack": 1.18, "defense": 0.95, "speed": 0.95, "cost": 1.08},
    "shell": {"attack": 0.88, "defense": 1.35, "speed": 0.72, "cost": 1.12},
    "fast": {"attack": 0.92, "defense": 0.86, "speed": 1.45, "cost": 1.10},
    "poison": {"attack": 1.10, "defense": 0.90, "speed": 0.95, "cost": 1.14},
    "giant": {"attack": 1.28, "defense": 1.15, "speed": 0.62, "cost": 1.28},
}


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def random_name(individual_id):
    return "%s-%02d" % (random.choice(NAME_PARTS), individual_id)


def random_sex():
    return random.choice(("male", "female"))


def random_genes():
    return {
        "move_rate": random.uniform(0.35, 0.9),
        "growth_rate": random.uniform(0.15, 0.5),
        "food_sense": random.randint(4, 14),
        "energy_efficiency": random.uniform(0.7, 1.3),
        "body_stability": random.uniform(0.65, 1.0),
        "attack": random.uniform(0.5, 1.8),
        "defense": random.uniform(0.5, 1.8),
        "aggression": random.uniform(0.1, 0.8),
        "body_size": random.uniform(0.7, 1.6),
        "muscle": random.uniform(0.65, 1.55),
        "armor": random.uniform(0.65, 1.55),
        "speed": random.uniform(0.65, 1.55),
        "move_speed": random.uniform(0.65, 1.7),
        "stamina": random.uniform(0.65, 1.55),
        "fertility": random.uniform(0.65, 1.55),
        "reproduction_power": random.uniform(0.65, 1.7),
        "appetite": random.uniform(0.65, 1.75),
        "libido": random.uniform(0.55, 1.85),
        "predation": random.uniform(0.05, 1.4),
        "fear": random.uniform(0.25, 1.7),
        "parental_care": random.uniform(0.15, 1.6),
        "lifespan": random.uniform(0.75, 1.45),
        "trait": random.choice(list(TRAITS.keys())),
    }


def mutate_genes(genes):
    child = genes.copy()
    for key in NUMERIC_GENE_KEYS:
        if random.random() < MUTATION_RATE:
            if key == "food_sense":
                child[key] += random.choice((-1, 1))
            else:
                child[key] *= random.uniform(1.0 - MUTATION_STRENGTH, 1.0 + MUTATION_STRENGTH)

    if random.random() < MUTATION_RATE:
        child["trait"] = random.choice(list(TRAITS.keys()))

    child["move_rate"] = clamp(child["move_rate"], 0.05, 1.0)
    child["growth_rate"] = clamp(child["growth_rate"], 0.02, 0.9)
    child["food_sense"] = int(clamp(child["food_sense"], 1, 24))
    child["energy_efficiency"] = clamp(child["energy_efficiency"], 0.45, 1.8)
    child["body_stability"] = clamp(child["body_stability"], 0.25, 1.0)
    child["attack"] = clamp(child["attack"], 0.1, 3.2)
    child["defense"] = clamp(child["defense"], 0.1, 3.2)
    child["aggression"] = clamp(child["aggression"], 0.0, 1.0)
    child["body_size"] = clamp(child["body_size"], 0.45, 2.4)
    child["muscle"] = clamp(child["muscle"], 0.3, 2.6)
    child["armor"] = clamp(child["armor"], 0.3, 2.6)
    child["speed"] = clamp(child["speed"], 0.25, 2.6)
    child["move_speed"] = clamp(child["move_speed"], 0.25, 2.8)
    child["stamina"] = clamp(child["stamina"], 0.3, 2.6)
    child["fertility"] = clamp(child["fertility"], 0.25, 2.4)
    child["reproduction_power"] = clamp(child["reproduction_power"], 0.25, 2.8)
    child["appetite"] = clamp(child["appetite"], 0.2, 2.8)
    child["libido"] = clamp(child["libido"], 0.1, 3.0)
    child["predation"] = clamp(child["predation"], 0.0, 3.0)
    child["fear"] = clamp(child["fear"], 0.0, 3.0)
    child["parental_care"] = clamp(child["parental_care"], 0.0, 3.0)
    child["lifespan"] = clamp(child["lifespan"], 0.45, 2.2)
    return child


def inherit_genes(mother_genes, father_genes):
    child = {}
    for key in NUMERIC_GENE_KEYS:
        if key == "food_sense":
            child[key] = int(round((mother_genes[key] + father_genes[key]) / 2.0))
        else:
            base = (mother_genes[key] + father_genes[key]) / 2.0
            inherited = random.choice((mother_genes[key], father_genes[key], base))
            child[key] = inherited
    child["trait"] = random.choice((mother_genes["trait"], father_genes["trait"]))
    return mutate_genes(child)


def make_individual(individual_id, parent_ids=None, genes=None, sex=None):
    if genes is None:
        genes = random_genes()
    if sex is None:
        sex = random_sex()
    return {
        "id": individual_id,
        "name": random_name(individual_id),
        "parent_ids": parent_ids or (),
        "mate_ids": set(),
        "sex": sex,
        "age": 0,
        "energy": INITIAL_ENERGY,
        "kills": 0,
        "genes": genes,
        "status": "normal",
        "state_timer": 0,
        "partner_id": None,
        "state_owner_id": None,
        "state_reason": None,
        "state_until": 0.0,
        "reproduction_cooldown": 0,
        "hunger": 0.0,
        "sexual_desire": 0.0,
    }


def is_close_family(a, b):
    a_parents = set(a["parent_ids"])
    b_parents = set(b["parent_ids"])
    if a["id"] in b_parents or b["id"] in a_parents:
        return True
    if a_parents and b_parents and a_parents == b_parents:
        return True
    return False


def is_mate(a, b):
    return b["id"] in a["mate_ids"] or a["id"] in b["mate_ids"]


def in_bounds(y, x):
    return 0 <= y < HEIGHT and 0 <= x < WIDTH


def cells_of(owner, individual_id):
    positions = np.argwhere(owner == individual_id)
    return [(int(y), int(x)) for y, x in positions]


def place_cells(owner, individual_id, cells):
    for y, x in cells:
        if in_bounds(y, x):
            owner[y, x] = individual_id


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


def body_shape_for_individual(individual):
    genes = individual["genes"]
    shape = set(FEMALE_BODY_SHAPE if individual["sex"] == "female" else MALE_BODY_SHAPE)

    if genes["body_size"] > 1.2:
        shape.update([(2, 0), (0, 2), (0, -2)])
    if genes["body_size"] > 1.7:
        shape.update([(-2, -1), (-2, 1), (2, -1), (2, 1)])
    if genes["muscle"] > 1.15 or genes["attack"] > 1.35:
        shape.update([(-1, -2), (-1, 2)])
    if genes["armor"] > 1.15 or genes["defense"] > 1.35:
        shape.update([(1, -2), (1, 2)])
    if genes["move_speed"] > 1.25 or genes["speed"] > 1.35:
        shape.update([(-3, 0)])
    if genes["fertility"] > 1.25 or genes["reproduction_power"] > 1.35:
        shape.update([(2, -1), (2, 1)])

    return list(shape)


def create_initial_body(owner, individual, y, x):
    cells = [(y + dy, x + dx) for dy, dx in body_shape_for_individual(individual)]
    if all(in_bounds(cy, cx) and owner[cy, cx] == -1 for cy, cx in cells):
        place_cells(owner, individual["id"], cells)
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


def max_cells_for(individual):
    return int(BASE_MAX_CELLS_PER_INDIVIDUAL * individual["genes"]["body_size"])


def age_thresholds(individual):
    scale = individual["genes"]["lifespan"]
    child_end = int(120 * scale)
    prime_end = int(1800 * scale)
    death_age = int(3000 * scale)
    return child_end, prime_end, death_age


def age_phase(individual):
    child_end, prime_end, death_age = age_thresholds(individual)
    if individual["age"] < child_end:
        return "child"
    if individual["age"] < prime_end:
        return "prime"
    if individual["age"] < death_age:
        return "old"
    return "dying"


def age_power(individual):
    phase = age_phase(individual)
    if phase == "child":
        child_end = max(1, age_thresholds(individual)[0])
        return 0.65 + 0.35 * individual["age"] / child_end
    if phase == "prime":
        return 1.25
    if phase == "old":
        return 1.25 * 0.60
    return 0.35


def is_reproductively_ready(individual):
    reproduction_threshold = REPRODUCTION_MIN_ENERGY / individual["genes"]["reproduction_power"]
    return (
        age_phase(individual) == "prime" and
        individual["energy"] >= reproduction_threshold and
        individual["reproduction_cooldown"] <= 0
    )


def can_reproduce(individual):
    return individual["status"] == "normal" and is_reproductively_ready(individual)


def update_basic_desires(individual):
    energy_target = INITIAL_ENERGY * 1.15
    hunger_base = clamp((energy_target - individual["energy"]) / energy_target, 0.0, 1.0)
    individual["hunger"] = clamp(hunger_base * individual["genes"]["appetite"], 0.0, 1.0)

    if can_reproduce(individual):
        reproduction_surplus = individual["energy"] / max(1.0, REPRODUCTION_MIN_ENERGY) - 0.7
        individual["sexual_desire"] = clamp(reproduction_surplus * individual["genes"]["libido"], 0.0, 1.0)
    else:
        individual["sexual_desire"] = 0.0


def trait_mod(individual, key):
    return TRAITS[individual["genes"]["trait"]][key]


def movement_chance(individual):
    genes = individual["genes"]
    size_penalty = max(0.35, genes["body_size"])
    return clamp(
        genes["move_rate"] * genes["move_speed"] * trait_mod(individual, "speed") *
        age_power(individual) / size_penalty,
        0.02,
        0.98,
    )


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
        return False

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
    move_cost = MOVE_ENERGY_COST * individual["genes"]["body_size"] * (0.75 + individual["genes"]["move_speed"] * 0.25)
    individual["energy"] -= move_cost / individual["genes"]["energy_efficiency"]
    return True


def movement_directions_with_fallback(preferred):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    ordered = []
    if preferred in directions:
        ordered.append(preferred)

    dy, dx = preferred
    perpendicular = []
    if dy != 0:
        perpendicular = [(0, 1), (0, -1)]
    elif dx != 0:
        perpendicular = [(1, 0), (-1, 0)]
    random.shuffle(perpendicular)

    opposite = (-dy, -dx)
    for direction in perpendicular + [opposite]:
        if direction in directions and direction not in ordered:
            ordered.append(direction)

    remaining = [direction for direction in directions if direction not in ordered]
    random.shuffle(remaining)
    return ordered + remaining


def move_individual_with_fallback(owner, food, individual, preferred_dy, preferred_dx):
    for dy, dx in movement_directions_with_fallback((preferred_dy, preferred_dx)):
        if move_individual(owner, food, individual, dy, dx):
            return True
    return False


def nearby_individual_ids(owner, individual_id, radius=1):
    ids = set()
    for y, x in cells_of(owner, individual_id):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if not in_bounds(ny, nx):
                    continue
                other_id = owner[ny, nx]
                if other_id >= 0 and other_id != individual_id:
                    ids.add(int(other_id))
    return list(ids)


def body_center(owner, individual):
    cells = cells_of(owner, individual["id"])
    if not cells:
        return None
    ys = [y for y, _ in cells]
    xs = [x for _, x in cells]
    return (sum(ys) / float(len(ys)), sum(xs) / float(len(xs)))


def direction_to_individual(owner, seeker, target):
    source_center = body_center(owner, seeker)
    target_center = body_center(owner, target)
    if source_center is None or target_center is None:
        return random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
    sy, sx = source_center
    ty, tx = target_center
    dy = ty - sy
    dx = tx - sx
    if abs(dx) > abs(dy):
        return (0, 1 if dx > 0 else -1)
    return (1 if dy > 0 else -1, 0)


def distance_between_individuals(owner, a, b):
    a_center = body_center(owner, a)
    b_center = body_center(owner, b)
    if a_center is None or b_center is None:
        return None
    ay, ax = a_center
    by, bx = b_center
    return abs(ay - by) + abs(ax - bx)


def direction_away_from_individual(owner, seeker, target):
    dy, dx = direction_to_individual(owner, seeker, target)
    return -dy, -dx


def strongest_score(owner, individual):
    cells = cells_of(owner, individual["id"])
    genes = individual["genes"]
    size_bonus = len(cells) * 0.16 * genes["body_size"]
    energy_bonus = max(0.0, individual["energy"]) * 0.006 * genes["stamina"]
    return (
        genes["attack"] * genes["muscle"] * trait_mod(individual, "attack") +
        genes["defense"] * genes["armor"] * trait_mod(individual, "defense") * 0.55 +
        genes["speed"] * trait_mod(individual, "speed") * 0.25 +
        size_bonus + energy_bonus
    ) * age_power(individual)


def attack_power(owner, individual):
    cells = cells_of(owner, individual["id"])
    genes = individual["genes"]
    return (
        genes["attack"] * genes["muscle"] * trait_mod(individual, "attack") * 2.1 +
        len(cells) * 0.13 * genes["body_size"] +
        genes["stamina"] * max(0.0, individual["energy"]) * 0.004
    ) * age_power(individual)


def defense_power(owner, individual):
    cells = cells_of(owner, individual["id"])
    genes = individual["genes"]
    return (
        genes["defense"] * genes["armor"] * trait_mod(individual, "defense") * 2.0 +
        len(cells) * 0.12 * genes["body_size"] +
        genes["speed"] * trait_mod(individual, "speed") * 0.35 +
        genes["stamina"] * max(0.0, individual["energy"]) * 0.003
    ) * age_power(individual)


def remove_body_cells(owner, individual, count):
    for _ in range(count):
        cells = cells_of(owner, individual["id"])
        if len(cells) <= MIN_CELLS_PER_INDIVIDUAL:
            individual["energy"] = 0
            return
        y, x = random.choice(cells)
        owner[y, x] = -1
        individual["energy"] -= NEW_CELL_ENERGY


def set_pair_state(a, b, status, timer, reason, seconds):
    owner_id = min(a["id"], b["id"])
    state_until = time.monotonic() + seconds
    for individual, partner in ((a, b), (b, a)):
        individual["status"] = status
        individual["state_timer"] = timer
        individual["partner_id"] = partner["id"]
        individual["state_owner_id"] = owner_id
        individual["state_reason"] = reason
        individual["state_until"] = state_until


def clear_state(individual):
    individual["status"] = "normal"
    individual["state_timer"] = 0
    individual["partner_id"] = None
    individual["state_owner_id"] = None
    individual["state_reason"] = None
    individual["state_until"] = 0.0


def cleanup_finished_states(individuals):
    now = time.monotonic()
    for individual in individuals.values():
        if individual["status"] == "normal":
            continue
        partner_id = individual["partner_id"]
        if partner_id not in individuals:
            clear_state(individual)
            continue
        if individual["status"] == "mating" and now >= individual["state_until"] + 0.25:
            clear_state(individual)


def start_fight(individuals, a, b, reason):
    if a["status"] != "normal" or b["status"] != "normal":
        return False
    if a["sex"] != b["sex"]:
        return False
    if is_close_family(a, b):
        return False
    if is_mate(a, b):
        return False
    set_pair_state(a, b, "fighting", FIGHT_STEPS, reason, FIGHT_SECONDS)
    print("fight_start: %s vs %s reason=%s" % (a["name"], b["name"], reason))
    return True


def resolve_fight(owner, individuals, a, b):
    a_score = attack_power(owner, a) + defense_power(owner, a) * 0.55
    b_score = attack_power(owner, b) + defense_power(owner, b) * 0.55
    a_score *= random.uniform(0.92, 1.08)
    b_score *= random.uniform(0.92, 1.08)

    if a_score >= b_score:
        winner, loser = a, b
        margin = a_score - b_score
    else:
        winner, loser = b, a
        margin = b_score - a_score

    damage = FIGHT_DAMAGE * (1.0 + margin / max(1.0, a_score + b_score))
    loser["energy"] -= damage / max(0.4, loser["genes"]["armor"])
    winner["energy"] -= FIGHT_ENERGY_COST / winner["genes"]["stamina"]
    loser["energy"] -= FIGHT_ENERGY_COST / loser["genes"]["stamina"]
    winner["energy"] += FIGHT_WIN_ENERGY

    lost_cells = 1 + int(max(0.0, margin) / 2.8)
    remove_body_cells(owner, loser, min(4, lost_cells))

    if winner["genes"]["trait"] == "poison":
        loser["energy"] -= 18.0
    if loser["genes"]["trait"] == "spikes":
        winner["energy"] -= 16.0

    if loser["energy"] <= 0:
        winner["kills"] += 1
        prey_cells = len(cells_of(owner, loser["id"]))
        gained_energy = prey_cells * MEAT_ENERGY_PER_CELL * winner["genes"]["predation"]
        winner["energy"] += gained_energy
        owner[owner == loser["id"]] = -1
        print("eat: %s consumed %s energy=%.1f" % (winner["name"], loser["name"], gained_energy))
    print("fight_end: %s beat %s" % (winner["name"], loser["name"]))


def process_pair_state(owner, individuals, individual, next_id, lineage):
    if individual["status"] == "normal":
        return next_id, False
    partner_id = individual["partner_id"]
    if partner_id not in individuals:
        clear_state(individual)
        return next_id, False
    partner = individuals[partner_id]
    if individual["id"] != individual["state_owner_id"]:
        return next_id, True

    individual["state_timer"] -= 1
    partner["state_timer"] = individual["state_timer"]
    if time.monotonic() < individual["state_until"]:
        return next_id, True

    if individual["status"] == "fighting":
        resolve_fight(owner, individuals, individual, partner)
    elif individual["status"] == "mating":
        next_id = resolve_mating(owner, individuals, lineage, next_id, individual, partner)

    clear_state(individual)
    if partner_id in individuals:
        clear_state(partner)
    return next_id, True


def grow_individual(owner, food, individual):
    individual_id = individual["id"]
    cells = cells_of(owner, individual_id)
    if len(cells) >= max_cells_for(individual):
        return
    if individual["energy"] < GROW_MIN_ENERGY:
        return
    if random.random() > individual["genes"]["growth_rate"] * age_power(individual):
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
    individual["energy"] -= GROW_ENERGY_COST * individual["genes"]["body_size"]


def lose_unstable_cells(owner, individual):
    stability = individual["genes"]["body_stability"] * (0.8 + 0.2 * age_power(individual))
    instability_rate = (1.0 - clamp(stability, 0.0, 1.0)) * 0.025
    if random.random() > instability_rate:
        return
    cells = cells_of(owner, individual["id"])
    if len(cells) <= MIN_CELLS_PER_INDIVIDUAL:
        return
    y, x = random.choice(cells)
    owner[y, x] = -1
    individual["energy"] -= NEW_CELL_ENERGY


def nearby_fertile_females(owner, individuals, male, radius=4):
    females = []
    for other_id in nearby_individual_ids(owner, male["id"], radius=radius):
        if other_id not in individuals:
            continue
        other = individuals[other_id]
        if other["sex"] == "female" and can_reproduce(other):
            females.append(other)
    return females


def nearby_fertile_males(owner, individuals, female, radius=4):
    males = []
    for other_id in nearby_individual_ids(owner, female["id"], radius=radius):
        if other_id not in individuals:
            continue
        other = individuals[other_id]
        if other["sex"] == "male" and can_reproduce(other):
            males.append(other)
    return males


def try_start_mate_fight(owner, individuals, male):
    if male["sex"] != "male" or not can_reproduce(male):
        return False
    females = nearby_fertile_females(owner, individuals, male, radius=5)
    females = [female for female in females if female["id"] not in male["mate_ids"]]
    if not females:
        return False
    female = max(females, key=lambda candidate: candidate["energy"] * candidate["genes"]["fertility"])
    rivals = [
        rival for rival in nearby_fertile_males(owner, individuals, female, radius=5)
        if rival["id"] != male["id"] and rival["status"] == "normal" and
        not is_close_family(male, rival) and not is_mate(male, rival)
    ]
    if not rivals:
        return False
    rival = max(rivals, key=lambda candidate: strongest_score(owner, candidate))
    contest_chance = clamp(male["genes"]["aggression"] * male["sexual_desire"] * 0.8, 0.05, 0.65)
    if random.random() > contest_chance:
        return False
    if strongest_score(owner, male) + male["genes"]["aggression"] > strongest_score(owner, rival) * 0.82:
        return start_fight(individuals, male, rival, "mate")
    return False


def try_start_general_fight(owner, individuals, individual):
    if individual["status"] != "normal":
        return False
    candidates = [individuals[other_id] for other_id in nearby_individual_ids(owner, individual["id"], radius=1) if other_id in individuals]
    candidates = [
        candidate for candidate in candidates
        if candidate["status"] == "normal" and candidate["sex"] == individual["sex"] and
        not is_close_family(individual, candidate) and not is_mate(individual, candidate)
    ]
    if not candidates:
        return False
    target = max(candidates, key=lambda candidate: max(0.1, strongest_score(owner, individual) - strongest_score(owner, candidate)))
    prey_weakness = clamp((INITIAL_ENERGY - target["energy"]) / INITIAL_ENERGY, 0.0, 1.0)
    predation_drive = individual["genes"]["predation"] * individual["hunger"] * (1.0 + prey_weakness)
    attack_drive = (individual["genes"]["aggression"] + predation_drive) * strongest_score(owner, individual)
    restraint = target["genes"]["defense"] * target["genes"]["armor"] * 0.65
    if attack_drive > restraint:
        return start_fight(individuals, individual, target, "territory")
    return False


def try_start_mating(owner, individuals, individual):
    if individual["sex"] != "female" or not can_reproduce(individual):
        return False
    males = nearby_fertile_males(owner, individuals, individual, radius=1)
    males = [male for male in males if male["status"] == "normal"]
    if len(males) != 1:
        return False
    male = males[0]
    if male["energy"] < REPRODUCTION_MIN_ENERGY:
        return False
    set_pair_state(individual, male, "mating", MATING_STEPS, "reproduction", MATING_SECONDS)
    print("mating_start: %s x %s" % (individual["name"], male["name"]))
    return True


def resolve_mating(owner, individuals, lineage, next_id, a, b):
    if a["sex"] == "female":
        mother, father = a, b
    else:
        mother, father = b, a
    if not is_reproductively_ready(mother) or not is_reproductively_ready(father):
        return next_id
    if len(individuals) >= MAX_INDIVIDUALS:
        return next_id

    mother["energy"] -= MATING_ENERGY_COST / mother["genes"]["reproduction_power"]
    father["energy"] -= MATING_ENERGY_COST * 0.55 / father["genes"]["reproduction_power"]
    mother["reproduction_cooldown"] = int(REPRODUCTION_COOLDOWN_STEPS / mother["genes"]["reproduction_power"])
    father["reproduction_cooldown"] = int(REPRODUCTION_COOLDOWN_STEPS / father["genes"]["reproduction_power"])
    mother["mate_ids"].add(father["id"])
    father["mate_ids"].add(mother["id"])
    child_count = random.randint(2, 5)

    parent_cells = cells_of(owner, mother["id"]) + cells_of(owner, father["id"])
    random.shuffle(parent_cells)
    born = 0
    for y, x in parent_cells:
        if born >= child_count or len(individuals) >= MAX_INDIVIDUALS:
            break
        start = find_empty_near(owner, y, x, radius=5)
        if start is None:
            continue
        child_genes = inherit_genes(mother["genes"], father["genes"])
        child = make_individual(next_id, parent_ids=(mother["id"], father["id"]), genes=child_genes)
        if create_initial_body(owner, child, start[0], start[1]):
            child["energy"] = INITIAL_ENERGY * 0.72
            individuals[child["id"]] = child
            lineage.append((child["id"], mother["id"], father["id"], child["name"], mother["name"], father["name"]))
            print("born: %s <- %s x %s sex=%s trait=%s" % (
                child["name"], mother["name"], father["name"], child["sex"], child["genes"]["trait"]))
            next_id += 1
            born += 1
    return next_id


def update_reproduction_cooldown(individual):
    if individual["reproduction_cooldown"] > 0:
        individual["reproduction_cooldown"] -= 1


def remove_dead(owner, individuals):
    for individual_id in list(individuals.keys()):
        individual = individuals[individual_id]
        _, _, death_age = age_thresholds(individual)
        cell_count = len(cells_of(owner, individual_id))
        if individual["energy"] <= 0 or cell_count < MIN_CELLS_PER_INDIVIDUAL or individual["age"] > death_age:
            print("dead: %s sex=%s age=%d phase=%s cells=%d kills=%d" % (
                individual["name"], individual["sex"], individual["age"], age_phase(individual), cell_count, individual["kills"]))
            owner[owner == individual_id] = -1
            partner_id = individual["partner_id"]
            if partner_id in individuals:
                clear_state(individuals[partner_id])
            del individuals[individual_id]


def print_population_summary(owner, step, individuals, food_count, lineage_count):
    print("step=%d alive=%d food=%d lineage=%d" % (step, len(individuals), food_count, lineage_count))
    ranked = sorted(
        individuals.values(),
        key=lambda individual: (individual["energy"], strongest_score(owner, individual), individual["kills"]),
        reverse=True,
    )
    for individual in ranked[:5]:
        genes = individual["genes"]
        print(
            "  %s %s phase=%s status=%s parents=%s cells=%d energy=%.1f kills=%d "
            "mates=%d cool=%d hunger=%.2f libido=%.2f atk=%.2f def=%.2f size=%.2f muscle=%.2f armor=%.2f "
            "speed=%.2f move=%.2f repr=%.2f pred=%.2f fear=%.2f care=%.2f trait=%s" % (
                individual["name"], individual["sex"], age_phase(individual), individual["status"],
                individual["parent_ids"], len(cells_of(owner, individual["id"])), individual["energy"],
                individual["kills"], len(individual["mate_ids"]), individual["reproduction_cooldown"],
                individual["hunger"], individual["sexual_desire"],
                genes["attack"], genes["defense"], genes["body_size"],
                genes["muscle"], genes["armor"], genes["speed"], genes["move_speed"],
                genes["reproduction_power"], genes["predation"], genes["fear"], genes["parental_care"],
                genes["trait"],
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
    genes = individual["genes"]
    cell_cost = len(cells) * BASE_ENERGY_COST * genes["body_size"] * trait_mod(individual, "cost")
    phase = age_phase(individual)
    if phase == "child":
        age_cost = 0.65
    elif phase == "old":
        age_cost = 1.22
    elif phase == "dying":
        age_cost = 1.55
    else:
        age_cost = 1.0
    individual["energy"] -= (cell_cost + env_cost) * age_cost / genes["energy_efficiency"]
    individual["age"] += 1


def phase_color(individual):
    phase = age_phase(individual)
    if phase == "child":
        return np.array([0.05, 0.88, 0.20])
    if phase == "prime":
        return np.array([0.96, 0.88, 0.12])
    if phase == "old":
        return np.array([0.14, 0.35, 1.00])
    return np.array([0.08, 0.12, 0.45])


def render_matrix(owner, food, environment, individuals):
    base = 0.82 - environment * 0.22
    image = np.dstack((base * 0.88, base * 0.92, base))
    image[food == 1] = np.array([0.78, 0.36, 0.10])

    ids = np.unique(owner[owner >= 0])
    for individual_id in ids:
        if int(individual_id) not in individuals:
            continue
        individual = individuals[int(individual_id)]
        color = phase_color(individual)
        if individual["sex"] == "female":
            color = color * np.array([1.0, 0.86, 1.0])
        if individual["status"] == "fighting":
            color = np.array([1.0, 0.08, 0.04])
        elif individual["status"] == "mating":
            color = np.array([1.0, 0.28, 0.82])
        image[owner == individual_id] = color
    return image


def nearby_fertile_partners(owner, individuals, individual, radius=12):
    partners = []
    for other_id in nearby_individual_ids(owner, individual["id"], radius=radius):
        if other_id not in individuals:
            continue
        other = individuals[other_id]
        if other["sex"] != individual["sex"] and can_reproduce(other):
            partners.append(other)
    return partners


def maybe_move_toward_mate(owner, food, individuals, individual):
    if not can_reproduce(individual):
        return False
    partners = nearby_fertile_partners(owner, individuals, individual, radius=12)
    if not partners:
        return False
    target = max(
        partners,
        key=lambda candidate: candidate["energy"] * candidate["genes"]["fertility"] * candidate["genes"]["reproduction_power"],
    )
    dy, dx = direction_to_individual(owner, individual, target)
    return move_individual_with_fallback(owner, food, individual, dy, dx)


def maybe_flee_from_threat(owner, food, individuals, individual):
    if individual["genes"]["fear"] <= 0.05:
        return False
    threats = []
    for other_id in nearby_individual_ids(owner, individual["id"], radius=6):
        if other_id not in individuals:
            continue
        other = individuals[other_id]
        if is_close_family(individual, other):
            continue
        if strongest_score(owner, other) > strongest_score(owner, individual) * 1.05:
            threats.append(other)
    if not threats:
        return False
    threat = max(threats, key=lambda candidate: strongest_score(owner, candidate))
    fear_drive = individual["genes"]["fear"] * (strongest_score(owner, threat) / max(0.1, strongest_score(owner, individual)))
    if fear_drive < 0.8:
        return False
    dy, dx = direction_away_from_individual(owner, individual, threat)
    return move_individual_with_fallback(owner, food, individual, dy, dx)


def maybe_move_to_child(owner, food, individuals, individual):
    if individual["genes"]["parental_care"] < 0.45:
        return False
    children = [
        other for other in individuals.values()
        if individual["id"] in other["parent_ids"] and age_phase(other) == "child"
    ]
    if not children:
        return False
    child = min(children, key=lambda candidate: candidate["energy"])
    distance = distance_between_individuals(owner, individual, child)
    if distance is None:
        return False
    if distance < 8.0 and child["energy"] >= INITIAL_ENERGY * 0.45:
        return False
    care_drive = individual["genes"]["parental_care"] * (1.2 if child["energy"] < INITIAL_ENERGY * 0.5 else 0.8)
    if care_drive < 0.7:
        return False
    dy, dx = direction_to_individual(owner, individual, child)
    return move_individual_with_fallback(owner, food, individual, dy, dx)


def maybe_move_to_parent(owner, food, individuals, individual):
    if age_phase(individual) != "child":
        return False
    parents = [individuals[parent_id] for parent_id in individual["parent_ids"] if parent_id in individuals]
    if not parents:
        return False
    target = max(parents, key=lambda parent: parent["genes"]["parental_care"] + strongest_score(owner, parent) * 0.05)
    distance = distance_between_individuals(owner, individual, target)
    if distance is None:
        return False
    if distance < 2.5:
        dy, dx = direction_away_from_individual(owner, individual, target)
        return move_individual_with_fallback(owner, food, individual, dy, dx)
    if distance < 7.0:
        return False
    dy, dx = direction_to_individual(owner, individual, target)
    return move_individual_with_fallback(owner, food, individual, dy, dx)


def maybe_leave_parent(owner, food, individuals, individual):
    if age_phase(individual) == "child":
        return False
    parents = [individuals[parent_id] for parent_id in individual["parent_ids"] if parent_id in individuals]
    if not parents:
        return False
    nearby_parents = [
        parent for parent in parents
        if parent["id"] in nearby_individual_ids(owner, individual["id"], radius=8)
    ]
    if not nearby_parents:
        return False
    parent = max(nearby_parents, key=lambda candidate: strongest_score(owner, candidate))
    dy, dx = direction_away_from_individual(owner, individual, parent)
    return move_individual_with_fallback(owner, food, individual, dy, dx)


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
            y = random.randrange(3, HEIGHT - 3)
            x = random.randrange(3, WIDTH - 3)
            initial_sex = "male" if next_id % 2 == 0 else "female"
            individual = make_individual(next_id, sex=initial_sex)
            if create_initial_body(owner, individual, y, x):
                individuals[next_id] = individual
                spawn_food_near(food, owner, y, x, INITIAL_FOOD_NEAR_INDIVIDUAL)
                print("spawn: %s sex=%s trait=%s genes=%s" % (
                    individual["name"], individual["sex"], individual["genes"]["trait"], individual["genes"]))
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
            update_reproduction_cooldown(individual)
            update_basic_desires(individual)

            next_id, busy = process_pair_state(owner, individuals, individual, next_id, lineage)
            if busy:
                apply_living_cost(owner, environment, individual)
                continue

            cells = cells_of(owner, individual_id)
            if not cells:
                individual["energy"] = 0
                continue

            if try_start_mating(owner, individuals, individual):
                apply_living_cost(owner, environment, individual)
                continue
            if try_start_mate_fight(owner, individuals, individual):
                apply_living_cost(owner, environment, individual)
                continue

            if random.random() < movement_chance(individual):
                moved = maybe_flee_from_threat(owner, food, individuals, individual)
                if not moved:
                    moved = maybe_leave_parent(owner, food, individuals, individual)
                if not moved:
                    moved = maybe_move_to_parent(owner, food, individuals, individual)
                if not moved and random.random() < individual["genes"]["parental_care"] * 0.35:
                    moved = maybe_move_to_child(owner, food, individuals, individual)
                if not moved and individual["sexual_desire"] > individual["hunger"]:
                    moved = maybe_move_toward_mate(owner, food, individuals, individual)
                if not moved:
                    dy, dx = nearest_food_direction(food, cells, individual["genes"]["food_sense"])
                    move_individual_with_fallback(owner, food, individual, dy, dx)

            grow_individual(owner, food, individual)
            try_start_general_fight(owner, individuals, individual)

            if individual["energy"] <= 0:
                continue

            lose_unstable_cells(owner, individual)
            keep_body_connected(owner, individual)
            apply_living_cost(owner, environment, individual)

        remove_dead(owner, individuals)
        cleanup_finished_states(individuals)

        if step % 100 == 0:
            food_count = int(np.sum(food))
            print_population_summary(owner, step, individuals, food_count, len(lineage))

        if not visualizer.update(render_matrix(owner, food, environment, individuals)):
            break

    visualizer.close()


if __name__ == '__main__':
    main()
