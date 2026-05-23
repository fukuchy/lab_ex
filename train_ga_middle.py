"""
GAで ab_ranforce.py の middle フェーズの評価関数重みだけを学習するスクリプト。

前提:
- このファイルを ab_ranforce.py と同じディレクトリに置いて実行する。
- ab_ranforce.py には FEATURE_WEIGHTS と alpha_beta(pos, depth, use_book, test_mode) があること。
- opening / end の重みは固定し、middle の7特徴量だけを遺伝子として扱う。

実行例:
    python train_ga_middle.py

出力:
    ga_middle_best_weights.json
    ga_middle_log.json
"""

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pyrev

import ab_rainforce as agent


# ==============================
# GA設定
# ==============================

FEATURE_NAMES = [
    "stone",
    "mobility",
    "position",
    "corner",
    "x_square",
    "c_square",
    "edge",
]

# middle の初期重みを基準個体にする
BASE_MIDDLE_WEIGHTS: Dict[str, float] = copy.deepcopy(agent.FEATURE_WEIGHTS["middle"])
BASE_ALL_WEIGHTS: Dict[str, Dict[str, float]] = copy.deepcopy(agent.FEATURE_WEIGHTS)

# 探索深さ。学習中は速度優先で小さめ推奨
DEPTH = 4

# 定石を使うか。
# middle重みの学習だけに集中したい場合は True でもよいが、
# 盤面の多様性を増やしたい場合は False も試す。
USE_BOOK = True

# 1個体を評価するための試合数。
# 偶数にして、先手・後手を同じ回数にする。
GAMES_PER_INDIVIDUAL = 100

# 集団サイズと世代数
POPULATION_SIZE = 100
GENERATIONS = 100

# 上位何個体を無条件で残すか
ELITE_SIZE = 2

# トーナメント選択で何個体から親を選ぶか
TOURNAMENT_SIZE = 20

# 交叉率・突然変異率
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.25

# 突然変異の強さ。各重みに対して標準偏差 ratio * abs(weight) 程度のノイズを加える。
MUTATION_RATIO = 0.20

# 重みが極端になりすぎないようにする範囲
WEIGHT_MIN = -200.0
WEIGHT_MAX = 200.0

# 評価値: 勝敗点 + 石差ボーナス
# 勝ち: +1, 引き分け: +0.5, 負け: 0
# 石差ボーナスは小さめにする。
DISC_DIFF_BONUS_SCALE = 0.01

RANDOM_SEED = 0


@dataclass
class Individual:
    weights: Dict[str, float]
    fitness: float = 0.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    avg_disc_diff: float = 0.0


def set_middle_weights(weights: Dict[str, float]) -> None:
    """
    ab_ranforce.py 側の middle 重みだけを差し替える。
    opening / end は固定。
    """
    agent.FEATURE_WEIGHTS["middle"] = copy.deepcopy(weights)


def reset_weights() -> None:
    """
    ab_ranforce.py 側の全重みを初期状態に戻す。
    """
    agent.FEATURE_WEIGHTS = copy.deepcopy(BASE_ALL_WEIGHTS)


def clip_weight(value: float) -> float:
    return max(WEIGHT_MIN, min(WEIGHT_MAX, value))


def random_individual() -> Individual:
    """
    初期重みの周辺にランダムな個体を作る。
    完全ランダムではなく、現在の重みを中心にばらつかせる。
    """
    weights = {}
    for name in FEATURE_NAMES:
        base = BASE_MIDDLE_WEIGHTS[name]
        sigma = max(1.0, abs(base) * 0.30)
        weights[name] = clip_weight(random.gauss(base, sigma))
    return Individual(weights=weights)


def baseline_individual() -> Individual:
    """
    現在の middle 重みをそのまま使う基準個体。
    毎世代に残しておくと、学習が悪化したか比較しやすい。
    """
    return Individual(weights=copy.deepcopy(BASE_MIDDLE_WEIGHTS))


def select_agent_move(weights: Dict[str, float], pos: pyrev.Position) -> int:
    """
    指定された middle 重みを agent にセットしてから手を選ぶ。
    """
    set_middle_weights(weights)
    return agent.alpha_beta(
        pos,
        depth=DEPTH,
        use_book=USE_BOOK,
        test_mode=False,
    )


def play_game(
    black_weights: Dict[str, float],
    white_weights: Dict[str, float],
) -> Tuple[str, int]:
    """
    1局対戦する。

    Returns:
        winner: "black" / "white" / "draw"
        black_score: 黒から見た石差。正なら黒勝ち、負なら白勝ち。
    """
    pos = pyrev.Position()

    while not pos.is_gameover():
        actions = list(pos.get_legal_moves())

        if not actions:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        if pos.side_to_move == pyrev.BLACK:
            move = select_agent_move(black_weights, pos)
        else:
            move = select_agent_move(white_weights, pos)

        if move is None or int(move) < 0:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        pos.do_move_at(np.int8(move))

    black_score = int(pos.get_score_from(pyrev.BLACK))

    if black_score > 0:
        return "black", black_score
    if black_score < 0:
        return "white", black_score
    return "draw", black_score


def evaluate_against_baseline(individual: Individual) -> Individual:
    """
    個体を基準重みと対戦させて適応度を計算する。
    先手・後手を半分ずつ行う。
    """
    wins = draws = losses = 0
    total_disc_diff = 0.0
    total_score = 0.0

    for game_index in range(GAMES_PER_INDIVIDUAL):
        individual_is_black = (game_index % 2 == 0)

        if individual_is_black:
            winner, black_score = play_game(
                black_weights=individual.weights,
                white_weights=BASE_MIDDLE_WEIGHTS,
            )
            disc_diff = black_score
            if winner == "black":
                result_score = 1.0
                wins += 1
            elif winner == "white":
                result_score = 0.0
                losses += 1
            else:
                result_score = 0.5
                draws += 1
        else:
            winner, black_score = play_game(
                black_weights=BASE_MIDDLE_WEIGHTS,
                white_weights=individual.weights,
            )
            disc_diff = -black_score
            if winner == "white":
                result_score = 1.0
                wins += 1
            elif winner == "black":
                result_score = 0.0
                losses += 1
            else:
                result_score = 0.5
                draws += 1

        total_disc_diff += disc_diff
        total_score += result_score + DISC_DIFF_BONUS_SCALE * disc_diff

    individual.wins = wins
    individual.draws = draws
    individual.losses = losses
    individual.avg_disc_diff = total_disc_diff / GAMES_PER_INDIVIDUAL
    individual.fitness = total_score / GAMES_PER_INDIVIDUAL
    return individual


def tournament_select(population: List[Individual]) -> Individual:
    """
    トーナメント選択。
    """
    candidates = random.sample(population, TOURNAMENT_SIZE)
    return max(candidates, key=lambda ind: ind.fitness)


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    一様交叉。
    特徴量ごとに、親1または親2の重みを受け継ぐ。
    """
    if random.random() > CROSSOVER_RATE:
        return Individual(weights=copy.deepcopy(parent1.weights))

    child_weights = {}
    for name in FEATURE_NAMES:
        if random.random() < 0.5:
            child_weights[name] = parent1.weights[name]
        else:
            child_weights[name] = parent2.weights[name]

    return Individual(weights=child_weights)


def mutate(individual: Individual) -> Individual:
    """
    突然変異。
    各重みに一定確率でガウスノイズを加える。
    """
    for name in FEATURE_NAMES:
        if random.random() < MUTATION_RATE:
            current = individual.weights[name]
            sigma = max(0.5, abs(current) * MUTATION_RATIO)
            individual.weights[name] = clip_weight(current + random.gauss(0.0, sigma))
    return individual


def make_next_generation(population: List[Individual]) -> List[Individual]:
    """
    エリート保存 + トーナメント選択 + 交叉 + 突然変異。
    """
    population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    next_population = [copy.deepcopy(ind) for ind in population[:ELITE_SIZE]]

    # 基準個体も1つ残す。初期重みより悪くなったかを比較しやすくするため。
    next_population.append(baseline_individual())

    while len(next_population) < POPULATION_SIZE:
        parent1 = tournament_select(population)
        parent2 = tournament_select(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        next_population.append(child)

    return next_population[:POPULATION_SIZE]


def save_best_weights(best: Individual, filename: str = "ga_middle_best_weights.json") -> None:
    """
    学習後の全フェーズ重みを保存する。
    middle だけ学習済みに差し替え、opening / end は元のまま保存する。
    """
    weights = copy.deepcopy(BASE_ALL_WEIGHTS)
    weights["middle"] = copy.deepcopy(best.weights)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=4)


def save_log(log: List[dict], filename: str = "ga_middle_log.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=4)


def print_individual(prefix: str, ind: Individual) -> None:
    weights_str = ", ".join(
        f"{name}={ind.weights[name]:.3f}" for name in FEATURE_NAMES
    )
    print(
        f"{prefix} fitness={ind.fitness:.3f} "
        f"W/D/L={ind.wins}/{ind.draws}/{ind.losses} "
        f"avg_diff={ind.avg_disc_diff:.2f} | {weights_str}"
    )


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    population: List[Individual] = [baseline_individual()]
    while len(population) < POPULATION_SIZE:
        population.append(random_individual())

    log: List[dict] = []

    for generation in range(1, GENERATIONS + 1):
        print(f"\n=== Generation {generation} ===")

        evaluated_population = []
        for i, individual in enumerate(population):
            evaluated = evaluate_against_baseline(individual)
            evaluated_population.append(evaluated)
            print_individual(f"[{i:02d}]", evaluated)

        evaluated_population.sort(key=lambda ind: ind.fitness, reverse=True)
        best = evaluated_population[0]
        print_individual("BEST", best)

        log.append({
            "generation": generation,
            "best_fitness": best.fitness,
            "wins": best.wins,
            "draws": best.draws,
            "losses": best.losses,
            "avg_disc_diff": best.avg_disc_diff,
            "best_middle_weights": copy.deepcopy(best.weights),
        })

        save_best_weights(best)
        save_log(log)

        population = make_next_generation(evaluated_population)

    final_best = max(population, key=lambda ind: ind.fitness)
    save_best_weights(final_best)
    save_log(log)
    reset_weights()

    print("\n学習終了")
    print("best weights saved to ga_middle_best_weights.json")
    print("log saved to ga_middle_log.json")


if __name__ == "__main__":
    main()
