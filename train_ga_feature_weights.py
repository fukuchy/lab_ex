"""
GAで ab_rainforce_no_book.py の opening・middle フェーズの評価関数重みを
同時に学習するスクリプト。

【前世代トップ個体を対戦相手にする方式】
- 世代1の対戦相手: スクリプト起動時点の初期重み
- 世代N(N>=2)の対戦相手: 世代N-1のトップ個体の重み
- 対戦相手が毎世代強くなるため、学習が常に「今の最強を超える」方向に進む

前提:
- このファイルを ab_rainforce_no_book.py と同じディレクトリに置いて実行する。
- end の重みは固定し、opening・middle の7+7=14特徴量を遺伝子として扱う。

実行例:
    python train_ga_feature_weights.py

出力:
    ga_feature_weights_best.json   <- 最新世代のベスト重み（毎世代上書き）
    ga_feature_weights_log.json    <- 全世代のログ
"""

import copy
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrev

import ab_rainforce_no_book as agent


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

# 最適化対象フェーズ。end は固定。
TRAIN_PHASES = ["opening", "middle"]

# スクリプト起動時点の全重みを保存（end フェーズ固定用・初期個体生成用）
INITIAL_ALL_WEIGHTS: Dict[str, Dict[str, float]] = copy.deepcopy(agent.FEATURE_WEIGHTS)

# 探索深さ。学習中は速度優先で小さめ推奨
DEPTH = 4

# 対局開始時にランダムに打つ手数（黒白合わせて）
# 4にすると黒2手・白2手がランダムに進んだ局面から評価対象の対局が始まる
RANDOM_OPENING_PLIES = 4

# 1個体を評価するための試合数。偶数にして先手・後手を同じ回数にする
GAMES_PER_INDIVIDUAL = 100

# 集団サイズと世代数
POPULATION_SIZE = 100
GENERATIONS = 100

# 上位何個体を無条件で次世代に残すか
ELITE_SIZE = 2

# トーナメント選択で何個体から親を選ぶか
TOURNAMENT_SIZE = 20

# 交叉率・突然変異率
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.25

# 突然変異の強さ。各重みに標準偏差 ratio * abs(weight) 程度のノイズを加える
MUTATION_RATIO = 0.20

# 重みが極端になりすぎないようにする範囲
WEIGHT_MIN = -200.0
WEIGHT_MAX = 200.0

# 勝敗点に加える石差ボーナスのスケール
# 勝ち: +1、引き分け: +0.5、負け: 0 をベースに石差を小さく加算する
DISC_DIFF_BONUS_SCALE = 0.01

RANDOM_SEED = 0

# 並列実行に使うプロセス数。None の場合は CPU コア数を使う
# Mac / Windows では if __name__ == "__main__" の中から main() を呼ぶこと
MAX_WORKERS = 20

# 出力ファイル名
OUTPUT_WEIGHTS_FILE = "ga_feature_weights_best.json"
OUTPUT_LOG_FILE = "ga_feature_weights_log.json"


# ==============================
# 個体
# ==============================

@dataclass
class Individual:
    """
    opening・middle の重みを1個体として保持する。

    phase_weights = {
        "opening": {"stone": ..., "mobility": ..., ...},
        "middle":  {"stone": ..., "mobility": ..., ...},
    }
    """
    phase_weights: Dict[str, Dict[str, float]]
    fitness: float = 0.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    avg_disc_diff: float = 0.0

    def to_all_weights(self) -> Dict[str, Dict[str, float]]:
        """
        end フェーズを初期値で補完した全フェーズ重みを返す。
        agent への適用・JSON 保存に使う。
        """
        all_w = copy.deepcopy(INITIAL_ALL_WEIGHTS)
        for phase in TRAIN_PHASES:
            all_w[phase] = copy.deepcopy(self.phase_weights[phase])
        return all_w


# ==============================
# 個体生成
# ==============================

def clip_weight(value: float) -> float:
    return max(WEIGHT_MIN, min(WEIGHT_MAX, value))


def random_phase_weights(base: Dict[str, float]) -> Dict[str, float]:
    """
    base を中心にガウスノイズを加えた重みを生成する。
    """
    return {
        name: clip_weight(random.gauss(base[name], max(1.0, abs(base[name]) * 0.30)))
        for name in FEATURE_NAMES
    }


def initial_individual() -> Individual:
    """
    初期重みをそのまま持つ基準個体。
    初期集団に1体含めることで、学習が退化した場合に比較できる。
    """
    return Individual(phase_weights={
        phase: copy.deepcopy(INITIAL_ALL_WEIGHTS[phase])
        for phase in TRAIN_PHASES
    })


def random_individual() -> Individual:
    """
    初期重みの周辺にランダムな個体を生成する。
    """
    return Individual(phase_weights={
        phase: random_phase_weights(INITIAL_ALL_WEIGHTS[phase])
        for phase in TRAIN_PHASES
    })


# ==============================
# エージェント操作
# ==============================

def set_all_weights(all_weights: Dict[str, Dict[str, float]]) -> None:
    """
    agent.FEATURE_WEIGHTS の opening・middle を差し替える。
    end は INITIAL_ALL_WEIGHTS のまま維持する。
    """
    for phase in TRAIN_PHASES:
        agent.FEATURE_WEIGHTS[phase] = copy.deepcopy(all_weights[phase])


def reset_weights() -> None:
    """
    agent.FEATURE_WEIGHTS を起動時の初期状態に戻す。
    """
    agent.FEATURE_WEIGHTS = copy.deepcopy(INITIAL_ALL_WEIGHTS)


def select_agent_move(all_weights: Dict[str, Dict[str, float]], pos: pyrev.Position) -> int:
    """
    指定された全フェーズ重みを agent にセットして手を選ぶ。
    """
    set_all_weights(all_weights)
    return agent.alpha_beta(pos, depth=DEPTH, test_mode=False)


# ==============================
# 対局
# ==============================

def play_game(
    black_all_weights: Dict[str, Dict[str, float]],
    white_all_weights: Dict[str, Dict[str, float]],
) -> Tuple[str, int]:
    """
    1局対戦する。

    Returns:
        winner     : "black" / "white" / "draw"
        black_score: 黒から見た石差（正=黒勝ち、負=白勝ち）
    """
    pos = pyrev.Position()

    # 序盤 RANDOM_OPENING_PLIES 手をランダムに進めて局面に多様性を持たせる
    for _ in range(RANDOM_OPENING_PLIES):
        if pos.is_gameover():
            break
        actions = list(pos.get_legal_moves())
        if not actions:
            if pos.can_pass():
                pos.do_pass()
            continue
        pos.do_move_at(np.int8(random.choice(actions)))

    # 本番対局
    while not pos.is_gameover():
        actions = list(pos.get_legal_moves())

        if not actions:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        if pos.side_to_move == pyrev.BLACK:
            move = select_agent_move(black_all_weights, pos)
        else:
            move = select_agent_move(white_all_weights, pos)

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


# ==============================
# 適応度評価
# ==============================

def evaluate_against_opponent(
    individual: Individual,
    opponent_all_weights: Dict[str, Dict[str, float]],
) -> Individual:
    """
    個体を opponent_all_weights と対戦させて適応度を計算する。
    先手・後手を半分ずつ行う。

    opponent_all_weights: 前世代のトップ個体の全フェーズ重み
    """
    wins = draws = losses = 0
    total_disc_diff = 0.0
    total_score = 0.0

    individual_all_weights = individual.to_all_weights()

    for game_index in range(GAMES_PER_INDIVIDUAL):
        individual_is_black = (game_index % 2 == 0)

        if individual_is_black:
            winner, black_score = play_game(
                black_all_weights=individual_all_weights,
                white_all_weights=opponent_all_weights,
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
                black_all_weights=opponent_all_weights,
                white_all_weights=individual_all_weights,
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


# ==============================
# GA 操作
# ==============================

def tournament_select(population: List[Individual]) -> Individual:
    candidates = random.sample(population, TOURNAMENT_SIZE)
    return max(candidates, key=lambda ind: ind.fitness)


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    一様交叉。opening・middle それぞれの特徴量ごとに親を選ぶ。
    """
    if random.random() > CROSSOVER_RATE:
        return Individual(phase_weights=copy.deepcopy(parent1.phase_weights))

    child_phase_weights = {}
    for phase in TRAIN_PHASES:
        child_phase_weights[phase] = {
            name: (
                parent1.phase_weights[phase][name]
                if random.random() < 0.5
                else parent2.phase_weights[phase][name]
            )
            for name in FEATURE_NAMES
        }

    return Individual(phase_weights=child_phase_weights)


def mutate(individual: Individual) -> Individual:
    """
    突然変異。opening・middle の各重みに一定確率でガウスノイズを加える。
    """
    for phase in TRAIN_PHASES:
        for name in FEATURE_NAMES:
            if random.random() < MUTATION_RATE:
                current = individual.phase_weights[phase][name]
                sigma = max(0.5, abs(current) * MUTATION_RATIO)
                individual.phase_weights[phase][name] = clip_weight(
                    current + random.gauss(0.0, sigma)
                )
    return individual


def make_next_generation(population: List[Individual]) -> List[Individual]:
    """
    エリート保存 + トーナメント選択 + 交叉 + 突然変異。
    """
    population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    # 上位 ELITE_SIZE 個体を無条件で次世代へ
    next_population = [copy.deepcopy(ind) for ind in population[:ELITE_SIZE]]

    # 残りをトーナメント選択 + 交叉 + 突然変異で生成
    while len(next_population) < POPULATION_SIZE:
        parent1 = tournament_select(population)
        parent2 = tournament_select(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        next_population.append(child)

    return next_population[:POPULATION_SIZE]


# ==============================
# 保存・表示
# ==============================

def save_best_weights(best: Individual, filename: str = OUTPUT_WEIGHTS_FILE) -> None:
    """
    ベスト個体の全フェーズ重みを JSON に保存する。
    opening・middle は学習済み値、end は初期値のまま。
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(best.to_all_weights(), f, ensure_ascii=False, indent=4)


def save_log(log: List[dict], filename: str = OUTPUT_LOG_FILE) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=4)


def print_individual(prefix: str, ind: Individual) -> None:
    lines = [
        f"{prefix} fitness={ind.fitness:.3f} "
        f"W/D/L={ind.wins}/{ind.draws}/{ind.losses} "
        f"avg_diff={ind.avg_disc_diff:.2f}"
    ]
    for phase in TRAIN_PHASES:
        w_str = ", ".join(
            f"{name}={ind.phase_weights[phase][name]:.3f}"
            for name in FEATURE_NAMES
        )
        lines.append(f"  [{phase}] {w_str}")
    print("\n".join(lines))


# ==============================
# 並列評価ワーカー
# ==============================

def evaluate_worker(
    task: Tuple[int, Individual, Dict[str, Dict[str, float]]]
) -> Tuple[int, Individual]:
    """
    ProcessPoolExecutor から呼び出すワーカー関数。

    task = (index, individual, opponent_all_weights)
    - opponent_all_weights: 前世代のトップ個体の全フェーズ重み
      （世代ごとに更新されるため、タスクに含めてプロセス間で共有する）
    - agent.FEATURE_WEIGHTS は各プロセスで独立しているため、
      set_all_weights() による書き換えは他プロセスに影響しない
    """
    index, individual, opponent_all_weights = task

    seed = RANDOM_SEED + index
    random.seed(seed)
    np.random.seed(seed)

    evaluated = evaluate_against_opponent(individual, opponent_all_weights)
    reset_weights()
    return index, evaluated


def evaluate_population_parallel(
    population: List[Individual],
    opponent_all_weights: Dict[str, Dict[str, float]],
) -> List[Individual]:
    """
    集団内の各個体評価をプロセス並列で実行する。
    opponent_all_weights を全タスクに渡すことで、
    全個体が同じ対戦相手（前世代のトップ）と戦う。
    """
    max_workers = MAX_WORKERS or os.cpu_count() or 1
    max_workers = max(1, min(max_workers, len(population)))

    evaluated_population: List[Optional[Individual]] = [None] * len(population)

    tasks = [(i, ind, opponent_all_weights) for i, ind in enumerate(population)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_worker, task) for task in tasks]

        completed = 0
        for future in as_completed(futures):
            index, evaluated = future.result()
            evaluated_population[index] = evaluated
            completed += 1
            print_individual(f"[{index:02d}]", evaluated)
            print(f"progress: {completed}/{len(population)} evaluated")

    return evaluated_population  # type: ignore[return-value]


# ==============================
# メイン
# ==============================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 初期集団: 基準個体1体 + ランダム個体で埋める
    population: List[Individual] = [initial_individual()]
    while len(population) < POPULATION_SIZE:
        population.append(random_individual())

    # global_best: これまでの全世代で最も強かった個体。
    # 常にこの個体を対戦相手とする。
    # 今世代のベスト個体が global_best に勝ち越した (fitness > 0.5) 場合のみ更新する。
    global_best: Individual = initial_individual()

    log: List[dict] = []

    for generation in range(1, GENERATIONS + 1):
        print(f"\n=== Generation {generation} ===")

        opponent_all_weights = global_best.to_all_weights()
        print(f"[opponent] global_best (W/D/L={global_best.wins}/{global_best.draws}/{global_best.losses})")

        evaluated_population = evaluate_population_parallel(population, opponent_all_weights)
        evaluated_population.sort(key=lambda ind: ind.fitness, reverse=True)

        best = evaluated_population[0]

        print("\n--- BEST of this generation ---")
        print_individual("BEST", best)
        print_individual("GLOBAL BEST", global_best)

        # fitness > 0.5 = global_best に勝ち越している → global_best を更新
        global_best_updated = best.fitness > 0.5
        if global_best_updated:
            global_best = copy.deepcopy(best)
            # global_best が更新されたため、fitness / W/D/L は次世代以降の評価で上書きされる。
            # ここではひとまず現世代の対戦結果をそのまま保持する。
            print(f"[global_best UPDATE] fitness={best.fitness:.3f} で global_best を更新しました")
            save_best_weights(global_best)
        else:
            print(f"[global_best KEEP] fitness={best.fitness:.3f} <= 0.5 のため global_best は据え置き")

        log.append({
            "generation": generation,
            "best_fitness": best.fitness,
            "wins": best.wins,
            "draws": best.draws,
            "losses": best.losses,
            "avg_disc_diff": best.avg_disc_diff,
            "global_best_updated": global_best_updated,
            "best_opening_weights": copy.deepcopy(best.phase_weights["opening"]),
            "best_middle_weights": copy.deepcopy(best.phase_weights["middle"]),
        })

        save_log(log)

        population = make_next_generation(evaluated_population)

    reset_weights()

    print("\n学習終了")
    print(f"best weights saved to {OUTPUT_WEIGHTS_FILE}")
    print(f"log saved to {OUTPUT_LOG_FILE}")


if __name__ == "__main__":
    main()
