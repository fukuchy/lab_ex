"""
GAで agent.py の opening・middle・end フェーズの評価関数重みを同時に学習するスクリプト。

【対戦相手の更新方式】
- global_best を常に対戦相手とする
- OPPONENT_UPDATE_INTERVAL 世代ごとに更新を試みる
- 今世代ベスト個体の勝率が OPPONENT_UPDATE_WIN_RATE を超えた場合のみ更新

【高速化 ①②③】
  ① ProcessPoolExecutor を main() で一度だけ生成し、全世代で使い回す
      → 毎世代のプロセス起動・終了コストを排除
  ② 1個体あたりの評価を EVAL_BATCH_SIZE 局単位のバッチタスクに細分化して並列発行
      → 全コアを均等に稼働させ、個体間の実行時間ばらつきによる手待ちを解消
  ③ set_all_weights の deepcopy を廃止し直接参照代入に変更
      探索中は agent.FEATURE_WEIGHTS を読むだけで書き換えないため参照共有は安全
      手番が切り替わった時のみ重みを差し替えることで代入回数も削減

前提:
- このファイルを agent.py と同じディレクトリに置いて実行する
- agent.py には FEATURE_WEIGHTS と alpha_beta(pos, depth, test_mode) があること

実行例:
    python train_ga_feature_weights.py

出力:
    ga_feature_weights_best.json   <- global_best が更新されるたびに上書き
    ga_feature_weights_log.json    <- 全世代のログ
"""

import copy
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrev

import agent


# ==============================
# GA 設定
# ==============================

FEATURE_NAMES = [
    "stone", "mobility", "position",
    "corner", "x_square", "c_square", "edge",
]

# 最適化対象フェーズ（ここに含めないフェーズは固定される）
TRAIN_PHASES = ["opening", "middle", "end"]

# スクリプト起動時点の全重みを保存（固定フェーズ補完・初期個体生成に使用）
INITIAL_ALL_WEIGHTS: Dict[str, Dict[str, float]] = copy.deepcopy(agent.FEATURE_WEIGHTS)

# 探索深さ。学習中は速度優先で小さめ推奨
DEPTH = 4

# 対局開始時にランダムに打つ手数（黒白合わせて）
# 4にすると黒2手・白2手がランダムに進んだ局面から対局が始まる
RANDOM_OPENING_PLIES = 4

# 1個体を評価するための試合数。偶数にして先手・後手を同じ回数にする
GAMES_PER_INDIVIDUAL = 100

# ② 1タスクあたりの対局数（バッチサイズ）
# GAMES_PER_INDIVIDUAL の約数にすること
# 小さいほどタスク数が増え負荷分散は改善するが IPC オーバーヘッドも増える
# 推奨: 5〜20
EVAL_BATCH_SIZE = 10

# 集団サイズと世代数
POPULATION_SIZE = 100
GENERATIONS = 100

# 上位何個体を無条件で次世代に残すか
ELITE_SIZE = 2

# トーナメント選択で何個体から親を選ぶか
TOURNAMENT_SIZE = 20

# 交叉率・突然変異率
CROSSOVER_RATE = 0.8
MUTATION_RATE  = 0.25

# 突然変異の強さ。各重みに標準偏差 ratio * abs(weight) 程度のノイズを加える
MUTATION_RATIO = 0.20

# 重みが極端になりすぎないようにする範囲
WEIGHT_MIN = -200.0
WEIGHT_MAX  =  200.0

# 勝敗点に加える石差ボーナスのスケール
# 勝ち: +1、引き分け: +0.5、負け: 0 をベースに石差を小さく加算する
DISC_DIFF_BONUS_SCALE = 0.01

# True にすると石差ボーナスを fitness に加算する
# False にすると勝敗のみで fitness を計算する（純粋な勝率を最大化したい場合）
USE_DISC_DIFF_BONUS = True

# 何世代ごとに global_best（対戦相手）の更新を試みるか
OPPONENT_UPDATE_INTERVAL = 10

# global_best を更新するために必要な勝率の閾値
# wins / GAMES_PER_INDIVIDUAL がこの値を超えた場合のみ更新する
OPPONENT_UPDATE_WIN_RATE = 0.90

RANDOM_SEED = 0

# ① 並列実行に使うプロセス数。None の場合は CPU コア数を使う
# Mac / Windows では if __name__ == "__main__" の中から main() を呼ぶこと
MAX_WORKERS = 20

# 出力ファイル名
OUTPUT_WEIGHTS_FILE = "ga_feature_weights_best.json"
OUTPUT_LOG_FILE     = "ga_feature_weights_log.json"


# ==============================
# 個体
# ==============================

@dataclass
class Individual:
    """
    TRAIN_PHASES の重みを1個体として保持する。

    phase_weights = {
        "opening": {"stone": ..., "mobility": ..., ...},
        "middle":  {"stone": ..., "mobility": ..., ...},
        "end":     {"stone": ..., "mobility": ..., ...},
    }
    """
    phase_weights: Dict[str, Dict[str, float]]
    fitness:       float = 0.0
    wins:          int   = 0
    draws:         int   = 0
    losses:        int   = 0
    avg_disc_diff: float = 0.0

    def to_all_weights(self) -> Dict[str, Dict[str, float]]:
        """
        TRAIN_PHASES 以外を INITIAL_ALL_WEIGHTS で補完した全フェーズ重みを返す。
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
    return {
        name: clip_weight(random.gauss(base[name], max(1.0, abs(base[name]) * 0.30)))
        for name in FEATURE_NAMES
    }


def initial_individual() -> Individual:
    """初期重みをそのまま持つ基準個体。"""
    return Individual(phase_weights={
        phase: copy.deepcopy(INITIAL_ALL_WEIGHTS[phase])
        for phase in TRAIN_PHASES
    })


def random_individual() -> Individual:
    """初期重みの周辺にランダムな個体を生成する。"""
    return Individual(phase_weights={
        phase: random_phase_weights(INITIAL_ALL_WEIGHTS[phase])
        for phase in TRAIN_PHASES
    })


# ==============================
# エージェント重み設定 (③ deepcopy 廃止)
# ==============================

def _set_weights_direct(all_weights: Dict[str, Dict[str, float]]) -> None:
    """
    ③ deepcopy を使わず直接参照を代入する高速版。

    対局中は agent.FEATURE_WEIGHTS の値を読むだけで書き換えないため、
    参照を共有しても安全。deepcopy (辞書の完全コピー) を毎手呼ぶコストを排除する。
    """
    for phase in TRAIN_PHASES:
        agent.FEATURE_WEIGHTS[phase] = all_weights[phase]


def reset_weights() -> None:
    """agent.FEATURE_WEIGHTS を起動時の初期状態に戻す。学習終了後に呼ぶ。"""
    agent.FEATURE_WEIGHTS = copy.deepcopy(INITIAL_ALL_WEIGHTS)


# ==============================
# 対局 (③ 重み切り替えを手番変化時のみに限定)
# ==============================

def play_game(
    black_all_weights: Dict[str, Dict[str, float]],
    white_all_weights: Dict[str, Dict[str, float]],
) -> Tuple[str, int]:
    """
    1局対戦する。

    ③ 手番が切り替わった時のみ _set_weights_direct() を呼ぶ。
       通常のオセロは毎手番が交互に変わるため呼び出し回数は変わらないが、
       deepcopy → 直接代入への変更により1回あたりのコストが大幅に下がる。

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

    # ③ 手番が変わった時のみ重みを切り替える（最初の手番でも初回セットが必要）
    last_side = None

    while not pos.is_gameover():
        actions = list(pos.get_legal_moves())

        if not actions:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        side = pos.side_to_move

        # ③ 手番が変わった場合のみ重みをセット（deepcopy なし）
        if side != last_side:
            _set_weights_direct(
                black_all_weights if side == pyrev.BLACK else white_all_weights
            )
            last_side = side

        move = agent.alpha_beta(pos, depth=DEPTH, test_mode=False)

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
# 並列評価ワーカー (②③)
# ==============================

def evaluate_batch_worker(
    task: Tuple[int, Dict, Dict, int, int]
) -> Tuple[int, int, int, int, float, float]:
    """
    ② バッチ単位（EVAL_BATCH_SIZE 局）の評価ワーカー。

    1個体=100局を1タスクとしていた旧方式を廃止し、
    EVAL_BATCH_SIZE 局単位の細粒度タスクに分割する。
    これにより:
      - 全ワーカーに均等に仕事が行き届く
      - 個体ごとの実行時間ばらつきによる手待ちを解消
      - ① プロセスプールを世代間で使い回せる（常に仕事があるため）

    task = (ind_index, ind_all_weights, opponent_all_weights, game_start, game_end)

    Returns:
        (ind_index, wins, draws, losses, disc_diff_sum, score_sum)
    """
    ind_index, ind_all_weights, opponent_all_weights, game_start, game_end = task

    # 個体・バッチごとに異なるシードを設定して結果の多様性を確保
    seed = RANDOM_SEED + ind_index * GAMES_PER_INDIVIDUAL + game_start
    random.seed(seed)
    np.random.seed(seed)

    wins = draws = losses = 0
    disc_diff_sum = 0.0
    score_sum     = 0.0

    for game_index in range(game_start, game_end):
        individual_is_black = (game_index % 2 == 0)

        if individual_is_black:
            winner, black_score = play_game(ind_all_weights, opponent_all_weights)
            disc_diff = black_score
            if winner == "black":
                wins += 1
                result_score = 1.0
            elif winner == "white":
                losses += 1
                result_score = 0.0
            else:
                draws += 1
                result_score = 0.5
        else:
            winner, black_score = play_game(opponent_all_weights, ind_all_weights)
            disc_diff = -black_score
            if winner == "white":
                wins += 1
                result_score = 1.0
            elif winner == "black":
                losses += 1
                result_score = 0.0
            else:
                draws += 1
                result_score = 0.5

        disc_diff_sum += disc_diff
        bonus          = DISC_DIFF_BONUS_SCALE * disc_diff if USE_DISC_DIFF_BONUS else 0.0
        score_sum     += result_score + bonus

    return ind_index, wins, draws, losses, disc_diff_sum, score_sum


# ==============================
# 並列評価 (①②)
# ==============================

def evaluate_population_parallel(
    population:            List[Individual],
    opponent_all_weights:  Dict[str, Dict[str, float]],
    executor:              ProcessPoolExecutor,         # ① 外部から受け取る
) -> List[Individual]:
    """
    ① 渡された executor (世代間で使い回す) にタスクを発行する。
       毎世代のプロセス起動・終了コストを排除。

    ② 1個体あたり GAMES_PER_INDIVIDUAL / EVAL_BATCH_SIZE 個のバッチタスクに分割。
       全コアが常に稼働するよう細かい粒度でタスクキューに投入する。

    バッチ結果は ind_index をキーに集計し、
    全バッチが揃った時点で Individual に反映・表示する。
    """
    batches_per_ind = GAMES_PER_INDIVIDUAL // EVAL_BATCH_SIZE
    total_batches   = len(population) * batches_per_ind

    # 集計バッファ (ind_index → 集計値)
    accum: Dict[int, Dict] = {
        i: {"wins": 0, "draws": 0, "losses": 0,
            "disc_diff": 0.0, "score": 0.0, "done": 0}
        for i in range(len(population))
    }

    # to_all_weights() をメインプロセスで事前計算してタスクに含める
    # ワーカーで Individual を pickle する必要がなくなる
    ind_weights_list = [ind.to_all_weights() for ind in population]

    tasks = [
        (i, ind_weights_list[i], opponent_all_weights,
         b * EVAL_BATCH_SIZE, (b + 1) * EVAL_BATCH_SIZE)
        for i in range(len(population))
        for b in range(batches_per_ind)
    ]

    evaluated_population: List[Optional[Individual]] = [None] * len(population)
    completed = 0

    futures = {executor.submit(evaluate_batch_worker, task): task for task in tasks}

    for future in as_completed(futures):
        ind_index, wins, draws, losses, disc_diff_sum, score_sum = future.result()

        acc = accum[ind_index]
        acc["wins"]      += wins
        acc["draws"]     += draws
        acc["losses"]    += losses
        acc["disc_diff"] += disc_diff_sum
        acc["score"]     += score_sum
        acc["done"]      += 1

        completed += 1
        # \r で同じ行を上書きしつつ進捗を表示
        print(f"  progress: {completed}/{total_batches} batches", end="\r", flush=True)

        # 全バッチが揃った個体を確定・表示
        if acc["done"] == batches_per_ind:
            ind = population[ind_index]
            ind.wins          = acc["wins"]
            ind.draws         = acc["draws"]
            ind.losses        = acc["losses"]
            ind.avg_disc_diff = acc["disc_diff"] / GAMES_PER_INDIVIDUAL
            ind.fitness       = acc["score"]     / GAMES_PER_INDIVIDUAL
            evaluated_population[ind_index] = ind
            print()  # \r 後の改行
            print_individual(f"  [{ind_index:02d}]", ind)

    print()  # 最終改行
    return evaluated_population  # type: ignore[return-value]


# ==============================
# GA 操作
# ==============================

def tournament_select(population: List[Individual]) -> Individual:
    candidates = random.sample(population, TOURNAMENT_SIZE)
    return max(candidates, key=lambda ind: ind.fitness)


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """一様交叉。各フェーズ・各特徴量ごとに親を選ぶ。"""
    if random.random() > CROSSOVER_RATE:
        return Individual(phase_weights=copy.deepcopy(parent1.phase_weights))

    child_phase_weights = {
        phase: {
            name: (
                parent1.phase_weights[phase][name]
                if random.random() < 0.5
                else parent2.phase_weights[phase][name]
            )
            for name in FEATURE_NAMES
        }
        for phase in TRAIN_PHASES
    }
    return Individual(phase_weights=child_phase_weights)


def mutate(individual: Individual) -> Individual:
    """突然変異。各フェーズの各重みに一定確率でガウスノイズを加える。"""
    for phase in TRAIN_PHASES:
        for name in FEATURE_NAMES:
            if random.random() < MUTATION_RATE:
                current = individual.phase_weights[phase][name]
                sigma   = max(0.5, abs(current) * MUTATION_RATIO)
                individual.phase_weights[phase][name] = clip_weight(
                    current + random.gauss(0.0, sigma)
                )
    return individual


def make_next_generation(population: List[Individual]) -> List[Individual]:
    """エリート保存 + トーナメント選択 + 交叉 + 突然変異。"""
    population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    next_population = [copy.deepcopy(ind) for ind in population[:ELITE_SIZE]]

    while len(next_population) < POPULATION_SIZE:
        parent1 = tournament_select(population)
        parent2 = tournament_select(population)
        child   = crossover(parent1, parent2)
        child   = mutate(child)
        next_population.append(child)

    return next_population[:POPULATION_SIZE]


# ==============================
# 保存・表示
# ==============================

def save_best_weights(best: Individual, filename: str = OUTPUT_WEIGHTS_FILE) -> None:
    """ベスト個体の全フェーズ重みを JSON に保存する。"""
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
        lines.append(f"    [{phase}] {w_str}")
    print("\n".join(lines))


# ==============================
# メイン (① executor を一度だけ生成)
# ==============================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 初期集団: 基準個体1体 + ランダム個体で埋める
    population: List[Individual] = [initial_individual()]
    while len(population) < POPULATION_SIZE:
        population.append(random_individual())

    global_best: Individual = initial_individual()

    log: List[dict] = []

    # ① ProcessPoolExecutor を一度だけ生成して全世代で使い回す
    # 毎世代のプロセス起動・終了コスト (pyrev インポートを含む) を排除する
    batches_per_ind    = GAMES_PER_INDIVIDUAL // EVAL_BATCH_SIZE
    max_tasks_per_gen  = POPULATION_SIZE * batches_per_ind
    max_workers_actual = MAX_WORKERS or os.cpu_count() or 1
    max_workers_actual = max(1, min(max_workers_actual, max_tasks_per_gen))

    print(f"[setup] MAX_WORKERS={max_workers_actual}, "
          f"EVAL_BATCH_SIZE={EVAL_BATCH_SIZE}, "
          f"tasks/generation={max_tasks_per_gen}")

    executor = ProcessPoolExecutor(max_workers=max_workers_actual)

    try:
        for generation in range(1, GENERATIONS + 1):
            print(f"\n=== Generation {generation} ===")
            print(f"[opponent] global_best "
                  f"(W/D/L={global_best.wins}/{global_best.draws}/{global_best.losses})")

            opponent_all_weights = global_best.to_all_weights()

            # ① 世代間で再利用する executor を渡す
            evaluated_population = evaluate_population_parallel(
                population, opponent_all_weights, executor
            )
            evaluated_population.sort(key=lambda ind: ind.fitness, reverse=True)

            best = evaluated_population[0]

            print("--- BEST of this generation ---")
            print_individual("BEST", best)
            print_individual("GLOBAL BEST", global_best)

            # OPPONENT_UPDATE_INTERVAL 世代ごとに更新を試みる
            win_rate         = best.wins / GAMES_PER_INDIVIDUAL
            is_update_timing = (generation % OPPONENT_UPDATE_INTERVAL == 0)
            global_best_updated = is_update_timing and (win_rate > OPPONENT_UPDATE_WIN_RATE)

            if is_update_timing:
                if global_best_updated:
                    global_best = copy.deepcopy(best)
                    print(
                        f"[global_best UPDATE] 勝率 {win_rate:.1%} > {OPPONENT_UPDATE_WIN_RATE:.0%} "
                        f"→ 世代 {generation} のトップ個体を次の対戦相手に設定"
                    )
                    save_best_weights(global_best)
                else:
                    next_t = OPPONENT_UPDATE_INTERVAL * (generation // OPPONENT_UPDATE_INTERVAL + 1)
                    print(
                        f"[global_best KEEP] 勝率 {win_rate:.1%} <= {OPPONENT_UPDATE_WIN_RATE:.0%} "
                        f"→ 対戦相手は据え置き（次の試行: 世代 {next_t}）"
                    )
            else:
                next_t = OPPONENT_UPDATE_INTERVAL * (generation // OPPONENT_UPDATE_INTERVAL + 1)
                print(f"[global_best KEEP] 次の更新試行: 世代 {next_t}")

            log.append({
                "generation":          generation,
                "best_fitness":        best.fitness,
                "win_rate":            win_rate,
                "wins":                best.wins,
                "draws":               best.draws,
                "losses":              best.losses,
                "avg_disc_diff":       best.avg_disc_diff,
                "global_best_updated": global_best_updated,
                "best_opening_weights": copy.deepcopy(best.phase_weights["opening"]),
                "best_middle_weights":  copy.deepcopy(best.phase_weights["middle"]),
                "best_end_weights":     copy.deepcopy(best.phase_weights["end"]),
            })

            save_log(log)
            population = make_next_generation(evaluated_population)

    finally:
        # ① 全世代終了後にプールを明示的にシャットダウン
        executor.shutdown(wait=True)

    reset_weights()

    print("\n学習終了")
    print(f"best weights saved to {OUTPUT_WEIGHTS_FILE}")
    print(f"log saved to {OUTPUT_LOG_FILE}")


if __name__ == "__main__":
    main()