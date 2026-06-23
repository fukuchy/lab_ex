"""
battle_rev.py - エージェント対戦スクリプト（並列化版）

高速化のポイント:
  ① ProcessPoolExecutor で複数ゲームを並列実行
     - 1局 = 1タスクとしてワーカープロセスに渡す
     - as_completed で完了した局から順次集計する
  ② select_agent_move を dict dispatch 化
     - 毎手番の if-elif 文字列比較をハッシュ参照に変更

注意:
  do_move のように wall-clock 時間制限を使うエージェントは、
  並列実行時に CPU を奪い合うため、制限時間内の探索深さが浅くなる。
  時間制限エージェントを正確に評価したい場合は MAX_WORKERS = 1 にすること。
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyrev
from pyrev import Position

import random_agent
import ab_randomWalk
import ab_ex
import mcts_ex
import MCTS
import mcts_murayama
import mcts_ai
import ab_opening
import ab_ending
import ab_rainforce
import ab_rainforce_after
import ab_rainforce_no_book
import agent
import agent_no_limit
import agent_limit
import agent_actual
import agent_final
import agent_final_tt
# ==============================
# 設定
# ==============================

AGENT_A = "agent_final"
AGENT_B = "agent_final_tt"

ab_depth       = 4
mcts_depth     = 1000
num_games      = 100
MOVE_TIME_LIMIT = 1      # agent の思考時間 (秒)
RANDUM_NUM      = 4      # ランダム手の数 (ab_randomWalk 用)
# ① 並列実行に使うプロセス数。
# 時間制限エージェントを含む場合は 1 に下げることを推奨。
MAX_WORKERS = 1


# ==============================
# ② dict dispatch による手番エージェント選択
# ==============================

def _build_dispatch():
    """
    エージェント名 → 着手関数のマッピングを構築する。
    モジュールロード時に1回だけ実行される。
    """
    return {
        "random":              lambda pos: random_agent.select_move(pos),
        "ab":                  lambda pos: ab_ex.alpha_beta(pos, depth=ab_depth),
        "mcts_fukuda":         lambda pos: mcts_ex.mcts(pos, mcts_depth),
        "mcts_ueki":           lambda pos: MCTS.mctsAction(pos, mcts_depth),
        "mcts_murayama":       lambda pos: mcts_ai.mctsAction(pos, mcts_depth),
        "ab_opening":          lambda pos: ab_opening.alpha_beta(pos, depth=ab_depth),
        "ab_ending":           lambda pos: ab_ending.alpha_beta(pos, depth=ab_depth),
        "ab_randomWalk":       lambda pos: ab_randomWalk.alpha_beta(pos, depth=ab_depth),
        "ab_rainforce":        lambda pos: ab_rainforce.alpha_beta(pos, depth=ab_depth),
        "ab_rainforce_after":  lambda pos: ab_rainforce_after.alpha_beta(pos, depth=ab_depth),
        "ab_rainforce_no_book":lambda pos: ab_rainforce_no_book.alpha_beta(pos, depth=ab_depth),
        "agent_no_limit":      lambda pos: agent_no_limit.alpha_beta(pos, depth=ab_depth),
        "agent":               lambda pos: agent.do_move(pos, MOVE_TIME_LIMIT),
        "agent_limit":         lambda pos: agent_limit.do_move(pos, MOVE_TIME_LIMIT),
        "agent_actual":        lambda pos: agent_actual.do_move(pos, MOVE_TIME_LIMIT),
        "agent_final":         lambda pos: agent_final.do_move(pos, MOVE_TIME_LIMIT),
        "agent_final_tt":      lambda pos: agent_final_tt.do_move(pos, MOVE_TIME_LIMIT),
    }

_AGENT_DISPATCH = _build_dispatch()


def select_agent_move(agent_name: str, pos: Position):
    fn = _AGENT_DISPATCH.get(agent_name)
    if fn is None:
        raise ValueError(f"unknown agent: {agent_name}")
    return fn(pos)


# ==============================
# ① ゲームワーカー (並列実行の1タスク)
# ==============================

def play_game_worker(args):
    """
    1局を実行して結果を返すワーカー関数。

    ProcessPoolExecutor からプロセス並列で呼ばれる。
    ワーカープロセスは各自でモジュールをインポートするため、
    グローバル状態の競合は発生しない。

    Returns
    -------
    (game_index, winner, a_is_black, a_move_times, b_move_times)
    """
    game_index, agent_a_name, agent_b_name = args

    a_is_black = (game_index % 2 == 0)
    black_agent = agent_a_name if a_is_black else agent_b_name
    white_agent = agent_b_name if a_is_black else agent_a_name

    a_move_times = []
    b_move_times = []

    pos = Position()

    while not pos.is_gameover():
        moves = list(pos.get_legal_moves())

        if not moves:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        is_black_turn = (pos.side_to_move == pyrev.BLACK)
        current_agent = black_agent if is_black_turn else white_agent
        is_agent_a    = (current_agent == agent_a_name)

        t0   = time.perf_counter()
        if pos.empty_square_count > 60 - RANDUM_NUM:
            move = random_agent.select_move(pos) 
        else:
            move = select_agent_move(current_agent, pos)
        t1   = time.perf_counter()
        elapsed = t1 - t0

        if is_agent_a:
            a_move_times.append(elapsed)
        else:
            b_move_times.append(elapsed)

        if move is None:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        pos.do_move_at(move)

    score = int(pos.get_score_from(pyrev.BLACK))
    if score > 0:
        winner = "black"
    elif score < 0:
        winner = "white"
    else:
        winner = "draw"

    return game_index, winner, a_is_black, a_move_times, b_move_times


# ==============================
# ① 並列対戦実行
# ==============================

def run_matches(num_games: int, max_workers: int = MAX_WORKERS):
    results = {"agent_a_win": 0, "agent_b_win": 0, "draw": 0}
    a_all_times = []
    b_all_times = []

    tasks = [(i, AGENT_A, AGENT_B) for i in range(num_games)]

    print(f"[battle] {AGENT_A} vs {AGENT_B} | {num_games}局 | "
          f"並列数={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(play_game_worker, task): task
                   for task in tasks}

        completed = 0
        for future in as_completed(futures):
            game_index, winner, a_is_black, a_times, b_times = future.result()
            a_all_times.extend(a_times)
            b_all_times.extend(b_times)

            if winner == "draw":
                results["draw"] += 1
            elif winner == "black":
                results["agent_a_win" if a_is_black else "agent_b_win"] += 1
            else:
                results["agent_b_win" if a_is_black else "agent_a_win"] += 1

            completed += 1
            if completed % 10 == 0:
                a_win = results["agent_a_win"]
                b_win = results["agent_b_win"]
                draw  = results["draw"]
                print(f"\n{completed} / {num_games} games finished")
                print(f"  {AGENT_A} 勝率: {a_win / completed:.3f}  "
                      f"{AGENT_B} 勝率: {b_win / completed:.3f}  "
                      f"引き分け率: {draw / completed:.3f}")

    return results, a_all_times, b_all_times


# ==============================
# 結果表示
# ==============================

def _time_summary(times: list) -> str:
    if not times:
        return "データなし"
    total  = sum(times)
    avg    = total / len(times)
    max_t  = max(times)
    min_t  = min(times)
    return (f"手数={len(times)}  合計={total:.1f}s  "
            f"平均={avg:.3f}s  最大={max_t:.3f}s  最小={min_t:.6f}s")


def print_results(results, num_games, a_times, b_times):
    a_win = results["agent_a_win"]
    b_win = results["agent_b_win"]
    draw  = results["draw"]

    print("\n===== 対戦結果 =====")
    print(f"エージェントA: {AGENT_A}")
    print(f"エージェントB: {AGENT_B}")
    print(f"総対戦数: {num_games}")
    print(f"エージェントA勝利: {a_win}  勝率: {a_win / num_games:.3f}")
    print(f"エージェントB勝利: {b_win}  勝率: {b_win / num_games:.3f}")
    print(f"引き分け:          {draw}    引き分け率: {draw / num_games:.3f}")
    print()
    print("===== 手番別時間サマリ =====")
    print(f"  {AGENT_A}: {_time_summary(a_times)}")
    print(f"  {AGENT_B}: {_time_summary(b_times)}")


# ==============================
# メイン
# ==============================

def main():
    t_start = time.perf_counter()

    results, a_times, b_times = run_matches(num_games)
    print_results(results, num_games, a_times, b_times)

    t_total = time.perf_counter() - t_start
    print(f"\n総実行時間: {t_total:.1f}s")


if __name__ == "__main__":
    main()