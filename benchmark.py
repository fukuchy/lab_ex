import time
from collections import defaultdict

import pyrev
from pyrev import Position

import random_agent
import ab_ex
import mcts_ex
import MCTS
import mcts_ai
import ab_opening
import ab_ending
import ab_rainforce_after  # 必要なら
import agent
import agent_no_limit
import agent_limit
import agent_actual

ab_depth = 4
mcts_depth = 500
num_games = 10
random_num = 4

AGENT_A = "mcts"
AGENT_B = "agent_actual"


def play_game(black_agent, white_agent):
    pos = Position()

    move_times = []  # 1手ごとの時間を記録するリスト
    turn_count = 0

    while not pos.is_gameover():
        moves = list(pos.get_legal_moves())

        if not moves:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        if pos.side_to_move == pyrev.BLACK:
            agent_name = black_agent
            color_name = "black"
        else:
            agent_name = white_agent
            color_name = "white"

        # ==============================
        # ここで1手の選択時間を計測
        # ==============================
        start_time = time.perf_counter()

        move = select_agent_move(agent_name, pos)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        move_times.append({
            "turn": turn_count,
            "agent": agent_name,
            "color": color_name,
            "move": move,
            "time": elapsed_time,
        })

        print(
            f"[Move Time] "
            f"turn={turn_count:02d}, "
            f"color={color_name}, "
            f"agent={agent_name}, "
            f"time={elapsed_time:.6f} sec"
        )

        if move is None:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        pos.do_move_at(move)
        turn_count += 1

    score = int(pos.get_score_from(pyrev.BLACK))

    if score > 0:
        winner = "black"
    elif score < 0:
        winner = "white"
    else:
        winner = "draw"

    return winner, move_times


def select_agent_move(agent_name, pos):
    if pos.empty_square_count > 60 - random_num:
        # ゲーム序盤はランダムに動かす
        return random_agent.select_move(pos)
    if agent_name == "random":
        return random_agent.select_move(pos)

    if agent_name == "ab":
        return ab_ex.alpha_beta(pos, depth=ab_depth)

    if agent_name == "mcts":
        return mcts_ex.mcts(pos, mcts_depth)

    if agent_name == "MCTS":
        return MCTS.monte_carlo_tree_search(pos, mcts_depth)

    if agent_name == "mcts_ai":
        return mcts_ai.monte_carlo_tree_search(pos, mcts_depth)

    if agent_name == "ab_opening":
        return ab_opening.alpha_beta(pos, depth=ab_depth)

    if agent_name == "ab_ending":
        return ab_ending.alpha_beta(pos, depth=ab_depth)

    if agent_name == "ab_rainforce_after":
        return ab_rainforce_after.alpha_beta(pos, depth=ab_depth)
    if agent_name == "agent":
        return agent.do_move(pos, 3)

    if agent_name == "agent_no_limit":
        return agent_no_limit.alpha_beta(pos, depth=ab_depth)

    if agent_name == "agent_limit":
        return agent_limit.alpha_beta(pos, 3)
    
    if agent_name == "agent_actual":
        return agent_actual.do_move(pos, 3)

    raise ValueError(f"unknown agent: {agent_name}")


def run_matches(num_games):
    results = {
        "agent_a_win": 0,
        "agent_b_win": 0,
        "draw": 0,
    }

    all_move_times = []

    for i in range(num_games):
        print(f"\n===== Game {i + 1} / {num_games} =====")

        # 先手後手を交互に入れ替える
        if i % 2 == 0:
            black_agent = AGENT_A
            white_agent = AGENT_B
            a_is_black = True
        else:
            black_agent = AGENT_B
            white_agent = AGENT_A
            a_is_black = False

        winner, move_times = play_game(black_agent, white_agent)

        # 勝者の色 (black/white) を、その対局でどちらの色だったエージェントが
        # 勝ったかに変換して agent_a_win / agent_b_win に振り分ける。
        # 先後を毎局入れ替えているため、色ベースの勝敗集計のままでは
        # 「黒の勝率」「白の勝率」になってしまい、エージェント自体の強さを
        # 正しく反映しない。
        if winner == "draw":
            results["draw"] += 1
        elif winner == "black":
            results["agent_a_win" if a_is_black else "agent_b_win"] += 1
        elif winner == "white":
            results["agent_b_win" if a_is_black else "agent_a_win"] += 1

        all_move_times.extend(move_times)

        print(f"winner: {winner}  "
              f"(black={black_agent}, white={white_agent})")

    return results, all_move_times


def print_time_summary(move_times):
    if not move_times:
        print("No move time data.")
        return

    total_time = sum(record["time"] for record in move_times)
    average_time = total_time / len(move_times)
    max_time = max(record["time"] for record in move_times)
    min_time = min(record["time"] for record in move_times)

    print("\n===== Move Time Summary =====")
    print(f"total moves   : {len(move_times)}")
    print(f"total time    : {total_time:.6f} sec")
    print(f"average / move: {average_time:.6f} sec")
    print(f"max / move    : {max_time:.6f} sec")
    print(f"min / move    : {min_time:.6f} sec")

    # エージェントごとの平均時間
    agent_times = defaultdict(list)

    for record in move_times:
        agent_times[record["agent"]].append(record["time"])

    print("\n===== Agent Time Summary =====")

    for agent_name, times in agent_times.items():
        avg = sum(times) / len(times)
        mx = max(times)
        mn = min(times)

        print(
            f"{agent_name}: "
            f"moves={len(times)}, "
            f"avg={avg:.6f} sec, "
            f"max={mx:.6f} sec, "
            f"min={mn:.6f} sec"
        )


def print_match_result(results, num_games):
    a_win = results["agent_a_win"]
    b_win = results["agent_b_win"]
    draw  = results["draw"]

    print("\n===== Match Result =====")
    print(f"エージェントA: {AGENT_A}")
    print(f"エージェントB: {AGENT_B}")
    print(f"総対戦数: {num_games}")
    print(f"{AGENT_A} 勝利数: {a_win}  勝率: {a_win / num_games:.3f}")
    print(f"{AGENT_B} 勝利数: {b_win}  勝率: {b_win / num_games:.3f}")
    print(f"引き分け数: {draw}  引き分け率: {draw / num_games:.3f}")


def main():
    results, move_times = run_matches(num_games)

    print_match_result(results, num_games)
    print_time_summary(move_times)


if __name__ == "__main__":
    main()