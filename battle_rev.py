import pyrev
from pyrev import Position

import random_agent
import ab_ex
import mcts_ex

ab_depth = 3
mcts_depth = 500
def play_game(black_agent, white_agent):
    pos = Position()

    while not pos.is_gameover():
        moves = list(pos.get_legal_moves())

        if not moves:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        if pos.side_to_move == pyrev.BLACK:
            move = select_agent_move(black_agent, pos)
        else:
            move = select_agent_move(white_agent, pos)

        if move is None:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        pos.do_move_at(move)

    score = int(pos.get_score_from(pyrev.BLACK))

    if score > 0:
        return "black"
    if score < 0:
        return "white"
    return "draw"


def select_agent_move(agent_name, pos):
    if agent_name == "random":
        return random_agent.select_move(pos)
    if agent_name == "ab":
        return ab_ex.alpha_beta(pos, depth=ab_depth)
    if agent_name == "mcts":
        return mcts_ex.mcts(pos, mcts_depth)
    raise ValueError(f"unknown agent: {agent_name}")


def run_matches(num_games):
    results = {
        "agent_a_win": 0,
        "agent_b_win": 0,
        "draw": 0,
    }

    for i in range(num_games):
        # 先手後手の偏りをなくすために交互に入れ替える
        if i % 2 == 0:
            black_agent = "mcts"
            white_agent = "random"
            mcts_is_black = True
        else:
            black_agent = "random"
            white_agent = "mcts"
            mcts_is_black = False

        winner = play_game(black_agent, white_agent)

        if winner == "draw":
            results["draw"] += 1
        elif winner == "black":
            if mcts_is_black:
                results["agent_a_win"] += 1
            else:
                results["agent_b_win"] += 1
        elif winner == "white":
            if mcts_is_black:
                results["agent_b_win"] += 1
            else:
                results["agent_a_win"] += 1

        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {num_games} games finished")

    return results


def print_results(results, num_games):
    agent_a_win = results["agent_a_win"]
    agent_b_win = results["agent_b_win"]
    draw = results["draw"]

    print("\n===== 対戦結果 =====")
    print(f"総対戦数: {num_games}")
    print(f"エージェントA勝利: {agent_a_win}")
    print(f"エージェントB勝利: {agent_b_win}")
    print(f"引き分け: {draw}")
    print()
    print(f"エージェントA勝率: {agent_a_win / num_games:.3f}")
    print(f"エージェントB勝率: {agent_b_win / num_games:.3f}")
    print(f"引き分け率: {draw / num_games:.3f}")


def main():
    num_games = 100

    results = run_matches(num_games)
    print_results(results, num_games)


if __name__ == "__main__":
    main()