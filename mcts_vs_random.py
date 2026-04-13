import pyrev
from pyrev import Position

import random_agent
import mcts_ex


def play_game(black_agent, white_agent, mcts_playout_num=100):
    pos = Position()

    while not pos.is_gameover():
        moves = list(pos.get_legal_moves())

        if not moves:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        if pos.side_to_move == pyrev.BLACK:
            move = select_agent_move(black_agent, pos, mcts_playout_num)
        else:
            move = select_agent_move(white_agent, pos, mcts_playout_num)

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


def select_agent_move(agent_name, pos, mcts_playout_num):
    if agent_name == "random":
        return random_agent.select_move(pos)
    if agent_name == "mcts":
        return mcts_ex.mcts(pos, playout_num=mcts_playout_num)
    raise ValueError(f"unknown agent: {agent_name}")


def run_matches(num_games=100, mcts_playout_num=100):
    results = {
        "random_win": 0,
        "mcts_win": 0,
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

        winner = play_game(
            black_agent,
            white_agent,
            mcts_playout_num=mcts_playout_num,
        )

        if winner == "draw":
            results["draw"] += 1
        elif winner == "black":
            if mcts_is_black:
                results["mcts_win"] += 1
            else:
                results["random_win"] += 1
        elif winner == "white":
            if mcts_is_black:
                results["random_win"] += 1
            else:
                results["mcts_win"] += 1

        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {num_games} games finished")

    return results


def print_results(results, num_games):
    random_win = results["random_win"]
    mcts_win = results["mcts_win"]
    draw = results["draw"]

    print("\n===== 対戦結果 =====")
    print(f"総対戦数: {num_games}")
    print(f"MCTSエージェント勝利: {mcts_win}")
    print(f"ランダムエージェント勝利: {random_win}")
    print(f"引き分け: {draw}")
    print()
    print(f"MCTSエージェント勝率: {mcts_win / num_games:.3f}")
    print(f"ランダムエージェント勝率: {random_win / num_games:.3f}")
    print(f"引き分け率: {draw / num_games:.3f}")


def main():
    num_games = 100
    mcts_playout_num = 100

    results = run_matches(
        num_games=num_games,
        mcts_playout_num=mcts_playout_num,
    )
    print_results(results, num_games)


if __name__ == "__main__":
    main()