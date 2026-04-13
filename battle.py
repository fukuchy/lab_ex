import pyrev
from pyrev import Position

import random_agent
import ab_ex


def play_game(black_agent, white_agent, ab_depth=4):
    pos = Position()

    while not pos.is_gameover():
        moves = list(pos.get_legal_moves())

        if not moves:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        if pos.side_to_move == pyrev.BLACK:
            move = select_agent_move(black_agent, pos, ab_depth)
        else:
            move = select_agent_move(white_agent, pos, ab_depth)

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


def select_agent_move(agent_name, pos, ab_depth):
    if agent_name == "random":
        return random_agent.select_move(pos)
    if agent_name == "ab":
        return ab_ex.alpha_beta(pos, depth=ab_depth)
    raise ValueError(f"unknown agent: {agent_name}")


def run_matches(num_games=100, ab_depth=4):
    results = {
        "random_win": 0,
        "ab_win": 0,
        "draw": 0,
    }

    for i in range(num_games):
        # 先手後手の偏りをなくすために交互に入れ替える
        if i % 2 == 0:
            black_agent = "ab"
            white_agent = "random"
            ab_is_black = True
        else:
            black_agent = "random"
            white_agent = "ab"
            ab_is_black = False

        winner = play_game(black_agent, white_agent, ab_depth=ab_depth)

        if winner == "draw":
            results["draw"] += 1
        elif winner == "black":
            if ab_is_black:
                results["ab_win"] += 1
            else:
                results["random_win"] += 1
        elif winner == "white":
            if ab_is_black:
                results["random_win"] += 1
            else:
                results["ab_win"] += 1

        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {num_games} games finished")

    return results


def print_results(results, num_games):
    random_win = results["random_win"]
    ab_win = results["ab_win"]
    draw = results["draw"]

    print("\n===== 対戦結果 =====")
    print(f"総対戦数: {num_games}")
    print(f"αβエージェント勝利: {ab_win}")
    print(f"ランダムエージェント勝利: {random_win}")
    print(f"引き分け: {draw}")
    print()
    print(f"αβエージェント勝率: {ab_win / num_games:.3f}")
    print(f"ランダムエージェント勝率: {random_win / num_games:.3f}")
    print(f"引き分け率: {draw / num_games:.3f}")


def main():
    num_games = 100
    ab_depth = 3

    results = run_matches(num_games=num_games, ab_depth=ab_depth)
    print_results(results, num_games)


if __name__ == "__main__":
    main()