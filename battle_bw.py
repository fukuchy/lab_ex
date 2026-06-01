import pyrev
from pyrev import Position

# 作成したエージェントを import する
# select_agent_move関数内でエージェント名と実装を対応させる

import random_agent
import ab_ex
import mcts_ex
import MCTS
import mcts_ai
import ab_opening
import ab_ending
import ab_rainforce
import ab_rainforce_after

ab_depth = 4
mcts_depth = 1000
num_games = 100


# 対戦させるエージェントを select_agent_move 関数で定義した名前で指定する
AGENT_A = "ab_rainforce_after"
AGENT_B = "mcts_fukuda"

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
    if agent_name == "mcts_fukuda":
        return mcts_ex.mcts(pos, mcts_depth)
    if agent_name == "mcts_ueki":
        return MCTS.mctsAction(pos, mcts_depth)
    if agent_name == "mcts_murayama":
        return mcts_ai.mctsAction(pos, mcts_depth)
    if agent_name == "ab_opening":
        return ab_opening.alpha_beta(pos, depth=ab_depth)
    if agent_name == "ab_ending":
        return ab_ending.alpha_beta(pos, depth=ab_depth)
    if agent_name == "ab_rainforce":
        return ab_rainforce.alpha_beta(pos, depth=ab_depth)
    if agent_name == "ab_rainforce_after":
        return ab_rainforce_after.alpha_beta(pos, depth=ab_depth)
    raise ValueError(f"unknown agent: {agent_name}")


def run_matches(num_games):
    results = {
        "agent_a_win": 0,
        "agent_b_win": 0,
        "draw": 0,
    }

    for i in range(num_games):
        agent_a = AGENT_A
        agent_b = AGENT_B

        winner = play_game(agent_a, agent_b)

        if winner == "draw":
            results["draw"] += 1
        elif winner == "black":
            results["agent_a_win"] += 1
        elif winner == "white":
            results["agent_b_win"] += 1

        if (i + 1) % 10 == 0:
            played = i + 1
            agent_a_win = results["agent_a_win"]
            agent_b_win = results["agent_b_win"]
            draw = results["draw"]

            print(f"\n{played} / {num_games} games finished")
            print(f"{AGENT_A} 勝利数: {agent_a_win}")
            print(f"{AGENT_B} 勝利数: {agent_b_win}")
            print(f"引き分け数: {draw}")
            print(f"{AGENT_A} 勝率: {agent_a_win / played:.3f}")
            print(f"{AGENT_B} 勝率: {agent_b_win / played:.3f}")
            print(f"引き分け率: {draw / played:.3f}")

    return results


def print_results(results, num_games):
    agent_a_win = results["agent_a_win"]
    agent_b_win = results["agent_b_win"]
    draw = results["draw"]

    print("\n===== 対戦結果 =====")
    print("エージェントA: ", AGENT_A)
    print("エージェントB: ", AGENT_B)
    print(f"総対戦数: {num_games}")
    print(f"エージェントA勝利: {agent_a_win}")
    print(f"エージェントB勝利: {agent_b_win}")
    print(f"引き分け: {draw}")
    print()
    print(f"エージェントA勝率: {agent_a_win / num_games:.3f}")
    print(f"エージェントB勝率: {agent_b_win / num_games:.3f}")
    print(f"引き分け率: {draw / num_games:.3f}")


def main():

    results = run_matches(num_games)
    print_results(results, num_games)


if __name__ == "__main__":
    main()