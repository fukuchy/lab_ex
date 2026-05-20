import time
import statistics
import pyrev
from pyrev import Position

import random_agent
import ab_ex
import mcts_ex
import ab_opening
import ab_ending
import ab_randomWalk

NUM_GAMES = 100
AB_DEPTH = 4
mcts_depth = 500


def select_agent_move(agent_name, pos):
    if agent_name == "random":
        return random_agent.select_move(pos)

    if agent_name == "ab":
        if pos.empty_square_count >= 56:
            return random_agent.select_move(pos)
        else:
            return ab_ex.alpha_beta(pos, depth=AB_DEPTH)

    if agent_name == "ab_opening":
        return ab_opening.alpha_beta(pos, depth=AB_DEPTH)

    if agent_name == "ab_ending":
        if pos.empty_square_count >= 56:
            return random_agent.select_move(pos)
        else:
            return ab_ending.alpha_beta(pos, depth=AB_DEPTH)

    if agent_name == "mcts":
        return mcts_ex.mcts(pos, mcts_depth)
    
    raise ValueError(f"unknown agent: {agent_name}")


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
    elif score < 0:
        return "white"
    else:
        return "draw"


def benchmark_self_play(agent_name, num_games):
    times = []
    results = {
        "black_win": 0,
        "white_win": 0,
        "draw": 0,
    }

    total_start = time.perf_counter()

    for i in range(num_games):
        start = time.perf_counter()

        result = play_game(agent_name, agent_name)

        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)

        if result == "black":
            results["black_win"] += 1
        elif result == "white":
            results["white_win"] += 1
        else:
            results["draw"] += 1

        print(f"{agent_name}: game {i + 1}/{num_games} finished in {elapsed:.4f} sec")

    total_end = time.perf_counter()
    total_time = total_end - total_start

    print("\n==============================")
    print(f"Agent: {agent_name}")
    print(f"Games: {num_games}")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Average time: {statistics.mean(times):.4f} sec/game")
    print(f"Min time: {min(times):.4f} sec")
    print(f"Max time: {max(times):.4f} sec")

    if len(times) >= 2:
        print(f"Std dev: {statistics.stdev(times):.4f} sec")

    print("Results:")
    print(f"  Black wins: {results['black_win']}")
    print(f"  White wins: {results['white_win']}")
    print(f"  Draws     : {results['draw']}")
    print("==============================\n")


def main():
    agents = [
        # "random",
        # "ab",
        "ab_opening",
        # "ab_ending",
        # "mcts",
    ]

    for agent in agents:
        benchmark_self_play(agent, NUM_GAMES)


if __name__ == "__main__":
    main()