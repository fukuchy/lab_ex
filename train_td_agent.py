import os

import pyrev
from pyrev import Position

import td_agent
import random_agent
import ab_ex
import mcts_ex
import MCTS


ab_depth = 1
mcts_depth = 500

# ===== 設定 =====
OPPONENT_AGENT = "random"   # "random", "ab", "mcts_fukuda", "mcts_ueki"
NUM_GAMES = 10000
SAVE_INTERVAL = 1000
LOG_INTERVAL = 100
WEIGHT_FILE = "td_weights.pkl"


def select_opponent_move(agent_name, pos):
    if agent_name == "random":
        return random_agent.select_move(pos)
    if agent_name == "ab":
        return ab_ex.alpha_beta(pos, depth=ab_depth)
    if agent_name == "mcts_fukuda":
        return mcts_ex.mcts(pos, mcts_depth)
    if agent_name == "mcts_ueki":
        return MCTS.mctsAction(pos, mcts_depth)

    raise ValueError(f"unknown opponent agent: {agent_name}")


def get_learning_color(td_color):
    if td_color == "black":
        return pyrev.BLACK
    if td_color == "white":
        return pyrev.WHITE

    raise ValueError("td_color must be 'black' or 'white'")


def is_td_turn(pos, learning_color):
    return pos.side_to_move == learning_color


def get_result(pos, learning_color):
    score = int(pos.get_score_from(pyrev.BLACK))

    if score == 0:
        return 0

    black_win = score > 0

    if learning_color == pyrev.BLACK:
        return 1 if black_win else -1
    else:
        return 1 if not black_win else -1


def play_train_game(td_color):
    pos = Position()
    learning_color = get_learning_color(td_color)

    last_td_pos = None

    while not pos.is_gameover():
        moves = list(pos.get_legal_moves())

        if not moves:
            if pos.can_pass():
                before_pass = pos.copy()
                pos.do_pass()

                if before_pass.side_to_move == learning_color:
                    if last_td_pos is not None:
                        td_agent.update_v(
                            last_td_pos,
                            0.0,
                            before_pass,
                            False
                        )

                    last_td_pos = before_pass.copy()

                continue
            break

        if is_td_turn(pos, learning_color):
            if last_td_pos is not None:
                td_agent.update_v(
                    last_td_pos,
                    0.0,
                    pos,
                    False
                )

            last_td_pos = pos.copy()

            action = td_agent.select_move(
                pos,
                epsilon=td_agent.agent.epsilon
            )

            if action is None:
                if pos.can_pass():
                    pos.do_pass()
                continue

            pos.do_move_at(action)

        else:
            action = select_opponent_move(OPPONENT_AGENT, pos)

            if action is None:
                if pos.can_pass():
                    pos.do_pass()
                continue

            pos.do_move_at(action)

    if last_td_pos is not None:
        reward = td_agent.get_reward(pos, learning_color)

        td_agent.update_v(
            last_td_pos,
            reward,
            pos,
            True
        )

    return get_result(pos, learning_color)


def train(num_games=NUM_GAMES):
    print("===== TD学習開始 =====")
    print("td_agent: black / white alternating")
    print(f"opponent: {OPPONENT_AGENT}")
    print(f"num_games: {num_games}")

    if os.path.exists(WEIGHT_FILE):
        td_agent.load_v(WEIGHT_FILE)
        print(f"既存の {WEIGHT_FILE} を読み込みました")
        print("initial weights:", td_agent.agent.weights)
    else:
        print(f"{WEIGHT_FILE} がないため、新規重みで開始します")

    result_history = []
    black_results = []
    white_results = []

    for i in range(num_games):
        td_color = "black" if i % 2 == 0 else "white"

        result = play_train_game(td_color)

        result_history.append(result)

        if td_color == "black":
            black_results.append(result)
        else:
            white_results.append(result)

        if (i + 1) % LOG_INTERVAL == 0:
            recent = result_history[-LOG_INTERVAL:]
            wins = recent.count(1)
            losses = recent.count(-1)
            draws = recent.count(0)

            recent_black = black_results[-(LOG_INTERVAL // 2):]
            recent_white = white_results[-(LOG_INTERVAL // 2):]

            black_win_rate = (
                recent_black.count(1) / len(recent_black)
                if recent_black else 0.0
            )
            white_win_rate = (
                recent_white.count(1) / len(recent_white)
                if recent_white else 0.0
            )

            print(
                f"{i + 1} / {num_games} games | "
                f"recent win rate: {wins / len(recent):.3f} | "
                f"black: {black_win_rate:.3f} | "
                f"white: {white_win_rate:.3f} | "
                f"W:{wins} L:{losses} D:{draws}"
            )

        if (i + 1) % SAVE_INTERVAL == 0:
            print("weights:", td_agent.agent.weights)
            td_agent.save_v(WEIGHT_FILE)

    td_agent.save_v(WEIGHT_FILE)

    print("学習完了:", WEIGHT_FILE, "に保存しました")
    print("final weights:", td_agent.agent.weights)


if __name__ == "__main__":
    train(NUM_GAMES)