import random
import pickle
import pyrev


class TDAgent:
    def __init__(self, alpha=0.003, gamma=1.0, epsilon=0.1): # 初回学習以降は alphaを下げる
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.weights = {
            "bias": 0.0,
            "piece_diff": 1.0,
            "mobility": 1.0,
            "positional": 1.0,
            "corner": 1.0,
        }

        self.eval_table = [
            120, -20,  20,   5,   5,  20, -20, 120,
            -20, -40,  -5,  -5,  -5,  -5, -40, -20,
             20,  -5,  15,   3,   3,  15,  -5,  20,
              5,  -5,   3,   3,   3,   3,  -5,   5,
              5,  -5,   3,   3,   3,   3,  -5,   5,
             20,  -5,  15,   3,   3,  15,  -5,  20,
            -20, -40,  -5,  -5,  -5,  -5, -40, -20,
            120, -20,  20,   5,   5,  20, -20, 120,
        ]

    def get_features(self, pos):
        me = pos.side_to_move
        opp = pyrev.to_opponent_color(me)

        my_discs = 0
        opp_discs = 0
        positional_score = 0.0

        for i in range(64):
            color = pos.get_square_color_at(i)

            if color == me:
                my_discs += 1
                positional_score += self.eval_table[i]
            elif color == opp:
                opp_discs += 1
                positional_score -= self.eval_table[i]

        total_discs = my_discs + opp_discs

        piece_diff = 0.0
        if total_discs > 0:
            piece_diff = (my_discs - opp_discs) / total_discs

        # 自分の合法手数
        my_moves = len(list(pos.get_legal_moves()))

        # 相手の合法手数を見るため、一度パスして相手番にする
        opp_pos = pos.copy()
        if opp_pos.can_pass():
            opp_pos.do_pass()
            opp_moves = len(list(opp_pos.get_legal_moves()))
        else:
            opp_moves = 0

        mobility = 0.0
        if my_moves + opp_moves > 0:
            mobility = (my_moves - opp_moves) / (my_moves + opp_moves)

        corners = [0, 7, 56, 63]
        my_corners = 0
        opp_corners = 0

        for c in corners:
            color = pos.get_square_color_at(c)

            if color == me:
                my_corners += 1
            elif color == opp:
                opp_corners += 1

        corner = (my_corners - opp_corners) / 4.0

        # 評価テーブル値を大きくなりすぎないよう正規化
        positional = positional_score / 1000.0

        return {
            "bias": 1.0,
            "piece_diff": piece_diff,
            "mobility": mobility,
            "positional": positional,
            "corner": corner,
        }

    def value(self, pos):
        features = self.get_features(pos)

        return sum(
            self.weights[name] * features[name]
            for name in self.weights
        )

    def select_move(self, pos, epsilon=None):
        moves = list(pos.get_legal_moves())

        if not moves:
            return None

        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.choice(moves)

        return max(moves, key=lambda move: self.value_after_move(pos, move))

    def value_after_move(self, pos, move):
        next_pos = pos.copy()
        next_pos.do_move_at(move)

        if next_pos.is_gameover():
            return self.final_value(next_pos, pos.side_to_move)

        # next_pos は相手番なので、自分視点では符号反転
        return -self.value(next_pos)

    def update(self, pos, reward, next_pos, done):
        features = self.get_features(pos)
        current_v = self.value(pos)

        if done:
            next_v = 0.0
        else:
            # next_pos は相手番なので符号反転
            next_v = -self.value(next_pos)

        td_target = reward + self.gamma * next_v
        td_error = td_target - current_v

        for name, x in features.items():
            self.weights[name] += self.alpha * td_error * x

    def reward(self, pos, learning_color):
        if not pos.is_gameover():
            return 0.0

        score = int(pos.get_score_from(pyrev.BLACK))

        if score == 0:
            return 0.0

        black_win = score > 0

        if learning_color == pyrev.BLACK:
            return 1.0 if black_win else -1.0
        else:
            return -1.0 if black_win else 1.0

    def final_value(self, pos, color):
        score = int(pos.get_score_from(pyrev.BLACK))

        if score == 0:
            return 0.0

        black_win = score > 0

        if color == pyrev.BLACK:
            return 1.0 if black_win else -1.0
        else:
            return -1.0 if black_win else 1.0

    def save(self, filename="td_weights.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.weights, f)

    def load(self, filename="td_weights.pkl"):
        with open(filename, "rb") as f:
            self.weights = pickle.load(f)


agent = TDAgent()


def select_move(pos, epsilon=None):
    return agent.select_move(pos, epsilon)


def update_v(pos, reward, next_pos, done):
    agent.update(pos, reward, next_pos, done)


def get_reward(pos, learning_color):
    return agent.reward(pos, learning_color)


def save_v(filename="td_weights.pkl"):
    agent.save(filename)


def load_v(filename="td_weights.pkl"):
    agent.load(filename)