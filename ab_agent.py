import math
import pyrev

# 盤面評価値
# 角は非常に高く、角の隣は危険、辺はやや高めにしています
EVAL_TABLE = [
    120, -20,  20,   5,   5,  20, -20, 120,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
     20,  -5,  15,   3,   3,  15,  -5,  20,
      5,  -5,   3,   3,   3,   3,  -5,   5,
      5,  -5,   3,   3,   3,   3,  -5,   5,
     20,  -5,  15,   3,   3,  15,  -5,  20,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
    120, -20,  20,   5,   5,  20, -20, 120,
]


def evaluate(pos, my_color):
    """
    局面評価関数
    盤面重み + 着手可能数 + 終局時の石差
    """
    if pos.is_gameover():
        score = int(pos.get_score_from(my_color))
        # 終局時は石差を強く反映
        return score * 10000

    value = 0

    # 盤面重み
    for coord in range(64):
        color = pos.get_square_color_at(coord)
        if color == my_color:
            value += EVAL_TABLE[coord]
        elif color == pyrev.to_opponent_color(my_color):
            value -= EVAL_TABLE[coord]

    # 現手番の合法手数（ mobility ）
    current_moves = len(list(pos.get_legal_moves()))

    # 手番を1回パスして相手の合法手数も見る
    # can_pass() / do_pass() / undo は公開APIとして用意されています。 :contentReference[oaicite:1]{index=1}
    if pos.can_pass():
        # 現在手番に合法手が無い場合
        opponent_moves = 0
    else:
        pos.do_pass()
        opponent_moves = len(list(pos.get_legal_moves()))
        pos.do_pass()

    # 手番側が自分かどうかで符号を合わせる
    if pos.side_to_move == my_color:
        value += 5 * current_moves
        value -= 5 * opponent_moves
    else:
        value -= 5 * current_moves
        value += 5 * opponent_moves

    return value


def alphabeta(pos, depth, alpha, beta, my_color):
    """
    αβ枝刈り付きミニマックス
    """
    if depth == 0 or pos.is_gameover():
        return evaluate(pos, my_color)

    moves = list(pos.get_legal_moves())

    # 合法手が無い場合はパス
    if not moves:
        if pos.can_pass():
            pos.do_pass()
            score = alphabeta(pos, depth - 1, alpha, beta, my_color)
            pos.do_pass()
            return score
        return evaluate(pos, my_color)

    maximizing = (pos.side_to_move == my_color)

    if maximizing:
        value = -math.inf
        for move in moves:
            flip = pos.calc_flip_discs(move)
            pos.do_move(move, flip)

            score = alphabeta(pos, depth - 1, alpha, beta, my_color)

            pos.undo(move, flip)

            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value

    else:
        value = math.inf
        for move in moves:
            flip = pos.calc_flip_discs(move)
            pos.do_move(move, flip)

            score = alphabeta(pos, depth - 1, alpha, beta, my_color)

            pos.undo(move, flip)

            if score < value:
                value = score
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        return value


def select_move(pos, depth=4):
    """
    現在局面から最善手を返す
    合法手がなければ None
    """
    moves = list(pos.get_legal_moves())
    if not moves:
        return None

    my_color = pos.side_to_move
    best_move = None
    best_score = -math.inf

    alpha = -math.inf
    beta = math.inf

    for move in moves:
        flip = pos.calc_flip_discs(move)
        pos.do_move(move, flip)

        score = alphabeta(pos, depth - 1, alpha, beta, my_color)

        pos.undo(move, flip)

        if score > best_score:
            best_score = score
            best_move = move

        if best_score > alpha:
            alpha = best_score

    return best_move