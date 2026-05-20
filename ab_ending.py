import math
import pyrev

# 盤面評価値
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

# 空きマスがこの数以下になったら終盤読み切り
ENDGAME_EMPTY_LIMIT = 8

# 勝敗を強く評価するための値
WIN_SCORE = 100000


def evaluate(pos, my_color):
    opp_color = pyrev.to_opponent_color(my_color)

    # 石差
    my_count = pos.get_disc_count_of(my_color)
    opp_count = pos.get_disc_count_of(opp_color)

    total_discs = my_count + opp_count
    stone_score = my_count - opp_count

    # 着手可能数
    my_moves = len(list(pos.get_legal_moves()))

    mobility_score = my_moves

    # 相手の合法手数
    copy_pos = pos.copy()

    copy_pos.do_pass()
    opp_moves = len(list(copy_pos.get_legal_moves()))

    mobility_score -= opp_moves

    # 位置評価
    pos_score = 0

    for board_index in range(64):
        color = pos.get_square_color_at(board_index)

        if color == my_color:
            pos_score += EVAL_TABLE[board_index]
        elif color == opp_color:
            pos_score -= EVAL_TABLE[board_index]

    # フェーズごとの重み
    if total_discs <= 20:
        w_stone = -2.0
        w_mobile = 8.0
        w_pos = 0.5

    elif total_discs <= 40:
        w_stone = 0.2
        w_mobile = 6.0
        w_pos = 1.5

    else:
        w_stone = 8.0
        w_mobile = 1.0
        w_pos = 0.3

    return (
        w_stone * stone_score
        + w_mobile * mobility_score
        + w_pos * pos_score
    )


def order_moves(pos, actions, my_color):
    scored_actions = []

    for action in actions:
        next_pos = pos.copy()
        next_pos.do_move_at(action)

        score = evaluate(next_pos, my_color)

        scored_actions.append((score, action))

    scored_actions.sort(reverse=True)

    return [action for score, action in scored_actions]


def endgame_evaluate(pos, my_color):
    opp_color = pyrev.to_opponent_color(my_color)

    my_count = pos.get_disc_count_of(my_color)
    opp_count = pos.get_disc_count_of(opp_color)

    diff = my_count - opp_count

    if diff > 0:
        return WIN_SCORE + diff

    elif diff < 0:
        return -WIN_SCORE + diff

    return 0


def order_moves_endgame(pos, actions):
    scored_actions = []

    for action in actions:
        next_pos = pos.copy()
        next_pos.do_move_at(action)

        opp_moves = len(list(next_pos.get_legal_moves()))

        score = -opp_moves

        scored_actions.append((score, action))

    scored_actions.sort(reverse=True)

    return [action for score, action in scored_actions]


def endgame_search(pos, alpha, beta, my_color):
    if pos.is_gameover():
        return endgame_evaluate(pos, my_color)

    actions = list(pos.get_legal_moves())

    if not actions:
        next_pos = pos.copy()
        next_pos.do_pass()

        return -endgame_search(
            next_pos,
            -beta,
            -alpha,
            pyrev.to_opponent_color(my_color)
        )

    # move ordering
    actions = order_moves_endgame(pos, actions)

    for action in actions:
        next_pos = pos.copy()
        next_pos.do_move_at(action)

        score = -endgame_search(
            next_pos,
            -beta,
            -alpha,
            pyrev.to_opponent_color(my_color)
        )

        if score >= beta:
            return score

        if score > alpha:
            alpha = score

    return alpha


def alpha_beta_rec(pos, alpha, beta, depth):
    my_color = pos.side_to_move

    my_count = pos.get_disc_count_of(my_color)
    opp_count = pos.get_disc_count_of(
        pyrev.to_opponent_color(my_color)
    )

    empty_count = 64 - my_count - opp_count

    # 終盤読み切り
    if empty_count <= ENDGAME_EMPTY_LIMIT:
        return endgame_search(pos, alpha, beta, my_color)

    # 深さ制限 or ゲーム終了
    if depth == 0 or pos.is_gameover():
        return evaluate(pos, my_color)

    actions = list(pos.get_legal_moves())

    if not actions:
        next_pos = pos.copy()
        next_pos.do_pass()

        return -alpha_beta_rec(
            next_pos,
            -beta,
            -alpha,
            depth - 1
        )

    # move ordering
    actions = order_moves(pos, actions, my_color)

    for action in actions:
        next_pos = pos.copy()
        next_pos.do_move_at(action)

        score = -alpha_beta_rec(
            next_pos,
            -beta,
            -alpha,
            depth - 1
        )

        if score >= beta:
            return score

        if score > alpha:
            alpha = score

    return alpha


def alpha_beta(pos, depth):
    best_action = -1

    alpha = -math.inf
    beta = math.inf

    my_color = pos.side_to_move

    actions = list(pos.get_legal_moves())

    # move ordering
    actions = order_moves(pos, actions, my_color)

    for action in actions:
        next_pos = pos.copy()
        next_pos.do_move_at(action)

        score = -alpha_beta_rec(
            next_pos,
            -beta,
            -alpha,
            depth - 1
        )

        if score > alpha:
            alpha = score
            best_action = action

    return best_action