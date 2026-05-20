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

def evaluate(pos, my_color):
    opp_color = pyrev.to_opponent_color(my_color)

    # 石差
    my_count = pos.get_disc_count_of(my_color)
    opp_count = pos.get_disc_count_of(opp_color)

    total_discs = my_count + opp_count
    stone_score = my_count - opp_count

    # 着手可能数

    my_moves = len(list(pos.get_legal_moves()))

    copy_pos = pos.copy()
    copy_pos.do_pass()
    opp_moves = len(list(copy_pos.get_legal_moves()))

    mobility_score = my_moves - opp_moves

    # 位置評価

    pos_score = 0
    for board_index in range(64):
        if pos.get_square_color_at(board_index) == my_color:
            pos_score += EVAL_TABLE[board_index]
        elif pos.get_square_color_at(board_index) == opp_color:
            pos_score -= EVAL_TABLE[board_index]
    
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


    return w_stone * stone_score + w_mobile * mobility_score + w_pos * pos_score
    # return w_stone * stone_score + w_pos * pos_score

def alpha_beta_rec(pos, alpha, beta, depth):
    if depth == 0 or pos.is_gameover():
        return evaluate(pos, pos.side_to_move)

    # actions = pos.get_legal_moves()
    actions = list(pos.get_legal_moves())

    # if not actions:
    if len(actions) == 0:
        next_pos = pos.copy()
        next_pos.do_pass()
        return -alpha_beta_rec(next_pos, -beta, -alpha, depth - 1)

    for action in actions:
        next_pos = pos.copy()
        next_pos.do_move_at(action)
        score = -1 * alpha_beta_rec(next_pos, -beta, -alpha, depth - 1)

        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    
    return alpha

def alpha_beta(pos, depth):
    best_action = -1
    alpha = -1 * math.inf
    beta = math.inf
    # actions = pos.get_legal_moves()
    actions = list(pos.get_legal_moves())
    if len(actions) == 0:
        return best_action
    
    for action in actions:
        next_pos = pos.copy()
        next_pos.do_move_at(action)
        score = -1 * alpha_beta_rec(next_pos, -beta, -alpha, depth - 1)
        if score > alpha:
            alpha = score
            best_action = action
    
    return best_action