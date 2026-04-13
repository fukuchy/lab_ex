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

# 石差、位置評価
def evaluate(pos, my_color):
    opp_color = pyrev.to_opponent_color(my_color)

    stone_score = pos.get_disc_count_of(my_color) - pos.get_disc_count_of(opp_color)

    pos_score = 0
    for board_index in range(64):
        if pos.get_square_color_at(board_index) == my_color:
            pos_score += EVAL_TABLE[board_index]
        elif pos.get_square_color_at(board_index) == opp_color:
            pos_score -= EVAL_TABLE[board_index]
    
    return stone_score + pos_score

def alpha_beta_rec(pos, alpha, beta, depth):
    if depth == 0 or pos.is_gameover():
        return evaluate(pos, pos.side_to_move)

    actions = pos.get_legal_moves()
    if not actions:
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
    actions = pos.get_legal_moves()
    for action in actions:
        next_pos = pos.copy()
        next_pos.do_move_at(action)
        score = -1 * alpha_beta_rec(next_pos, -beta, -alpha, depth - 1)
        if score > alpha:
            alpha = score
            best_action = action
    
    return best_action