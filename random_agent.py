import random

def select_move(pos):
    moves = list(pos.get_legal_moves())
    if not moves:
        return None

    return random.choice(moves)