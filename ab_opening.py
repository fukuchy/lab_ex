import math, os, json, random, pyrev, numpy as np

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

ENDGAME_EMPTY_LIMIT = 11
WIN_SCORE = 100000


# ==============================
# 定石ブック設定
# ==============================
USE_OPENING_BOOK = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPENING_LINES_FILE = os.path.join(BASE_DIR, "opening_lines.json")

# Trueにすると、各手で定石か探索かを出力
BOOK_TEST_MODE = False

OPENING_BOOK = None

def build_opening_book_from_lines(lines_file=OPENING_LINES_FILE, test_mode=BOOK_TEST_MODE):
    book = {}

    try:
        with open(lines_file, "r", encoding="utf-8") as f:
            lines = json.load(f)
    except FileNotFoundError:
        if test_mode:
            print(f"[Book] {lines_file} が見つかりません")
        return {}

    for line in lines:
        pos = pyrev.Position()

        for move_str in line:
            move = int(pyrev.parse_coord_str(move_str))
            hash_code = str(int(pos.calc_hash_code()))

            if hash_code not in book:
                book[hash_code] = []

            if move not in book[hash_code]:
                book[hash_code].append(move)

            pos.do_move_at(np.int8(move))

    if test_mode:
        print(f"[Book] opening lines loaded: {lines_file}")
        print(f"[Book] lines: {len(lines)}")
        print(f"[Book] positions: {len(book)}")

    return book

def load_opening_book(test_mode=BOOK_TEST_MODE):
    global OPENING_BOOK

    if OPENING_BOOK is None:
        OPENING_BOOK = build_opening_book_from_lines(test_mode = BOOK_TEST_MODE)

    return OPENING_BOOK

def select_book_move(pos, test_mode = BOOK_TEST_MODE):
    book = load_opening_book(test_mode = BOOK_TEST_MODE)

    hash_code = str(int(pos.calc_hash_code()))

    if test_mode:
        print(f"[Book] current hash: {hash_code}")

    if hash_code not in book:
        if test_mode:
            print("[Book] この局面はブックにありません")
        return None

    legal_moves = set(int(m) for m in pos.get_legal_moves())

    candidates = [
        int(m)
        for m in book[hash_code]
        if int(m) in legal_moves
    ]

    if test_mode:
        print(
            "[Book] book candidates:",
            [coord_str(m) for m in book[hash_code]]
        )
        print(
            "[Book] legal moves:",
            [coord_str(m) for m in legal_moves]
        )

    if not candidates:
        if test_mode:
            print("[Book] ブック候補はありますが、合法手ではありません")
        return None

    move = random.choice(candidates)

    return np.int8(move)

def coord_str(move):
    if move is None or int(move) < 0:
        return "None"
    return pyrev.coord_to_str(np.int8(move))

def evaluate(pos, my_color):
    opp_color = pyrev.to_opponent_color(my_color)

    my_count = pos.get_disc_count_of(my_color)
    opp_count = pos.get_disc_count_of(opp_color)

    total_discs = my_count + opp_count
    stone_score = my_count - opp_count

    my_moves = len(list(pos.get_legal_moves()))

    mobility_score = my_moves

    copy_pos = pos.copy()
    copy_pos.do_pass()
    opp_moves = len(list(copy_pos.get_legal_moves()))

    mobility_score = my_moves - opp_moves

    pos_score = 0

    for board_index in range(64):
        color = pos.get_square_color_at(board_index)

        if color == my_color:
            pos_score += EVAL_TABLE[board_index]
        elif color == opp_color:
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

    if my_count > opp_count:
        return WIN_SCORE
    elif my_count < opp_count:
        return -WIN_SCORE
    else:
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

        # test 勝てる手が一つあれば大差かどうかは見ない

        if score == WIN_SCORE:
            return WIN_SCORE
    
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

    if empty_count <= ENDGAME_EMPTY_LIMIT:
        return endgame_search(pos, alpha, beta, my_color)

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


def alpha_beta_search_only(pos, depth):
    """
    定石を使わない通常のαβ探索。
    """
    best_action = -1

    alpha = -math.inf
    beta = math.inf

    my_color = pos.side_to_move

    actions = list(pos.get_legal_moves())

    if not actions:
        return best_action

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


def alpha_beta(pos, depth, use_book=USE_OPENING_BOOK, test_mode=BOOK_TEST_MODE):
    """
    定石ブックつきαβエージェント。

    use_book=True:
        opening_book.jsonを参照する

    test_mode=True:
        定石を使ったか、通常のαβ法を使ったかを出力する
    """

    if use_book:
        book_move = select_book_move(pos, test_mode = BOOK_TEST_MODE)

        if book_move is not None:
            if test_mode:
                print(f"[Book] 定石を使用: {coord_str(book_move)}")
            return book_move

        if test_mode:
            print("[AlphaBeta] 定石なし / 定石外れ / 定石終了: 通常のαβ法を使用")

    else:
        if test_mode:
            print("[AlphaBeta] 定石ブックOFF: 通常のαβ法を使用")

    best_action = alpha_beta_search_only(pos, depth)

    if test_mode:
        print(f"[AlphaBeta] 選択手: {coord_str(best_action)}")

    return best_action