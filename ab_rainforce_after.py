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
GA_BEST_WEIGHTS_FILE = os.path.join(BASE_DIR, "ga_middle_best_weights.json")

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

# ==============================
# 評価関数用：特徴量と重み
# ==============================

# 角
CORNERS = [0, 7, 56, 63]

# X-square: 角の斜め隣
X_SQUARES = {
    0: [9],
    7: [14],
    56: [49],
    63: [54],
}

# C-square: 角の縦横隣
C_SQUARES = {
    0: [1, 8],
    7: [6, 15],
    56: [48, 57],
    63: [55, 62],
}

# 辺。ただし角は除く
EDGES = [
    1, 2, 3, 4, 5, 6,
    8, 16, 24, 32, 40, 48,
    15, 23, 31, 39, 47, 55,
    57, 58, 59, 60, 61, 62,
]

# 8方向
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


# 各局面段階ごとの初期重み
# 後で強化学習する場合は、この値を学習対象にする

DEFAULT_FEATURE_WEIGHTS = {
    "opening": {
        "stone": -2.0,
        "mobility": 8.0,
        "position": 0.5,
        "corner": 30.0,
        "x_square": -20.0,
        "c_square": -10.0,
        "edge": 2.0,
    },
    "middle": {
        "stone": 0.2,
        "mobility": 6.0,
        "position": 1.5,
        "corner": 40.0,
        "x_square": -25.0,
        "c_square": -12.0,
        "edge": 3.0,
    },
    "end": {
        "stone": 8.0,
        "mobility": 1.0,
        "position": 0.3,
        "corner": 50.0,
        "x_square": -10.0,
        "c_square": -5.0,
        "edge": 5.0,
    },
}

def load_feature_weights(weights_file=GA_BEST_WEIGHTS_FILE):
    """
    学習済み重みJSONを読み込む。
    
    middle のみ学習したJSONにも、
    opening/middle/end 全体を含むJSONにも対応する。
    """
    import copy

    weights = copy.deepcopy(DEFAULT_FEATURE_WEIGHTS)

    try:
        with open(weights_file, "r", encoding="utf-8") as f:
            learned = json.load(f)
    except FileNotFoundError:
        print(f"[Weights] {weights_file} が見つかりません。初期重みを使用します。")
        return weights
    
    if "middle" in learned:
        weights["middle"].update(learned["middle"])

    return weights

FEATURE_WEIGHTS = load_feature_weights()


def get_phase(pos, my_color):
    """
    石数から序盤・中盤・終盤を判定する。
    終盤読み切りに入る前の評価関数用。
    """
    opp_color = pyrev.to_opponent_color(my_color)

    my_count = pos.get_disc_count_of(my_color)
    opp_count = pos.get_disc_count_of(opp_color)

    total_discs = my_count + opp_count

    if total_discs <= 20:
        return "opening"
    elif total_discs <= 40:
        return "middle"
    else:
        return "end"


def count_legal_moves_for(pos, color):
    """
    指定した色の合法手数を数える。
    
    PyRevでは基本的に side_to_move の合法手を返すため、
    必要に応じて copy して do_pass() する。
    """
    if pos.side_to_move == color:
        return len(list(pos.get_legal_moves()))

    copy_pos = pos.copy()
    copy_pos.do_pass()
    return len(list(copy_pos.get_legal_moves()))


def count_position_score(pos, my_color, opp_color):
    """
    盤面位置評価。
    EVAL_TABLEを使って、自分の石なら加点、相手の石なら減点。
    """
    score = 0

    for board_index in range(64):
        color = pos.get_square_color_at(board_index)

        if color == my_color:
            score += EVAL_TABLE[board_index]
        elif color == opp_color:
            score -= EVAL_TABLE[board_index]

    return score


def count_corner_score(pos, my_color, opp_color):
    """
    角の数。
    自分の角数 - 相手の角数。
    """
    score = 0

    for index in CORNERS:
        color = pos.get_square_color_at(index)

        if color == my_color:
            score += 1
        elif color == opp_color:
            score -= 1

    return score


def count_danger_square_score(pos, my_color, opp_color, square_map):
    """
    X-square / C-square の危険度を計算する。
    
    角が空いている場合のみ、その周辺マスを危険マスとして評価する。
    自分が危険マスに置いている場合は +1。
    相手が危険マスに置いている場合は -1。
    
    ただし、この特徴量には負の重みを掛けるので、
    自分が危険マスにいるほど評価値は下がる。
    """
    score = 0

    for corner, danger_squares in square_map.items():
        corner_color = pos.get_square_color_at(corner)

        # 角がすでに取られている場合、その周辺は危険マスとして扱わない
        if corner_color == my_color or corner_color == opp_color:
            continue

        for index in danger_squares:
            color = pos.get_square_color_at(index)

            if color == my_color:
                score += 1
            elif color == opp_color:
                score -= 1

    return score


def count_edge_score(pos, my_color, opp_color):
    """
    辺の石数。
    自分の辺の石数 - 相手の辺の石数。
    """
    score = 0

    for index in EDGES:
        color = pos.get_square_color_at(index)

        if color == my_color:
            score += 1
        elif color == opp_color:
            score -= 1

    return score


def is_empty(pos, index):
    """
    そのマスが空かどうかを判定する。
    PyRevの空マス定数に依存しないように、
    黒でも白でもなければ空とみなす。
    """
    color = pos.get_square_color_at(index)
    return color != pyrev.BLACK and color != pyrev.WHITE


def has_empty_neighbor(pos, index):
    """
    指定マスの周囲8方向に空きマスがあるか。
    """
    row = index // 8
    col = index % 8

    for dr, dc in DIRECTIONS:
        nr = row + dr
        nc = col + dc

        if 0 <= nr < 8 and 0 <= nc < 8:
            neighbor_index = nr * 8 + nc

            if is_empty(pos, neighbor_index):
                return True

    return False


# def count_frontier_score(pos, my_color, opp_color):
#     """
#     フロンティア石の数。
    
#     フロンティア石とは、周囲に空きマスがある石のこと。
#     ひっくり返されやすく、不安定な石なので少ない方がよい。
    
#     自分のフロンティア石数 - 相手のフロンティア石数 を返す。
#     この特徴量には負の重みを掛ける。
#     """
#     score = 0

#     for index in range(64):
#         color = pos.get_square_color_at(index)

#         if color != my_color and color != opp_color:
#             continue

#         if has_empty_neighbor(pos, index):
#             if color == my_color:
#                 score += 1
#             elif color == opp_color:
#                 score -= 1

#     return score


def extract_features(pos, my_color):
    """
    局面から評価用の特徴量を取り出す。
    
    すべて「my_colorから見て良いなら正、悪いなら負」
    になるようにする。
    """
    opp_color = pyrev.to_opponent_color(my_color)

    my_count = pos.get_disc_count_of(my_color)
    opp_count = pos.get_disc_count_of(opp_color)

    stone_score = my_count - opp_count

    my_moves = count_legal_moves_for(pos, my_color)
    opp_moves = count_legal_moves_for(pos, opp_color)
    mobility_score = my_moves - opp_moves

    position_score = count_position_score(pos, my_color, opp_color)
    corner_score = count_corner_score(pos, my_color, opp_color)
    x_square_score = count_danger_square_score(
        pos,
        my_color,
        opp_color,
        X_SQUARES
    )
    c_square_score = count_danger_square_score(
        pos,
        my_color,
        opp_color,
        C_SQUARES
    )
    edge_score = count_edge_score(pos, my_color, opp_color)
    # frontier_score = count_frontier_score(pos, my_color, opp_color)

    return {
        "stone": stone_score,
        "mobility": mobility_score,
        "position": position_score,
        "corner": corner_score,
        "x_square": x_square_score,
        "c_square": c_square_score,
        "edge": edge_score,
        # "frontier": frontier_score,
    }


def evaluate(pos, my_color):
    """
    特徴量追加版の評価関数。
    
    評価値 = 特徴量 × 重み の合計。
    """
    phase = get_phase(pos, my_color)
    features = extract_features(pos, my_color)
    weights = FEATURE_WEIGHTS[phase]

    score = 0.0

    for name, value in features.items():
        score += weights[name] * value

    return score

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