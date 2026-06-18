"""
agent.py - Optimized αβ agent (ab_rainforce_no_book.py の改良版)

ab_rainforce_no_book.py からの改善点:
  ④ copy_to() + do_move() + SearchContext による位置プール
     - pos.copy() の代わりに既存オブジェクトへの copy_to() でオブジェクト生成コストを削減
     - do_move_at() の代わりに合法手判定なしの do_move() で着手コストを削減
     - order_moves 内で calc_flip_discs() を先行計算し、探索ループで再利用することで
       calc_flip_discs() の二重呼び出しを排除
  ⑤ 置換表キーを calc_hash_code() (Zobrist) に一本化
     - PyRev が calc_hash_code() を持つことを GitHub README で確認済み
     - 64マス走査のフォールバックを削除し、整数キーに一本化
  ⑥ 特徴量計算の最適化
     - player_disc_count / opponent_disc_count / empty_square_count プロパティで O(1) 取得
     - get_square_owner_at() (条件分岐なし、README で推奨) でマス所有者を取得
     - αβ 探索中に position_score を差分更新し、末端ノードでの 64 マスフル走査を排除
       (mobility 以外の特徴量は安価なため、position_score のみ差分更新する)
"""

import math
import os
import json

import numpy as np
import pyrev
from pyrev import BoardCoordinateIterator


# ==============================
# 定数
# ==============================

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

# 角・危険マス・辺
CORNERS     = [0, 7, 56, 63]
CORNER_SET  = frozenset(CORNERS)   # O(1) 判定用

X_SQUARES = {0: [9],  7: [14], 56: [49], 63: [54]}
C_SQUARES = {0: [1, 8], 7: [6, 15], 56: [48, 57], 63: [55, 62]}

EDGES = [
     1,  2,  3,  4,  5,  6,
     8, 16, 24, 32, 40, 48,
    15, 23, 31, 39, 47, 55,
    57, 58, 59, 60, 61, 62,
]


# ==============================
# 置換表 (Transposition Table)
# ==============================

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# 終盤の完全読みは深さ概念がないため大きな値で代用
TT_FULL_DEPTH = 1 << 30


def _make_tt_key(pos):
    """
    置換表キーを生成する。

    GitHub README で calc_hash_code() (Zobrist) の存在が確認されているため、
    フォールバックなしで直接使用する。
    side_to_move をキーに含めることで、同一石配置で手番が異なる局面を区別する。
    """
    return int(pos.calc_hash_code()) * 2 + int(pos.side_to_move)


def _store_tt(tt, key, depth, score, orig_alpha, orig_beta):
    """
    探索結果を置換表に格納する。
    同キーの既存エントリより探索深さが深い場合のみ上書きする。
    """
    if score <= orig_alpha:
        flag = TT_UPPER
    elif score >= orig_beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    existing = tt.get(key)
    if existing is None or existing[0] <= depth:
        tt[key] = (depth, flag, score)


# ==============================
# SearchContext (④ 位置プール)
# ==============================

class SearchContext:
    """
    1回の探索 (alpha_beta_search_only 呼び出し) で使い捨てる作業領域。

    pos_pool:
        深さ(ply)ごとに事前確保した Position オブジェクトのリスト。
        copy_to() + do_move() で使い回すことで、
        各ノードでの pos.copy() によるオブジェクト生成コストを排除する。

        pool[ply] は "ply 番目に再帰する際の子ノード位置" として使用する。
        ply=0 ... alpha_beta_search_only の直接の子
        ply=1 ... alpha_beta_rec(ply=1) の子
        ...

    scratch_pos:
        order_moves / extract_features 内での一時的な局面計算専用。
        探索ループと競合しないよう、必ず使用後に上書きされる形で使う。
    """
    # DEPTH(4) + ENDGAME_EMPTY_LIMIT(14) + パスの余裕 = 40 で十分
    MAX_PLY = 40

    def __init__(self):
        self.pos_pool   = [pyrev.Position() for _ in range(self.MAX_PLY)]
        self.scratch_pos = pyrev.Position()


# ==============================
# 重み設定
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GA_BEST_WEIGHTS_FILE = os.path.join(BASE_DIR, "ga_feature_weights_best.json")


DEFAULT_FEATURE_WEIGHTS = {
    "opening": {
        "stone": -7.582455331455142,
        "mobility": 16.18866956229727,
        "position": 0.12725804041507488,
        "corner": 19.503799787093783,
        "x_square": -40.8086233830389,
        "c_square": -8.453664200147834,
        "edge": 1.8249661016193859
    },
    "middle": {
        "stone": -1.3604442968945822,
        "mobility": 2.702777632920106,
        "position": 0.10845898171496211,
        "corner": 24.919928660762068,
        "x_square": -14.297343155358652,
        "c_square": -7.238055035591753,
        "edge": 2.65844257698898
    },
    "end": {
        "stone": 1.856383977915157,
        "mobility": 1.3418858389925643,
        "position": 4.733433589139915,
        "corner": 52.43996828837947,
        "x_square": -9.19687830824963,
        "c_square": -4.671500311679356,
        "edge": 2.850741580985523
    },
}


def load_feature_weights(weights_file=GA_BEST_WEIGHTS_FILE):
    """
    学習済み重み JSON を読み込む。
    opening / middle / end のいずれか、または複数を含む JSON に対応する。
    """
    import copy
    weights = copy.deepcopy(DEFAULT_FEATURE_WEIGHTS)
    try:
        with open(weights_file, "r", encoding="utf-8") as f:
            learned = json.load(f)
    except FileNotFoundError:
        print(f"[Weights] {weights_file} が見つかりません。初期重みを使用します。")
        return weights
    for phase in ("opening", "middle", "end"):
        if phase in learned:
            weights[phase].update(learned[phase])
    return weights


FEATURE_WEIGHTS = load_feature_weights()


# ==============================
# 局面フェーズ判定
# ==============================

def get_phase(pos):
    """
    石数から序盤・中盤・終盤を判定する。
    disc_count プロパティは O(1)。
    """
    total = pos.disc_count
    if total <= 20:
        return "opening"
    elif total <= 40:
        return "middle"
    else:
        return "end"


# ==============================
# 特徴量計算 (⑥ 最適化版)
# ==============================

def _compute_position_score_full(pos):
    """
    pos の position_score を全マス走査でフル計算する。
    alpha_beta_search_only の呼び出し時に一度だけ使う。
    以降の差分更新のための初期値として使用する。

    get_square_owner_at() は get_square_color_at() より高速 (README 参照)。
    PLAYER = side_to_move の石、OPPONENT = 相手の石。
    """
    score = 0
    for i in range(64):
        owner = pos.get_square_owner_at(i)
        if owner == pyrev.PLAYER:
            score += EVAL_TABLE[i]
        elif owner == pyrev.OPPONENT:
            score -= EVAL_TABLE[i]
    return score


def extract_features(pos, ctx, cur_pos_score):
    """
    評価用の特徴量を取り出す。

    cur_pos_score:
        探索中に差分更新してきた position_score。
        64 マスフル走査が不要なため、末端ノードの評価が高速になる。

    get_square_owner_at() を全体的に使用:
        PLAYER  = pos.side_to_move の石
        OPPONENT = 相手の石
        NULL_OWNER = 空マス
    """
    # ---- 石数 (O(1) プロパティ) ----
    stone_score = pos.player_disc_count - pos.opponent_disc_count

    # ---- 着手可能数 ----
    my_moves = sum(1 for _ in pos.get_legal_moves())

    # 相手の着手可能数: scratch_pos に do_pass() してカウント
    # (pos.copy() を避け、事前確保の scratch_pos を使い回す)
    pos.copy_to(ctx.scratch_pos)
    ctx.scratch_pos.do_pass()
    opp_moves = sum(1 for _ in ctx.scratch_pos.get_legal_moves())

    mobility_score = my_moves - opp_moves

    # ---- 位置スコア (差分更新済みの値を直接使用) ----
    position_score = cur_pos_score

    # ---- 角 ----
    corner_score = 0
    for c in CORNERS:
        owner = pos.get_square_owner_at(c)
        if owner == pyrev.PLAYER:
            corner_score += 1
        elif owner == pyrev.OPPONENT:
            corner_score -= 1

    # ---- X-square / C-square (角が空いている場合のみ危険マスとして評価) ----
    x_square_score = 0
    for corner, danger_squares in X_SQUARES.items():
        if pos.get_square_owner_at(corner) != pyrev.NULL_OWNER:
            continue
        for idx in danger_squares:
            owner = pos.get_square_owner_at(idx)
            if owner == pyrev.PLAYER:
                x_square_score += 1
            elif owner == pyrev.OPPONENT:
                x_square_score -= 1

    c_square_score = 0
    for corner, danger_squares in C_SQUARES.items():
        if pos.get_square_owner_at(corner) != pyrev.NULL_OWNER:
            continue
        for idx in danger_squares:
            owner = pos.get_square_owner_at(idx)
            if owner == pyrev.PLAYER:
                c_square_score += 1
            elif owner == pyrev.OPPONENT:
                c_square_score -= 1

    # ---- 辺 ----
    edge_score = 0
    for idx in EDGES:
        owner = pos.get_square_owner_at(idx)
        if owner == pyrev.PLAYER:
            edge_score += 1
        elif owner == pyrev.OPPONENT:
            edge_score -= 1

    return {
        "stone":    stone_score,
        "mobility": mobility_score,
        "position": position_score,
        "corner":   corner_score,
        "x_square": x_square_score,
        "c_square": c_square_score,
        "edge":     edge_score,
    }


def evaluate(pos, ctx, cur_pos_score):
    """
    特徴量 × 重みの線形評価関数。
    cur_pos_score は αβ 探索中に差分更新された position_score。
    """
    phase    = get_phase(pos)
    features = extract_features(pos, ctx, cur_pos_score)
    weights  = FEATURE_WEIGHTS[phase]
    return sum(weights[name] * value for name, value in features.items())


# ==============================
# 手の並べ替え (④⑥ 統合最適化版)
# ==============================

def order_moves_with_flips(pos, actions, ctx):
    """
    手の並べ替えを行い、(action, flip_bits) のペアリストを返す。

    ④ calc_flip_discs() を先行計算して返すことで、
       呼び出し元の探索ループで do_move(action, flip) として再利用でき、
       calc_flip_discs() の二重呼び出しを排除できる。

    並べ替えスコアは軽量な3指標で計算:
      - EVAL_TABLE[action] : 着手マスの位置評価
      - 角スコア(100倍)     : 着手後の (自分の角 - 相手の角)
      - 相手の着手可能数    : 着手後に相手が打てる手数 (少ないほど良い)

    scratch_pos は 1 手ずつ上書き使用する (競合なし)。

    角スコアは着手後の scratch_pos から get_square_owner_at() で取得する。
    do_move() 後は side_to_move が相手に切り替わるため、
      scratch_pos.PLAYER   = 相手の石
      scratch_pos.OPPONENT = 自分 (元の side_to_move) の石
    となる点に注意。
    """
    scored = []
    for action in actions:
        action_int = int(action)
        flip_bits  = pos.calc_flip_discs(action)

        # ④ scratch_pos に着手して後続の指標を計算
        pos.copy_to(ctx.scratch_pos)
        ctx.scratch_pos.do_move(action, flip_bits)

        # 位置評価テーブル
        score = EVAL_TABLE[action_int]

        # 角スコア (do_move 後は PLAYER/OPPONENT が反転している)
        corner_score = 0
        for c in CORNERS:
            owner = ctx.scratch_pos.get_square_owner_at(c)
            if owner == pyrev.OPPONENT:    # 自分 (元 PLAYER) の石
                corner_score += 1
            elif owner == pyrev.PLAYER:    # 相手の石
                corner_score -= 1
        score += corner_score * 100

        # 相手の着手可能数 (do_move 後は相手が side_to_move)
        opp_moves = sum(1 for _ in ctx.scratch_pos.get_legal_moves())
        score -= opp_moves * 10

        scored.append((score, action, flip_bits))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [(a, f) for _, a, f in scored]


def order_moves_endgame_with_flips(pos, actions, ctx):
    """
    終盤探索用の軽量な手の並べ替え。
    相手の着手可能数のみで順序付けし、flip_bits を返す。
    """
    scored = []
    for action in actions:
        flip_bits = pos.calc_flip_discs(action)

        pos.copy_to(ctx.scratch_pos)
        ctx.scratch_pos.do_move(action, flip_bits)
        opp_moves = sum(1 for _ in ctx.scratch_pos.get_legal_moves())

        scored.append((-opp_moves, action, flip_bits))

    scored.sort()
    return [(a, f) for _, a, f in scored]


# ==============================
# 終盤探索 (完全読み)
# ==============================

def endgame_search(pos, alpha, beta, tt, ctx, ply):
    """
    終盤の完全読み探索。

    player_disc_count / opponent_disc_count は O(1) プロパティ。
    copy_to() + do_move() でオブジェクト生成を排除。
    """
    if pos.is_gameover():
        # player_disc_count は pos.side_to_move の石数
        if pos.player_disc_count > pos.opponent_disc_count:
            return WIN_SCORE
        elif pos.player_disc_count < pos.opponent_disc_count:
            return -WIN_SCORE
        return 0

    key          = _make_tt_key(pos)
    orig_alpha   = alpha
    orig_beta    = beta

    entry = tt.get(key)
    if entry is not None:
        _, flag, value = entry
        if flag == TT_EXACT:
            return value
        elif flag == TT_LOWER:
            if value > alpha:
                alpha = value
        elif flag == TT_UPPER:
            if value < beta:
                beta = value
        if alpha >= beta:
            return value

    actions = list(pos.get_legal_moves())

    if not actions:
        # パス: pool[ply] に copy して do_pass()
        next_pos = ctx.pos_pool[ply]
        pos.copy_to(next_pos)
        next_pos.do_pass()

        score = -endgame_search(next_pos, -beta, -alpha, tt, ctx, ply + 1)

        _store_tt(tt, key, TT_FULL_DEPTH, score, orig_alpha, orig_beta)
        return score

    action_flip_pairs = order_moves_endgame_with_flips(pos, actions, ctx)

    best_score = -math.inf

    for action, flip_bits in action_flip_pairs:
        next_pos = ctx.pos_pool[ply]
        pos.copy_to(next_pos)
        next_pos.do_move(action, flip_bits)  # ④ 合法手判定なし

        score = -endgame_search(next_pos, -beta, -alpha, tt, ctx, ply + 1)

        if score > best_score:
            best_score = score

        if score == WIN_SCORE:
            _store_tt(tt, key, TT_FULL_DEPTH, WIN_SCORE, orig_alpha, orig_beta)
            return WIN_SCORE

        if score >= beta:
            alpha = score
            break

        if score > alpha:
            alpha = score

    _store_tt(tt, key, TT_FULL_DEPTH, best_score, orig_alpha, orig_beta)
    return best_score


# ==============================
# αβ 探索 (メイン)
# ==============================

def alpha_beta_rec(pos, alpha, beta, depth, tt, ctx, ply, cur_pos_score):
    """
    αβ 探索の再帰関数。

    ply         : pos_pool のインデックス。子ノードは pool[ply] に copy_to する。
    cur_pos_score: 差分更新された position_score (pos.side_to_move 視点)。
                  終端ノード評価で 64 マス再走査を避けるために使用する。

    差分更新ルール:
      - 着手 (action, flip_bits) 後の位置スコア変化量:
          delta = EVAL_TABLE[action] + 2 * sum(EVAL_TABLE[f] for f in flipped)
      - 次ノード (相手視点) の cur_pos_score = -(cur_pos_score + delta)
      - パス後: cur_pos_score = -cur_pos_score (視点反転のみ、盤面変化なし)
    """
    # 終盤読み切りへ移行
    if pos.empty_square_count <= ENDGAME_EMPTY_LIMIT:
        return endgame_search(pos, alpha, beta, tt, ctx, ply)

    # 末端ノード評価
    if depth == 0 or pos.is_gameover():
        return evaluate(pos, ctx, cur_pos_score)

    # 置換表ルックアップ (⑤ 整数キー)
    key        = _make_tt_key(pos)
    orig_alpha = alpha
    orig_beta  = beta

    entry = tt.get(key)
    if entry is not None:
        e_depth, flag, value = entry
        if e_depth >= depth:
            if flag == TT_EXACT:
                return value
            elif flag == TT_LOWER:
                if value > alpha:
                    alpha = value
            elif flag == TT_UPPER:
                if value < beta:
                    beta = value
            if alpha >= beta:
                return value

    actions = list(pos.get_legal_moves())

    if not actions:
        # パス
        next_pos = ctx.pos_pool[ply]
        pos.copy_to(next_pos)
        next_pos.do_pass()

        score = -alpha_beta_rec(
            next_pos, -beta, -alpha, depth - 1, tt, ctx, ply + 1,
            -cur_pos_score  # 視点のみ反転
        )

        _store_tt(tt, key, depth, score, orig_alpha, orig_beta)
        return score

    # ④ flip を先行計算して返す (do_move での再計算を排除)
    action_flip_pairs = order_moves_with_flips(pos, actions, ctx)

    best_score = -math.inf

    for action, flip_bits in action_flip_pairs:
        # ④ copy_to() + do_move() でオブジェクト生成を排除
        next_pos = ctx.pos_pool[ply]
        pos.copy_to(next_pos)
        next_pos.do_move(action, flip_bits)

        # ⑥ position_score を差分更新
        delta = EVAL_TABLE[int(action)]
        for fc in BoardCoordinateIterator(flip_bits):
            delta += 2 * EVAL_TABLE[int(fc)]
        next_pos_score = -(cur_pos_score + delta)

        score = -alpha_beta_rec(
            next_pos, -beta, -alpha, depth - 1, tt, ctx, ply + 1,
            next_pos_score
        )

        if score > best_score:
            best_score = score

        if score >= beta:
            alpha = score
            break

        if score > alpha:
            alpha = score

    _store_tt(tt, key, depth, best_score, orig_alpha, orig_beta)
    return best_score


def alpha_beta_search_only(pos, depth):
    """
    αβ 探索のエントリポイント。

    SearchContext と置換表をここで生成し、探索が終わったら破棄する。
    cur_pos_score を _compute_position_score_full() で初期化し、
    以降は差分更新のみで末端ノードまで引き渡す。
    """
    actions = list(pos.get_legal_moves())
    if not actions:
        return -1

    ctx           = SearchContext()
    tt            = {}
    cur_pos_score = _compute_position_score_full(pos)  # ⑥ 初期値計算 (1回のみ)

    best_action = -1
    alpha       = -math.inf
    beta        = math.inf

    # ④ flip を先行計算して返す
    action_flip_pairs = order_moves_with_flips(pos, actions, ctx)

    for action, flip_bits in action_flip_pairs:
        # ④ pool[0] を使って子ノードを生成
        next_pos = ctx.pos_pool[0]
        pos.copy_to(next_pos)
        next_pos.do_move(action, flip_bits)

        # ⑥ position_score 差分更新
        delta = EVAL_TABLE[int(action)]
        for fc in BoardCoordinateIterator(flip_bits):
            delta += 2 * EVAL_TABLE[int(fc)]
        next_pos_score = -(cur_pos_score + delta)

        score = -alpha_beta_rec(
            next_pos, -beta, -alpha, depth - 1, tt, ctx, 1,
            next_pos_score
        )

        if score > alpha:
            alpha       = score
            best_action = action

    return best_action


def alpha_beta(pos, depth, test_mode=False):
    """
    αβ エージェントの公開インターフェース。
    ab_rainforce_no_book.py と同じシグネチャを維持する。
    """
    best_action = alpha_beta_search_only(pos, depth)

    if test_mode:
        if best_action is None or int(best_action) < 0:
            print("[AlphaBeta] 選択手: None")
        else:
            print(f"[AlphaBeta] 選択手: {pyrev.coord_to_str(np.int8(best_action))}")

    return best_action
