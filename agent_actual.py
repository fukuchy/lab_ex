"""
agent.py - Optimized αβ agent (ab_rainforce_no_book.py の改良版)

ab_rainforce_no_book.py からの改善点:
  ④ copy_to() + do_move() + SearchContext による位置プール
     - pos.copy() の代わりに既存オブジェクトへの copy_to() でオブジェクト生成コストを削減
     - do_move_at() の代わりに合法手判定なしの do_move() で着手コストを削減
  ⑤ 置換表キーを calc_hash_code() (Zobrist) に一本化
     - 64マス走査のフォールバックを削除し、整数キーに一本化
  ⑥(旧) 特徴量計算の最適化 (position_score の差分更新)
     - player_disc_count / opponent_disc_count / empty_square_count プロパティで O(1) 取得
     - get_square_owner_at() (条件分岐なし、README で推奨) でマス所有者を取得

さらに以下を追加実装:
  ① order_moves_with_flips の軽量化
     - 候補手ごとの copy_to + do_move + 角4マス走査 + get_legal_moves を全廃
     - EVAL_TABLE (角・X/C-squareの価値を内包済み) + フリップ数のみで並べ替え
  ② corner_score / edge_score / x_square_score / c_square_score の差分更新
     - corner_score, edge_score: ビットマスク + 着手マス判定のみで常に O(1) 差分更新
       (角はフリップされない、辺は popcount で正確に計算できるため)
     - x_square_score, c_square_score: 角の占有状態に依存するため、
       「角 or 危険マスに変化があったときだけ」next_pos からフル再計算する
       fast path (変化なし: 単純に符号反転のみ) / slow path (まれに発生するフル再計算)
       の2段構成にすることで、ほとんどのノードで O(1) 化する
  ⑥ SearchContext の使い回し
     - do_move の反復深化ループの外側で1回だけ生成し、全深さで使い回す
     - 毎深さ 40個 (MAX_PLY) の Position オブジェクトを再生成するコストを排除
  ⑦ 置換表の使い回し
     - tt も同様にループ外側で1回だけ生成し、深さをまたいで再利用する
     - _store_tt は depth が浅いエントリを上書きしない設計のため、
       浅い深さで得た探索結果を次の深さでも安全に活用できる
       (標準的な反復深化+置換表の手法そのもの)
"""

import math
import os
import json
import time

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
EDGE_SET = frozenset(EDGES)


# ==============================
# ビットマスク定数 (② 差分更新用)
# ==============================
#
# corner_score / edge_score を O(1) で差分更新するため、
# 各特徴量に対応するマスをビット集合として事前計算しておく。
# x_square_score / c_square_score は角の占有状態に依存するため、
# 「角 or 危険マスのいずれかが変化したか」を判定する DANGER_MASK として使う。

def _mask_from_squares(squares):
    m = 0
    for s in squares:
        m |= (1 << s)
    return m


CORNER_MASK = _mask_from_squares(CORNERS)
EDGE_MASK   = _mask_from_squares(EDGES)
XSQ_MASK    = _mask_from_squares([sq for lst in X_SQUARES.values() for sq in lst])
CSQ_MASK    = _mask_from_squares([sq for lst in C_SQUARES.values() for sq in lst])

# 角・X-square・C-square のいずれかに触れた場合のみ
# x_square_score / c_square_score の再計算が必要になる。
DANGER_MASK = CORNER_MASK | XSQ_MASK | CSQ_MASK


# ==============================
# 置換表 (Transposition Table)
# ==============================

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# 終盤の完全読みは深さ概念がないため大きな値で代用
TT_FULL_DEPTH = 1 << 30


# ==============================
# タイムアウト例外
# ==============================

class _SearchTimeout(Exception):
    """
    endgame_search からタイムアウトを通知する内部例外。
    do_move の反復深化ループでキャッチし、
    最後に完了した深さの結果を返すために使う。
    """
    pass


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
    # 盤面は64マスなので、パスを考慮しても ply が 64 を超えることは実質的にない。
    # ⑥ で SearchContext を反復深化の全深さで使い回すようになったため、
    # 安全マージンとして盤面サイズと同じ 64 に設定する。
    MAX_PLY = 64

    def __init__(self, deadline: float = math.inf):
        self.pos_pool    = [pyrev.Position() for _ in range(self.MAX_PLY)]
        self.scratch_pos = pyrev.Position()
        self.node_count  = 0          # タイムアウトチェック用ノードカウンタ
        self.deadline    = deadline   # 探索期限 (time.perf_counter() の絶対値)


# ==============================
# 重み設定
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GA_BEST_WEIGHTS_FILE = os.path.join(BASE_DIR, "ga_middle_best_weights.json")


DEFAULT_FEATURE_WEIGHTS = {
    "opening": {
        "stone": -1.057879575503218,
        "mobility": 8.414318362549801,
        "position": -0.009823972577531426,
        "corner": 15.661617473189732,
        "x_square": -106.74159322137517,
        "c_square": -6.844459394295814,
        "edge": -0.10787899484266494
    },
    "middle": {
        "stone": -1.8795933150765822,
        "mobility": 2.5289310439091235,
        "position": 0.013889032950499855,
        "corner": 9.131581671319736,
        "x_square": -22.964217605593127,
        "c_square": -5.459317219241519,
        "edge": 3.0695394948193213
    },
    "end": {
        "stone": 2.756986809785344,
        "mobility": 1.8956620499677932,
        "position": 0.5602700812797425,
        "corner": 97.77476018333562,
        "x_square": -14.224791094827243,
        "c_square": -5.339905587801405,
        "edge": 7.480266043841487
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

def _compute_danger_squares_full(pos):
    """
    x_square_score, c_square_score を pos から直接フル計算する。

    角が空いている場合のみ、その周辺マスを危険マスとして評価する。
    DANGER_MASK に変化があった場合 (② のレアパス) にのみ呼ばれる。
    呼び出し頻度が低いため、64マス全走査ではなく対象マスのみを
    直接チェックする従来通りのロジックで十分高速。
    """
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

    return x_square_score, c_square_score


def _compute_full_feat_state(pos):
    """
    探索開始時 (alpha_beta_search_only の呼び出し時) に1回だけ呼ばれる、
    feat_state = (position_score, corner_score, edge_score,
                  x_square_score, c_square_score)
    のフル計算。以降は _apply_move / _apply_pass で差分更新される。

    position_score・corner_score・edge_score は1回の64マス走査でまとめて
    計算する (それぞれ別ループにする必要がないため統合)。
    x_square_score・c_square_score は角の占有状態に依存するため
    _compute_danger_squares_full() に分離する。
    """
    pos_score    = 0
    corner_score = 0
    edge_score   = 0

    for i in range(64):
        owner = pos.get_square_owner_at(i)
        bit = 1 << i
        if owner == pyrev.PLAYER:
            pos_score += EVAL_TABLE[i]
            if bit & CORNER_MASK:
                corner_score += 1
            if bit & EDGE_MASK:
                edge_score += 1
        elif owner == pyrev.OPPONENT:
            pos_score -= EVAL_TABLE[i]
            if bit & CORNER_MASK:
                corner_score -= 1
            if bit & EDGE_MASK:
                edge_score -= 1

    xsq_score, csq_score = _compute_danger_squares_full(pos)

    return (pos_score, corner_score, edge_score, xsq_score, csq_score)


def _apply_move(action, flip_bits, next_pos, cur_feat):
    """
    ② 着手 (action, flip_bits) 適用後の feat_state を差分更新で計算する。

    cur_feat: 着手前の feat_state (pos.side_to_move 視点)
    next_pos: 着手後の局面 (do_move 済み)

    戻り値は次ノード (next_pos.side_to_move 視点、つまり符号反転後) の feat_state。

    - position_score: 着手マス + フリップマスの EVAL_TABLE 差分 (従来通り)
    - corner_score  : 角はフリップされないため、着手マスが角かどうかのみで
                      常に正確に差分更新できる (slow path 不要)
    - edge_score    : フリップビットと EDGE_MASK の popcount で
                      常に正確に差分更新できる (slow path 不要)
    - x_square_score / c_square_score:
        角の占有状態に依存するため、DANGER_MASK (角+X-square+C-square) に
        触れた場合のみ next_pos からフル再計算する (slow path)。
        触れていない場合は符号反転のみで済む (fast path、ほとんどのケース)。
    """
    action_int = int(action)
    flip_int   = int(flip_bits)

    cur_pos_score, cur_corner, cur_edge, cur_xsq, cur_csq = cur_feat

    # position_score 差分 (従来通り)
    delta_pos = EVAL_TABLE[action_int]
    for fc in BoardCoordinateIterator(flip_bits):
        delta_pos += 2 * EVAL_TABLE[int(fc)]

    # corner_score 差分 (常に O(1)、角はフリップされないため着手マスのみ判定)
    delta_corner = 1 if action_int in CORNER_SET else 0

    # edge_score 差分 (常に O(1)、ビットマスク popcount)
    flip_edge_hits = bin(flip_int & EDGE_MASK).count('1')
    action_is_edge = 1 if action_int in EDGE_SET else 0
    delta_edge = action_is_edge + 2 * flip_edge_hits

    new_pos_score = -(cur_pos_score + delta_pos)
    new_corner    = -(cur_corner + delta_corner)
    new_edge      = -(cur_edge + delta_edge)

    # x_square / c_square: 角 or 危険マスに変化があった場合のみフル再計算
    touched_mask = (1 << action_int) | flip_int
    if touched_mask & DANGER_MASK:
        new_xsq, new_csq = _compute_danger_squares_full(next_pos)
    else:
        new_xsq = -cur_xsq
        new_csq = -cur_csq

    return (new_pos_score, new_corner, new_edge, new_xsq, new_csq)


def _apply_pass(cur_feat):
    """
    パス時の feat_state 更新。盤面は変化しないため符号反転のみ。
    """
    a, b, c, d, e = cur_feat
    return (-a, -b, -c, -d, -e)


def extract_features(pos, ctx, feat_state):
    """
    評価用の特徴量を取り出す。

    feat_state = (position_score, corner_score, edge_score,
                  x_square_score, c_square_score)
        探索中に _apply_move / _apply_pass で差分更新されてきた値。
        ② により、これら4特徴量は末端ノードでの再計算が不要になっている。

    mobility のみ、合法手生成が必要なため引き続き pos から直接計算する
    (差分更新が困難なため、従来通り)。
    stone は player_disc_count / opponent_disc_count で O(1) 取得。
    """
    pos_score, corner_score, edge_score, x_square_score, c_square_score = feat_state

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

    return {
        "stone":    stone_score,
        "mobility": mobility_score,
        "position": pos_score,
        "corner":   corner_score,
        "x_square": x_square_score,
        "c_square": c_square_score,
        "edge":     edge_score,
    }


def evaluate(pos, ctx, feat_state):
    """
    特徴量 × 重みの線形評価関数。
    feat_state は αβ 探索中に差分更新された (② 参照)。
    """
    phase    = get_phase(pos)
    features = extract_features(pos, ctx, feat_state)
    weights  = FEATURE_WEIGHTS[phase]
    return sum(weights[name] * value for name, value in features.items())


# ==============================
# 手の並べ替え (④⑥ 統合最適化版)
# ==============================

# フリップ数1個あたりの並べ替えペナルティ。
# 多く取りすぎる手は中盤でモビリティを失いやすいという定番ヒューリスティック。
_FLIP_ORDER_PENALTY = 2.0


def order_moves_with_flips(pos, actions):
    """
    手の並べ替えを行い、(action, flip_bits) のペアリストを返す (① 軽量版)。

    【旧実装の問題点】
    候補手ごとに copy_to + do_move + 角4マス走査 + get_legal_moves を
    行っており、これは中盤探索の全ノードに対して発生する重いコストだった。

    【改善点】
    copy_to を一切行わず、以下の2指標のみで並べ替える。
      - EVAL_TABLE[action] : 着手マスの位置評価
                             (角+120, X-square-40, C-square-20 などを
                              すでに内包しているため、これ単体でも十分な目安)
      - フリップ数         : 取りすぎる手にペナルティを与える
                             (calc_flip_discs() は do_move() に再利用するため
                              どのみち必要であり、追加コストはほぼゼロ)
    """
    scored = []
    for action in actions:
        action_int = int(action)
        flip_bits  = pos.calc_flip_discs(action)
        flip_count = bin(int(flip_bits)).count('1')

        score = EVAL_TABLE[action_int] - flip_count * _FLIP_ORDER_PENALTY

        scored.append((-score, action, flip_bits))

    scored.sort()
    return [(a, f) for _, a, f in scored]


def order_moves_endgame_with_flips(pos, actions):
    """
    終盤探索用の軽量な手の並べ替え。

    【旧実装の問題点】
    候補手ごとに copy_to + do_move + get_legal_moves を行っていた。
    終盤は数百万ノードに達するため、このオーバーヘッドが探索時間の
    大部分を占めるボトルネックになっていた。

    【改善点】
    copy_to を一切行わず、calc_flip_discs() の結果から
    フリップ数をカウントするだけで並べ替えスコアを計算する。
    flip_bits は do_move() に再利用するため calc_flip_discs() は
    どのみち必要であり、追加コストはほぼゼロ。

    角への着手を最優先し、次にフリップ数が多い手を優先する。
    """
    scored = []
    for action in actions:
        action_int = int(action)
        flip_bits  = pos.calc_flip_discs(action)
        flip_count = bin(int(flip_bits)).count('1')

        score = 10000 if action_int in CORNER_SET else 0
        score += flip_count

        scored.append((-score, action, flip_bits))

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

    タイムアウト:
        4096 ノードごとに time.perf_counter() を確認し、
        ctx.deadline を超えていたら _SearchTimeout を送出する。
        do_move の反復深化ループがこれをキャッチし、
        最後に完了した深さの結果を返す。
    """
    # タイムアウトチェック (4096ノードごと、ビット演算で剰余を回避)
    ctx.node_count += 1
    if not (ctx.node_count & 4095):
        if time.perf_counter() > ctx.deadline:
            raise _SearchTimeout()

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

    # ctx 引数を削除した軽量版
    action_flip_pairs = order_moves_endgame_with_flips(pos, actions)

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

def alpha_beta_rec(pos, alpha, beta, depth, tt, ctx, ply, feat_state):
    """
    αβ 探索の再帰関数。

    ply        : pos_pool のインデックス。子ノードは pool[ply] に copy_to する。
    feat_state : (position_score, corner_score, edge_score,
                  x_square_score, c_square_score) の差分更新済みタプル
                 (pos.side_to_move 視点)。終端ノード評価で再走査を避けるために使用する。

    各特徴量の差分更新ルールは _apply_move / _apply_pass を参照。

    タイムアウト:
        endgame_search と同様に 4096 ノードごとに ctx.deadline を確認し、
        超えていたら _SearchTimeout を送出する。これにより、中盤探索の
        途中であっても time_limit_sec を過ぎた時点で打ち切れる。
    """
    # タイムアウトチェック (4096ノードごと、endgame_search と同じカウンタを共有)
    ctx.node_count += 1
    if not (ctx.node_count & 4095):
        if time.perf_counter() > ctx.deadline:
            raise _SearchTimeout()

    # 終盤読み切りへ移行
    if pos.empty_square_count <= ENDGAME_EMPTY_LIMIT:
        return endgame_search(pos, alpha, beta, tt, ctx, ply)

    # 末端ノード評価
    if depth == 0 or pos.is_gameover():
        return evaluate(pos, ctx, feat_state)

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
            _apply_pass(feat_state)
        )

        _store_tt(tt, key, depth, score, orig_alpha, orig_beta)
        return score

    # ① flip を先行計算して返す (do_move での再計算を排除)
    action_flip_pairs = order_moves_with_flips(pos, actions)

    best_score = -math.inf

    for action, flip_bits in action_flip_pairs:
        # ④ copy_to() + do_move() でオブジェクト生成を排除
        next_pos = ctx.pos_pool[ply]
        pos.copy_to(next_pos)
        next_pos.do_move(action, flip_bits)

        # ② feat_state を差分更新
        next_feat = _apply_move(action, flip_bits, next_pos, feat_state)

        score = -alpha_beta_rec(
            next_pos, -beta, -alpha, depth - 1, tt, ctx, ply + 1,
            next_feat
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


def alpha_beta_search_only(pos, depth, deadline: float = math.inf, ctx=None, tt=None):
    """
    αβ 探索のエントリポイント。

    ⑥⑦ ctx / tt を外部 (do_move) から受け取れるようにし、反復深化の
        全深さで使い回せるようにする。None の場合はこれまで通り
        この呼び出し内だけで使い捨てる ctx / tt を新規生成する
        (alpha_beta() からの単発呼び出しなど、反復深化を伴わない用途向け)。

    feat0 = _compute_full_feat_state(pos) で初期値を計算し、
    以降は _apply_move / _apply_pass による差分更新のみで末端ノードまで引き渡す。

    deadline : time.perf_counter() の絶対値。
               endgame_search がこの時刻を超えたら _SearchTimeout を送出する。
               do_move から呼ばれる場合に設定される。
    """
    actions = list(pos.get_legal_moves())
    if not actions:
        return -1, None

    if ctx is None:
        ctx = SearchContext(deadline=deadline)
    else:
        ctx.deadline = deadline

    if tt is None:
        tt = {}

    feat0 = _compute_full_feat_state(pos)  # ② 初期値計算 (1回のみ)

    best_action = -1
    alpha       = -math.inf
    beta        = math.inf

    # ① flip を先行計算して返す
    action_flip_pairs = order_moves_with_flips(pos, actions)

    for action, flip_bits in action_flip_pairs:
        # ④ pool[0] を使って子ノードを生成
        next_pos = ctx.pos_pool[0]
        pos.copy_to(next_pos)
        next_pos.do_move(action, flip_bits)

        # ② feat_state 差分更新
        next_feat = _apply_move(action, flip_bits, next_pos, feat0)

        score = -alpha_beta_rec(
            next_pos, -beta, -alpha, depth - 1, tt, ctx, 1,
            next_feat
        )

        if score > alpha:
            alpha       = score
            best_action = action

    return best_action, alpha


def alpha_beta(pos, depth, test_mode=False):
    """
    αβ エージェントの公開インターフェース。
    ab_rainforce_no_book.py と同じシグネチャを維持する。
    """
    best_action, _ = alpha_beta_search_only(pos, depth)

    if test_mode:
        if best_action is None or int(best_action) < 0:
            print("[AlphaBeta] 選択手: None")
        else:
            print(f"[AlphaBeta] 選択手: {pyrev.coord_to_str(np.int8(best_action))}")

    return best_action


# ==============================
# 反復深化αβ (時間制限付き)
# ==============================

def do_move(position: pyrev.Position, time_limit_sec: float, test_mode: bool = False) -> int:
    time_limit_sec_set = time_limit_sec - 0.05
    """
    反復深化αβ探索で time_limit_sec 秒以内に最善手を返す。

    深さ 1 から順に alpha_beta_search_only を呼び出す。
    打ち切りは「予測」ではなく、実際に time_limit_sec を過ぎた時点で
    探索を中断する方式で行う:
      - 次の深さを開始する前に残り時間を確認し、すでに time_limit_sec を
        過ぎていれば新しい深さの探索は開始しない。
      - 探索の最中 (alpha_beta_rec / endgame_search 内、4096ノードごと) に
        ctx.deadline を超えたことを検知すると _SearchTimeout を送出し、
        その深さの探索を即座に中断する。中断された深さの結果は不完全なため
        破棄し、最後に完了した深さ (best_action / best_depth) の結果を返す。

    終盤読み切り (empty_square_count <= ENDGAME_EMPTY_LIMIT) に入った場合は
    完全読み済みのため、その深さで確定して終了する。

    Parameters
    ----------
    position       : pyrev.Position  現在の局面
    time_limit_sec : float           思考時間の上限 (秒)
    test_mode      : bool            True のとき探索ログを標準出力に表示する

    Returns
    -------
    int : 選択した着手座標。合法手がない場合は -1。
    """
    start       = time.perf_counter()
    best_action = -1
    best_depth  = 0

    deadline = start + time_limit_sec_set

    # ⑥⑦ SearchContext と置換表を反復深化ループの外側で1回だけ生成し、
    # 全深さで使い回す。
    #   ⑥ SearchContext: 深さごとに 64個 (MAX_PLY) の Position オブジェクトを
    #      再生成するコストを排除する。
    #   ⑦ 置換表: _store_tt は既存エントリより深い探索結果でしか上書きしない
    #      設計のため、浅い深さ (depth=1,2,3...) で得た結果を次の深さでも
    #      安全に再利用できる。標準的な「反復深化 + 置換表」の手法そのもの。
    ctx = SearchContext(deadline=deadline)
    tt  = {}

    for depth in range(1, 64):

        # 探索開始前に残り時間を確認し、ゼロ以下なら新しい深さを開始しない
        if time.perf_counter() - start >= time_limit_sec_set:
            if test_mode:
                print(f"[do_move] 残り時間なし → depth={depth} の探索を開始せず終了")
            break

        t0 = time.perf_counter()
        try:
            action, score = alpha_beta_search_only(
                position, depth, deadline=deadline, ctx=ctx, tt=tt
            )
        except _SearchTimeout:
            # この深さの探索中に time_limit_sec を超過 → 結果は不完全なので破棄
            # best_action は直近に完了した depth=best_depth の結果のまま使う
            if test_mode:
                print(f"[do_move] depth={depth} 探索中にタイムアウト "
                      f"→ depth={best_depth} の結果を使用")
            break
        t1 = time.perf_counter()

        elapsed_this = t1 - t0          # この深さにかかった時間
        elapsed_total = t1 - start      # 開始からの合計時間
        remaining     = time_limit_sec_set - elapsed_total

        # 合法手が存在した場合のみ結果を更新 (この深さの探索は完了している)
        if int(action) >= 0:
            best_action = action
            best_depth  = depth

        if test_mode:
            coord = (pyrev.coord_to_str(np.int8(action))
                     if int(action) >= 0 else "None")
            print(f"[do_move] depth={depth:2d}  action={coord}"
                  f"  elapsed={elapsed_this:.3f}s  remaining={remaining:.3f}s")

        # 終盤完全読み完了の判定。
        # depth >= empty_square_count - ENDGAME_EMPTY_LIMIT になった時点で
        # 探索木のすべての葉が endgame_search に到達し、完全読みが完了する。
        # ルート局面が直接 endgame 以下の場合 (depth=1 で成立) も含む。
        if position.empty_square_count - depth <= ENDGAME_EMPTY_LIMIT:
            if test_mode:
                if score is not None and score >= WIN_SCORE:
                    result_str = "自分が勝利"
                elif score is not None and score <= -WIN_SCORE:
                    result_str = "相手が勝利"
                elif score == 0:
                    result_str = "引き分け"
                else:
                    result_str = f"スコア={score}"
                print(f"[do_move] 終盤完全読み完了 → 確定 (結果: {result_str})")
            break

        # 次のループの先頭にある残り時間チェックと、探索内部の _SearchTimeout
        # だけで打ち切りを判断する (予測ベースの早期打ち切りは行わない)。

    if test_mode:
        final_coord = (pyrev.coord_to_str(np.int8(best_action))
                       if best_action >= 0 else "None")
        print(f"[do_move] 最終選択: {final_coord}  "
              f"到達深さ: {best_depth}  "
              f"合計時間: {time.perf_counter() - start:.3f}s")

    return best_action