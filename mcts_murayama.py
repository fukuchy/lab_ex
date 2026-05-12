import math
import random
from pyrev import Position


INF = 10**9

def advance(pos: Position, coord):
    flip = pos.calc_flip_discs(coord)
    pos.do_move(coord, flip)

# MCTS
C = 1.0                 # UCB1の計算に使う重み(定数)
EXPAND_THRESHOLD = 10   # ノードを展開する閾値

class Node:
    def __init__(self, pos: Position, root_color):
        self.pos = pos.copy()   # 状態保持
        self.root_color = root_color
        self.w = 0.0            # 勝ちスコア合計
        self.n = 0              # 試行回数
        self.child_nodes = []   # 子ノード

    def evaluate(self):
        # 終局
        if self.pos.is_gameover():
            value = self.get_value()
            self.w += value
            self.n += 1
            return value

        # 未展開
        if not self.child_nodes:
            value = playout(self.pos.copy(), self.root_color)

            self.w += value
            self.n += 1

            if self.n == EXPAND_THRESHOLD:
                self.expand()

            return value

        # 展開済み
        else:
            child = self.nextChildNode()
            value = child.evaluate()

            self.w += value
            self.n += 1
            return value
    
    def get_value(self):
        # ルートプレイヤー視点で評価
        if self.pos.side_to_move == self.root_color:
            my = self.pos.player_disc_count
            opp = self.pos.opponent_disc_count
        else:
            my = self.pos.opponent_disc_count
            opp = self.pos.player_disc_count

        if my > opp:
            return 1.0
        elif my < opp:
            return 0.0
        else:
            return 0.5
        
    def expand(self):
        moves = list(self.pos.get_legal_moves())
        self.child_nodes = []

        for move in moves:
            next_pos = self.pos.copy()
            advance(next_pos, move)
            self.child_nodes.append(Node(next_pos, self.root_color))

    def nextChildNode(self):
        # 未訪問ノード優先
        for child in self.child_nodes:
            if child.n == 0:
                return child

        # 合計試行回数
        t = sum(child.n for child in self.child_nodes)

        best_value = -INF
        best_child = None

        for child in self.child_nodes:
            ucb1 = (
                (child.w / child.n) +
                C * math.sqrt(2.0 * math.log(t) / child.n)
            )

            if ucb1 > best_value:
                best_value = ucb1
                best_child = child

        return best_child

def mctsAction(pos: Position, playout_number: int):
    root = Node(pos, pos.side_to_move)
    root.expand()

    for _ in range(playout_number):
        root.evaluate()

    moves = list(pos.get_legal_moves())

    best_move = None
    best_n = -1

    for i in range(len(moves)):
        n = root.child_nodes[i].n
        if n > best_n:
            best_n = n
            best_move = moves[i]

    return best_move

def mctsAI(pos: Position):
    return mctsAction(pos, 500)

def playout(pos: Position, root_color):
    # 終局判定
    if pos.is_gameover():
        # BLACK/WHITEの石数を正しく取得
        if pos.side_to_move == root_color:
            my = pos.player_disc_count
            opp = pos.opponent_disc_count
        else:
            my = pos.opponent_disc_count
            opp = pos.player_disc_count

        if my > opp:
            return 1.0
        elif my < opp:
            return 0.0
        else:
            return 0.5

    moves = list(pos.get_legal_moves())

    if len(moves) == 0:
        if pos.can_pass():
            next_pos = pos.copy()
            next_pos.do_pass()
            return playout(next_pos, root_color)
        else:
            return 0.5

    move = random.choice(moves)
    next_pos = pos.copy()
    advance(next_pos, move)

    return playout(next_pos, root_color)