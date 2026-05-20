import math
import random
from pyrev import Position

#from game_utils import advance

INF = 10**9

def advance(pos: Position, coord):
    flip = pos.calc_flip_discs(coord)
    pos.do_move(coord, flip)

def playout(pos: Position):
    # 終局
    if pos.is_gameover():
        if pos.player_disc_count > pos.opponent_disc_count:
            return 1.0
        elif pos.player_disc_count < pos.opponent_disc_count:
            return 0.0
        else:
            return 0.5

    moves = list(pos.get_legal_moves())

    if len(moves) == 0:
        next_pos = pos.copy()
        next_pos.do_pass()

        return 1.0 - playout(next_pos)

    move = random.choice(moves)

    next_pos = pos.copy()
    advance(next_pos, move)

    return 1.0 - playout(next_pos)


# MCTS
C = 1.4                 # UCB1の計算に使う重み(定数)
EXPAND_THRESHOLD = 10   # ノードを展開する閾値

class Node:
    def __init__(self, pos: Position):
        self.pos = pos.copy()
        
        self.w = 0.0            # 勝ちスコア合計
        self.n = 0              # 試行回数
        self.child_nodes = []   # 子ノード

    def evaluate(self):
        # 終局
        if self.pos.is_gameover():
            if self.pos.player_disc_count > self.pos.opponent_disc_count:
                value = 1.0
            elif self.pos.player_disc_count < self.pos.opponent_disc_count:
                value = 0.0
            else:
                value = 0.5

            self.w += value
            self.n += 1

            return value

        # 未展開
        if not self.child_nodes:
            value = playout(self.pos.copy())

            self.w += value
            self.n += 1

            if self.n == EXPAND_THRESHOLD:
                self.expand()

            return value

        # 展開済み
        else:
            child = self.nextChildNode()
            value = 1.0 - child.evaluate()

            self.w += value
            self.n += 1

            return value
        
    def expand(self):
        moves = list(self.pos.get_legal_moves())
        self.child_nodes = []

        if len(moves) == 0:
            next_pos = self.pos.copy()
            next_pos.do_pass()

            self.child_nodes.append(Node(next_pos))
            return
    
        for move in moves:
            next_pos = self.pos.copy()
            advance(next_pos, move)
            self.child_nodes.append(Node(next_pos))

    def nextChildNode(self):
        # まずは全手を最低1回は調べたいため、未訪問ノード優先
        for child_node in self.child_nodes:
            if child_node.n == 0:
                return child_node

        # 合計試行回数
        t = sum(child_node.n for child_node in self.child_nodes)

        best_value = -INF
        best_child = None

        for child_node in self.child_nodes:
            ucb1_value = (
                1.0 - (child_node.w / child_node.n)
                + C * math.sqrt(2.0 * math.log(t) / child_node.n)
            )

            if ucb1_value > best_value:
                best_value = ucb1_value
                best_child = child_node

        return best_child

def mctsAction(pos: Position, playout_number: int):
    moves = list(pos.get_legal_moves())

    if len(moves) == 0:
        return None
    
    root_node = Node(pos)
    root_node.expand()

    for _ in range(playout_number):
        root_node.evaluate()

    best_move_searched_number = -1
    best_move = None

    for i in range(len(moves)):
        n = root_node.child_nodes[i].n #ノードiの訪問回数

        if n > best_move_searched_number:
            best_move_searched_number = n #最も多く訪問しているノードの回数に更新
            best_move = moves[i] #最も多く訪問しているノード

    return best_move

def mctsAI(pos: Position):
    return mctsAction(pos, 500)