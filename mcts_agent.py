import math
import random
import pyrev


class Node:
    def __init__(self, parent, move, player_to_move, root_player, untried_moves):
        self.parent = parent
        self.move = move
        self.player_to_move = player_to_move
        self.root_player = root_player

        self.children = []
        self.untried_moves = untried_moves[:]

        self.visits = 0
        self.wins = 0.0

    def ucb1(self, c=math.sqrt(2)):
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self):
        return max(self.children, key=lambda child: child.ucb1())

    def add_child(self, move, pos):
        child = Node(
            parent=self,
            move=move,
            player_to_move=pos.side_to_move,
            root_player=self.root_player,
            untried_moves=list(pos.get_legal_moves())
        )
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result


def rollout(pos, root_player):
    while not pos.is_gameover():
        moves = list(pos.get_legal_moves())

        if not moves:
            if pos.can_pass():
                pos.do_pass()
                continue
            break

        move = random.choice(moves)
        pos.do_move_at(move)

    score = int(pos.get_score_from(root_player))
    if score > 0:
        return 1.0
    if score == 0:
        return 0.5
    return 0.0


def apply_move(pos, move):
    flip = pos.calc_flip_discs(move)
    pos.do_move(move, flip)
    return flip


def undo_move(pos, move, flip):
    pos.undo(move, flip)


def select_move(pos, num_simulations=500):
    root_player = pos.side_to_move
    root = Node(
        parent=None,
        move=None,
        player_to_move=pos.side_to_move,
        root_player=root_player,
        untried_moves=list(pos.get_legal_moves())
    )

    if not root.untried_moves:
        return None

    for _ in range(num_simulations):
        node = root
        move_stack = []
        pass_count = 0

        # 1. Selection
        while not node.untried_moves and node.children:
            node = node.select_child()
            flip = apply_move(pos, node.move)
            move_stack.append((node.move, flip))

        # 2. Expansion
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            flip = apply_move(pos, move)
            move_stack.append((move, flip))
            node = node.add_child(move, pos)

        # 3. Simulation
        sim_move_stack = []
        while not pos.is_gameover():
            moves = list(pos.get_legal_moves())

            if not moves:
                if pos.can_pass():
                    pos.do_pass()
                    sim_move_stack.append(("PASS", None))
                    pass_count += 1
                    if pass_count >= 2:
                        break
                    continue
                break

            pass_count = 0
            move = random.choice(moves)
            flip = apply_move(pos, move)
            sim_move_stack.append((move, flip))

        score = int(pos.get_score_from(root_player))
        if score > 0:
            result = 1.0
        elif score == 0:
            result = 0.5
        else:
            result = 0.0

        # 4. Backpropagation
        while node is not None:
            node.update(result)
            node = node.parent

        # Undo simulation
        while sim_move_stack:
            move, flip = sim_move_stack.pop()
            if move == "PASS":
                pos.do_pass()
            else:
                undo_move(pos, move, flip)

        # Undo tree traversal
        while move_stack:
            move, flip = move_stack.pop()
            undo_move(pos, move, flip)

    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move