import math
import random
import pyrev

EXPAND_THRESHOLD = 10

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.value = 0.0
        self.children_nodes = []
        self.visits = 0

    def evaluate(self):
        my_color = self.pos.side_to_move
        opp_color = pyrev.to_opponent_color(my_color)

        if self.pos.is_gameover():
            value = 0.5

            if self.pos.get_disc_count_of(my_color) > self.pos.get_disc_count_of(opp_color):
                value = 1.0
            elif self.pos.get_disc_count_of(my_color) < self.pos.get_disc_count_of(opp_color):
                value = 0.0
            
            self.value += value
            self.visits += 1
            return value
        
        moves = list(self.pos.get_legal_moves())
        if not moves and self.pos.can_pass():
            next_pos = self.pos.copy()
            next_pos.do_pass()
            value = 1.0 - Node(next_pos).evaluate()
            self.value += value
            self.visits += 1
            return value
        
        if (len(self.children_nodes) == 0):
            current_pos = self.pos.copy()
            value = playout(current_pos)
            self.value += value
            self.visits += 1

            if self.visits == EXPAND_THRESHOLD:
                self.expand()

            return value
        
        else:
            value = 1.0 - self.next_child().evaluate()
            self.value += value
            self.visits += 1
            return value
    
    def expand(self):
        actions = list(self.pos.get_legal_moves())
        self.children_nodes = []
        for action in actions:
            next_pos = self.pos.copy()
            next_pos.do_move_at(action)
            self.children_nodes.append(Node(next_pos))
    
    def next_child(self):
        for child in self.children_nodes:
            if child.visits == 0:
                return child
            
        t = 0.0
        for child in self.children_nodes:
            t += child.visits
        
        best_score = -1 * math.inf
        best_action_index = -1
        for i in range(len(self.children_nodes)):
            child = self.children_nodes[i]
            ucb1_value = 1 - child.value / child.visits + math.sqrt(2 * math.log(t) / child.visits)
            if ucb1_value > best_score:
                best_score = ucb1_value
                best_action_index = i
        
        return self.children_nodes[best_action_index]

def playout(pos):

    if pos.is_gameover():
        my_color = pos.side_to_move
        opp_color = pyrev.to_opponent_color(my_color)
        if pos.get_disc_count_of(my_color) > pos.get_disc_count_of(opp_color):
            return 1.0
        elif pos.get_disc_count_of(my_color) < pos.get_disc_count_of(opp_color):
            return 0.0
        else:
            return 0.5
    
    moves = list(pos.get_legal_moves())

    if not moves:
        if pos.can_pass():
            pos.do_pass()
            return 1.0 - playout(pos)
        return 0.5
    
    pos.do_move_at(random.choice(moves))
    return 1.0 - playout(pos)

def mcts(pos, playout_num):
    actions = list(pos.get_legal_moves())
    if not actions:
        return None
    
    root = Node(pos.copy())

    root.expand()

    for i in range(playout_num):
        root.evaluate()
        
    best_action_searched_number = -1
    best_action_index = -1

    assert len(actions) == len(root.children_nodes)

    for i in range(len(actions)):
        n = root.children_nodes[i].visits
        if n > best_action_searched_number:
            best_action_searched_number = n
            best_action_index = i
    
    return actions[best_action_index]

