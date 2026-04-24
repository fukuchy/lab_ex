import pyrev
import math
import random
from pyrev import Position

C=1.4
EXPAND_THRESHOLD=25


class Node():
    def __init__(self,position):
        self.position_=position.copy()
        self.w_=0.0
        self.child_nodes_=[]
        self.n_=0
        self.player=position.side_to_move

    def playout(self,position):
        while not position.is_gameover():
            action=list(position.get_legal_moves())
            if len(action)==0:
                position.do_pass()
            else:
                i=random.choice(action)
                flip=position.calc_flip_discs(i)
                position.do_move(i,flip)
        if position.is_gameover():
            my = position.get_disc_count_of(self.player)
            op = position.disc_count-position.get_disc_count_of(self.player)
            if my>op:
                return 1
            elif my<op:
                return 0
            else:
                return 0.5
                
                
    
    def evaluate(self):
        if self.position_.is_gameover():
            white = self.position_.player_disc_count
            black = self.position_.opponent_disc_count
            if white>black:
                value=1
            elif black>white:
                value=0
            else:
                value=0.5
            
            self.w_+=value
            self.n_+=1
            return value
        if len(self.child_nodes_)==0:
            position_copy=self.position_.copy()
            value=self.playout(position_copy)
            self.w_+=value
            self.n_+=1

            if self.n_>=EXPAND_THRESHOLD:
                self.expand()
            return value
        else:
            value=1-self.nextChildNode().evaluate()
            self.w_+=value
            self.n_+=1
            return value

    def expand(self):
        legal_actions=self.position_.get_legal_moves()
        for i in legal_actions:
            next_position=self.position_.copy()
            flip=next_position.calc_flip_discs(i)
            next_position.do_move(i,flip)
            self.child_nodes_.append(Node(next_position))
            
    def nextChildNode(self):
        for i in self.child_nodes_:
            if i.n_==0:
                return i
        t=0
        for j in self.child_nodes_:
            t+=j.n_
        best_value=-math.inf
        best_action_index=-1
        for k in range(len(self.child_nodes_)):
            child_node=self.child_nodes_[k]
            ucb1=1-child_node.w_/child_node.n_+C*math.sqrt(2*math.log(t)/child_node.n_)
            if ucb1>best_value:
                best_action_index=k
                best_value=ucb1
        return self.child_nodes_[best_action_index]
            
    

def mctsAction(state,playout_number):
    root_node=Node(state)
    root_node.expand()
    for i in range(playout_number):
        root_node.evaluate()
        
    legal_actions=list(state.get_legal_moves())
    best_action_searched_number=-1
    best_action_index=-1

    assert len(legal_actions)==len(root_node.child_nodes_)    
    for i in range(len(legal_actions)):
        n=root_node.child_nodes_[i].n_
        if(n>best_action_searched_number):
            best_action_index=i
            best_action_searched_number=n
    return legal_actions[best_action_index]
