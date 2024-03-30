from connect4 import Connect4
from copy import deepcopy
from exceptions import AgentException
from typing import Tuple

class AlphaBetaAgent:
    def __init__(self, token):
        self.my_token = token

    def decide(self, connect4: 'Connect4'):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my turn')
        _, move = self.alphabeta(connect4, 1, 5, -float('inf'), float('inf'))
        return move
    
    def heuristic(self, connect4: 'Connect4')->int:
        weights = [0,5,15,50]
        total = 0
        score = 0
        for four in connect4.iter_fours():
            total += 1
            ours = four.count(self.my_token)
            empty = four.count('_')
            theirs = 4 - ours - empty

            if theirs != 0:
                continue

            score += weights[ours]
        return score / (total * weights[-1])
    
    def alphabeta(self, connect4: 'Connect4', x: int, d: int, alpha: int, beta: int) -> Tuple[int, int]:
        if connect4.game_over and connect4.wins == self.my_token:
            return 1, -1
        
        if connect4.game_over and connect4.wins == None:
            return 0, -1
        
        if connect4.game_over and connect4.wins != self.my_token:
            return -1, -1
        
        if d == 0:
            return self.heuristic(connect4), -1

        best_score = -1 if x == 1 else 1
        best_move = 0
        for possible_move in connect4.possible_drops():
            simulation = deepcopy(connect4)
            simulation.drop_token(possible_move)
            if x == 1:
                score, _ = self.alphabeta(simulation, 0, d-1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = possible_move
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            else:
                score, _ = self.alphabeta(simulation, 1, d-1, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = possible_move
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return [best_score, best_move]