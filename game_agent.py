"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

import numpy as np
from isolation import Board

class SearchTimeout(Exception):
    """ Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #the huristic will be just to get the number of moves a player can do.
    #print('Active player has {} moves'.format(len(game.get_legal_moves(game.active_player))))

    return float(len(game.get_legal_moves(game.active_player)))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return float(len(game.get_legal_moves(game.active_player)) - len(game.get_legal_moves(game.inactive_player)))

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    board_size = game.width * game.height

    midgame = round(board_size * 0.75)
    endgame = round(board_size * 0.45)

    if len(game.get_blank_spaces()) >= midgame:
        #print('IN START GAME')
        return float(len(game.get_legal_moves(game.active_player)) - 4*len(game.get_legal_moves(game.inactive_player)))


    elif (len(game.get_blank_spaces()) < midgame) & (len(game.get_blank_spaces()) >= endgame):
        #print('\t\tIN MID GAME')
        return float(len(game.get_legal_moves(game.active_player)) - len(game.get_legal_moves(game.inactive_player)))

    else:
        #print('\t\t\tIN END GAME')
        return float(4*len(game.get_legal_moves(game.active_player)) - len(game.get_legal_moves(game.inactive_player)))

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=1, score_fn=custom_score_3, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game,self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move



    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        best_move = []

        best_score = -9999
        for move in game.get_legal_moves(game.active_player):

            clone = game.forecast_move(move)

            move_score = self.MIN(clone,depth-1)
            #print('doing move {} that got a score of {}'.format(move,move_score))

            if move_score > best_score:
                best_move = move
                best_score = move_score

        return best_move

    def MIN(self,game,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if not game.get_legal_moves():
            return self.score(game, self)

        if depth <= 0:
            return self.score(game, self)

        best_score = 9999

        for move in game.get_legal_moves(game.active_player):
            clone = game.forecast_move(move)
            move_score = self.MAX(clone,depth - 1)


            if move_score < best_score:
                best_score = move_score

        return best_score

    def MAX(self,game,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if not game.get_legal_moves():
            return self.score(game, self)

        if depth <= 0:
            return self.score(game, game.active_player)

        best_score = -9999

        for move in game.get_legal_moves(game.active_player):
            clone = game.forecast_move(move)

            move_score = self.MIN(clone,depth-1)


            if move_score > best_score:
                best_score = move_score

        return best_score

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)
        self.search_depth = 0
        try:
            
            while self.time_left() > self.TIMER_THRESHOLD:
                next_move = self.alphabeta(game, self.search_depth)

                if not next_move:
                    return best_move
                else:
                    #print('DEPTH {} \t TIME {} \t MOVE {}'.format(self.search_depth, self.time_left(),next_move))
                    best_move = next_move
                self.search_depth += 1



            #return self.alphabeta(game, self.search_depth)
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move



    def alphabeta(self, game,depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        best_move = (-1,-1)

        val = float("-inf")
        for move in game.get_legal_moves(game.active_player):

            clone = game.forecast_move(move)
            move_score = self.alpha_min(clone, depth-1, alpha, beta)

            if move_score > val:
                best_move = move
                val = move_score
            alpha = max(alpha,move_score)

        return best_move

    def alpha_min(self,game,depth,alpha,beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if not game.get_legal_moves():
            return self.score(game, self)

        if depth <= 0:

            return self.score(game,self)
        '''
        how I implimented
        
        val = float('inf')
        for move in game.get_legal_moves(game.active_player):

            clone = game.forecast_move(move)
            val_p = self.alpha_max(clone,depth-1, alpha,beta)


            if val_p < val:
                val = val_p

            if val_p <= alpha:
                return val

            if val_p < beta:
                beta = val_p
        
        return val
        '''
        # Instructor suggestion

        val = float('inf')

        for move in game.get_legal_moves(game.active_player):
            val = min(val, self.alpha_max(game.forecast_move(move),depth-1,alpha,beta))
            if val <= alpha:
                return val
            beta = min(val,beta)
        return val

    def alpha_max(self,game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if not game.get_legal_moves():
            return self.score(game, self)

        if depth <= 0:
            return self.score(game, self)

        '''
        my implementation
        val = float("-inf")
        for move in game.get_legal_moves(game.active_player):

            clone = game.forecast_move(move)
            val_p = self.alpha_min(clone,depth-1, alpha, beta)

            if val_p > val:

                val = val_p

            if val_p >= beta:

                return val

            if val_p > alpha:

                alpha = val_p

        return val
        '''

        # Instructor  suggestion
        val = float('-inf')
        for move in game.get_legal_moves(game.active_player):
            val = max(val,self.alpha_min(game.forecast_move(move),depth-1,alpha,beta))
            if val >= beta:
                return val
            alpha = max(val,alpha)
        return val


'''
# looking at the game in the console

play_game = True

if play_game:
    from sample_players import GreedyPlayer, RandomPlayer

    player2 = MinimaxPlayer(search_depth=5,score_fn=custom_score_2)
    player3 = AlphaBetaPlayer(score_fn=custom_score)
    player1 = GreedyPlayer()

    game = Board(player1, player3)
    #print(game.to_string())

    game.play(2000)


    if game.utility(player1) == float('inf'):
        print('player 1 has won')
    elif game.utility(player2) == float('inf'):
        print('player 2 has won')
    else:
        print('Neither won')
'''