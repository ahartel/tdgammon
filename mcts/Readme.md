# Monte Carlo Tree Search

In this crate, I implemented Monte Carlo Tree Search.

Starting point was a uniformly random implementation for Tic Tac Toe.
This implementation already shows that a uniformly random MCTS with 30 playouts is significantly
better than a player that just uniformly randomly selects the next move.

Second game I implemented was Connect 4 in whic a uniformly random MCTS wins 80% of the games
against a purely random player with 75 playouts.