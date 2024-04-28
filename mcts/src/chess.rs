use std::fmt::{self, Debug, Formatter};

use crate::searchtree::Node;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ChessPlayer {
    White,
    Black,
}

impl ChessPlayer {
    fn _other(&self) -> ChessPlayer {
        match self {
            ChessPlayer::Black => ChessPlayer::White,
            ChessPlayer::White => ChessPlayer::Black,
        }
    }
}

#[derive(PartialEq, Debug, Eq, Hash)]
pub enum ChessResult {
    Draw,
    Win(ChessPlayer),
}

#[derive(Clone, Eq, PartialEq, Hash, Copy)]
/// When implementing this chess module, I realized a design mistake.
/// The mistake was to conflate the player and the pieces.
/// This mistake did not surface with TTT and C4 because there, every player
/// has only one type of pieces which are "colored" in the same way the player is.
pub enum ChessPiece {
    Pawn(ChessPlayer),
    Bishop(ChessPlayer),
    Knight(ChessPlayer),
    Rook(ChessPlayer),
    Queen(ChessPlayer),
    King(ChessPlayer),
}

impl ChessPiece {
    fn color(&self) -> ChessPlayer {
        match self {
            ChessPiece::Pawn(c) => *c,
            ChessPiece::Bishop(c) => *c,
            ChessPiece::Knight(c) => *c,
            ChessPiece::Rook(c) => *c,
            ChessPiece::Queen(c) => *c,
            ChessPiece::King(c) => *c,
        }
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
/// A Connnect Four game state
/// Index for the bottom left is 0, bottom right is 6 and for the top right is 41
pub struct ChessState {
    board: [Option<ChessPiece>; 64],
    pub whose_turn: ChessPlayer,
}

impl ChessState {
    pub fn new() -> ChessState {
        let mut board = [None; 64];
        board[8..16]
            .iter_mut()
            .for_each(|x| *x = Some(ChessPiece::Pawn(ChessPlayer::White)));
        board[48..56]
            .iter_mut()
            .for_each(|x| *x = Some(ChessPiece::Pawn(ChessPlayer::Black)));

        board[0] = Some(ChessPiece::Rook(ChessPlayer::White));
        board[1] = Some(ChessPiece::Knight(ChessPlayer::White));
        board[2] = Some(ChessPiece::Bishop(ChessPlayer::White));
        board[3] = Some(ChessPiece::Queen(ChessPlayer::White));
        board[4] = Some(ChessPiece::King(ChessPlayer::White));
        board[5] = Some(ChessPiece::Bishop(ChessPlayer::White));
        board[6] = Some(ChessPiece::Knight(ChessPlayer::White));
        board[7] = Some(ChessPiece::Rook(ChessPlayer::White));

        board[56] = Some(ChessPiece::Rook(ChessPlayer::Black));
        board[57] = Some(ChessPiece::Knight(ChessPlayer::Black));
        board[58] = Some(ChessPiece::Bishop(ChessPlayer::Black));
        board[59] = Some(ChessPiece::Queen(ChessPlayer::Black));
        board[60] = Some(ChessPiece::King(ChessPlayer::Black));
        board[61] = Some(ChessPiece::Bishop(ChessPlayer::Black));
        board[62] = Some(ChessPiece::Knight(ChessPlayer::Black));
        board[63] = Some(ChessPiece::Rook(ChessPlayer::Black));

        ChessState {
            board,
            whose_turn: ChessPlayer::White,
        }
    }

    pub fn empty() -> ChessState {
        ChessState {
            board: [None; 64],
            whose_turn: ChessPlayer::White,
        }
    }

    fn with_turn(self, player: ChessPlayer) -> ChessState {
        ChessState {
            board: self.board,
            whose_turn: player,
        }
    }

    fn with_index(self, idx: usize, piece: ChessPiece) -> Result<ChessState, ()> {
        if idx >= 64 || self.board[idx].is_some() {
            return Err(());
        }
        let mut pos = self.board;
        pos[idx] = Some(piece);
        Ok(ChessState {
            board: pos,
            whose_turn: self.whose_turn,
        })
    }
}

impl Debug for ChessState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for row in (0..8).rev() {
            for col in 0..8 {
                match self.board[row * 8 + col] {
                    Some(ChessPiece::Pawn(ChessPlayer::White)) => write!(f, "♟")?,
                    Some(ChessPiece::Pawn(ChessPlayer::Black)) => write!(f, "♙")?,
                    Some(ChessPiece::Bishop(ChessPlayer::White)) => write!(f, "♝")?,
                    Some(ChessPiece::Bishop(ChessPlayer::Black)) => write!(f, "♗")?,
                    Some(ChessPiece::Knight(ChessPlayer::White)) => write!(f, "♞")?,
                    Some(ChessPiece::Knight(ChessPlayer::Black)) => write!(f, "♘")?,
                    Some(ChessPiece::Rook(ChessPlayer::White)) => write!(f, "♜")?,
                    Some(ChessPiece::Rook(ChessPlayer::Black)) => write!(f, "♖")?,
                    Some(ChessPiece::Queen(ChessPlayer::White)) => write!(f, "♛")?,
                    Some(ChessPiece::Queen(ChessPlayer::Black)) => write!(f, "♕")?,
                    Some(ChessPiece::King(ChessPlayer::White)) => write!(f, "♚")?,
                    Some(ChessPiece::King(ChessPlayer::Black)) => write!(f, "♔")?,
                    None => {
                        if row % 2 == col % 2 {
                            write!(f, "□")?
                        } else {
                            write!(f, "■")?
                        }
                    }
                }
                write!(f, " ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Node for ChessState {
    type Winner = ChessResult;

    fn possible_next_states(&self) -> Vec<ChessState> {
        let mut states = Vec::new();
        for (i, &piece) in self.board.iter().enumerate() {
            if let Some(piece) = piece {
                match piece {
                    ChessPiece::Pawn(c) => {
                        if c == self.whose_turn {
                            match c {
                                ChessPlayer::White => {
                                    if i < 56 {
                                        // Move one square forward
                                        if self.board[i + 8].is_none() {
                                            let mut new_board = self.board.clone();
                                            new_board[i + 8] =
                                                Some(ChessPiece::Pawn(self.whose_turn));
                                            new_board[i] = None;
                                            states.push(ChessState {
                                                board: new_board,
                                                whose_turn: self.whose_turn._other(),
                                            });
                                        }
                                        // Atack diagonally
                                        if i % 8 != 0 {
                                            if let Some(piece) = self.board[i + 9] {
                                                if i < 55
                                                    && piece.color() == self.whose_turn._other()
                                                {
                                                    let mut new_board = self.board.clone();
                                                    new_board[i + 9] =
                                                        Some(ChessPiece::Pawn(self.whose_turn));
                                                    new_board[i] = None;
                                                    states.push(ChessState {
                                                        board: new_board,
                                                        whose_turn: self.whose_turn._other(),
                                                    });
                                                }
                                            }
                                        }
                                        if i % 8 != 7 {
                                            if let Some(piece) = self.board[i + 7] {
                                                if piece.color() == self.whose_turn._other() {
                                                    let mut new_board = self.board.clone();
                                                    new_board[i + 7] =
                                                        Some(ChessPiece::Pawn(self.whose_turn));
                                                    new_board[i] = None;
                                                    states.push(ChessState {
                                                        board: new_board,
                                                        whose_turn: self.whose_turn._other(),
                                                    });
                                                }
                                            }
                                        }
                                    }
                                    // Move two squares forward
                                    if i >= 8
                                        && i < 16
                                        && self.board[i + 16].is_none()
                                        && self.board[i + 8].is_none()
                                    {
                                        let mut new_board = self.board.clone();
                                        new_board[i + 16] = Some(ChessPiece::Pawn(self.whose_turn));
                                        new_board[i] = None;
                                        states.push(ChessState {
                                            board: new_board,
                                            whose_turn: self.whose_turn._other(),
                                        });
                                    }
                                }
                                ChessPlayer::Black => {
                                    // Move one square forward
                                    if i > 7 {
                                        if self.board[i - 8].is_none() {
                                            let mut new_board = self.board.clone();
                                            new_board[i - 8] =
                                                Some(ChessPiece::Pawn(self.whose_turn));
                                            new_board[i] = None;
                                            states.push(ChessState {
                                                board: new_board,
                                                whose_turn: self.whose_turn._other(),
                                            });
                                        }
                                        if i % 8 != 0 {
                                            // Atack diagonally
                                            if let Some(piece) = self.board[i - 9] {
                                                if piece.color() == self.whose_turn._other() {
                                                    let mut new_board = self.board.clone();
                                                    new_board[i - 9] =
                                                        Some(ChessPiece::Pawn(self.whose_turn));
                                                    new_board[i] = None;
                                                    states.push(ChessState {
                                                        board: new_board,
                                                        whose_turn: self.whose_turn._other(),
                                                    });
                                                }
                                            }
                                        }
                                        if i % 8 != 7 {
                                            if let Some(piece) = self.board[i - 7] {
                                                if piece.color() == self.whose_turn._other() {
                                                    let mut new_board = self.board.clone();
                                                    new_board[i - 7] =
                                                        Some(ChessPiece::Pawn(self.whose_turn));
                                                    new_board[i] = None;
                                                    states.push(ChessState {
                                                        board: new_board,
                                                        whose_turn: self.whose_turn._other(),
                                                    });
                                                }
                                            }
                                        }
                                    }
                                    // Move two squares forward
                                    if i >= 48
                                        && i < 56
                                        && self.board[i - 16].is_none()
                                        && self.board[i - 8].is_none()
                                    {
                                        let mut new_board = self.board.clone();
                                        new_board[i - 16] = Some(ChessPiece::Pawn(self.whose_turn));
                                        new_board[i] = None;
                                        states.push(ChessState {
                                            board: new_board,
                                            whose_turn: self.whose_turn._other(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                    ChessPiece::Bishop(_c) => {}
                    ChessPiece::Knight(_c) => {}
                    ChessPiece::Rook(_c) => {}
                    ChessPiece::Queen(_c) => {}
                    ChessPiece::King(_c) => {}
                }
            }
        }
        states
    }

    fn is_terminal(&self) -> Option<ChessResult> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chess::{ChessPiece, ChessPlayer, ChessState},
        searchtree::Node,
    };

    #[test]
    fn can_instaniate_empty_board() {
        dbg!(ChessState::empty());
    }

    #[test]
    fn can_instaniate_default_board() {
        dbg!(ChessState::new());
    }

    #[test]
    fn can_find_white_pawn_moves() {
        let state = ChessState::empty()
            .with_index(8, ChessPiece::Pawn(ChessPlayer::White))
            .unwrap()
            .with_index(9, ChessPiece::Pawn(ChessPlayer::White))
            .unwrap()
            .with_index(15, ChessPiece::Pawn(ChessPlayer::Black))
            .unwrap()
            .with_index(17, ChessPiece::Pawn(ChessPlayer::Black))
            .unwrap()
            .with_index(18, ChessPiece::Pawn(ChessPlayer::Black))
            .unwrap()
            .with_index(48, ChessPiece::Pawn(ChessPlayer::Black))
            .unwrap();
        let next_states = state.possible_next_states();
        assert_eq!(next_states.len(), 4);
    }

    #[test]
    fn can_find_black_pawn_moves() {
        let state = ChessState::empty()
            .with_turn(ChessPlayer::Black)
            .with_index(10, ChessPiece::Pawn(ChessPlayer::White))
            .unwrap()
            .with_index(7, ChessPiece::Pawn(ChessPlayer::White))
            .unwrap()
            .with_index(8, ChessPiece::Pawn(ChessPlayer::White))
            .unwrap()
            .with_index(16, ChessPiece::Pawn(ChessPlayer::Black))
            .unwrap()
            .with_index(17, ChessPiece::Pawn(ChessPlayer::Black))
            .unwrap()
            .with_index(48, ChessPiece::Pawn(ChessPlayer::Black))
            .unwrap();
        let next_states = state.possible_next_states();
        assert_eq!(next_states.len(), 5);
    }

    #[test]
    fn lots_of_next_moves() {
        let state = ChessState::new();
        let next_states = state.possible_next_states();
        dbg!(&next_states);
    }
}
