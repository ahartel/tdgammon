/// When implementing this chess module, I realized a design mistake.
/// The mistake was to conflate the player and the pieces.
/// This mistake did not surface with TTT and C4 because there, every player
/// has only one type of pieces which are "colored" in the same way the player is.
use std::fmt::{self, Debug, Formatter};

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
pub enum ChessPiece {
    Pawn(ChessPlayer),
    Bishop(ChessPlayer),
    Knight(ChessPlayer),
    Rook(ChessPlayer),
    Queen(ChessPlayer),
    King(ChessPlayer),
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

#[cfg(test)]
mod tests {
    use crate::chess::ChessState;

    #[test]
    fn can_instaniate_empty_board() {
        dbg!(ChessState::empty());
    }

    #[test]
    fn can_instaniate_default_board() {
        dbg!(ChessState::new());
    }
}
