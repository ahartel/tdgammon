use std::fmt::{self, Debug, Formatter};

use super::searchtree::Node;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum TTTPlayer {
    X,
    O,
}

impl TTTPlayer {
    fn other(&self) -> TTTPlayer {
        match self {
            TTTPlayer::X => TTTPlayer::O,
            TTTPlayer::O => TTTPlayer::X,
        }
    }
}

#[derive(PartialEq, Debug, Eq, Hash)]
pub enum TTTResult {
    Draw,
    Win(TTTPlayer),
}

#[derive(Clone, Eq, Hash, PartialEq)]
/// A Tic Tac Toe game state
pub struct TTTPos {
    board: [Option<TTTPlayer>; 9],
    pub whose_turn: TTTPlayer,
}

impl TTTPos {
    pub fn new() -> TTTPos {
        TTTPos {
            board: [None; 9],
            whose_turn: TTTPlayer::X,
        }
    }

    fn with_index(&self, idx: usize) -> Result<TTTPos, ()> {
        if idx >= 9 {
            return Err(());
        }
        let mut pos = self.board.clone();
        pos[idx] = Some(self.whose_turn);
        Ok(TTTPos {
            board: pos,
            whose_turn: self.whose_turn.other(),
        })
    }

    pub fn is_terminal(&self) -> Option<TTTResult> {
        if self
            .board
            .iter()
            .take(3)
            .all(|&player| player.is_some() && player == self.board[0])
        {
            return Some(TTTResult::Win(self.board[0].unwrap()));
        } else if self
            .board
            .iter()
            .skip(3)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[3])
        {
            return Some(TTTResult::Win(self.board[3].unwrap()));
        } else if self
            .board
            .iter()
            .skip(6)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[6])
        {
            return Some(TTTResult::Win(self.board[6].unwrap()));
        } else if self
            .board
            .iter()
            .step_by(3)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[0])
        {
            return Some(TTTResult::Win(self.board[0].unwrap()));
        } else if self
            .board
            .iter()
            .skip(1)
            .step_by(3)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[1])
        {
            return Some(TTTResult::Win(self.board[1].unwrap()));
        } else if self
            .board
            .iter()
            .skip(2)
            .step_by(3)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[2])
        {
            return Some(TTTResult::Win(self.board[2].unwrap()));
        } else if self
            .board
            .iter()
            .step_by(4)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[0])
        {
            return Some(TTTResult::Win(self.board[0].unwrap()));
        } else if self
            .board
            .iter()
            .skip(2)
            .step_by(2)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[2])
        {
            return Some(TTTResult::Win(self.board[2].unwrap()));
        }
        if self.board.iter().all(|&player| player.is_some()) {
            return Some(TTTResult::Draw);
        }
        None
    }
}

impl Debug for TTTPos {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, &player) in self.board.iter().enumerate() {
            if i % 3 == 0 {
                writeln!(f)?;
            }
            match player {
                Some(TTTPlayer::X) => write!(f, "X")?,
                Some(TTTPlayer::O) => write!(f, "O")?,
                None => write!(f, ".")?,
            }
        }
        Ok(())
    }
}

impl Node for TTTPos {
    fn possible_next_states(&self) -> Vec<TTTPos> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, &player)| {
                if player.is_none() {
                    Some(self.with_index(i).unwrap())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl TTTPos {
        pub fn from_slice(v: &[Option<TTTPlayer>], whose_turn: TTTPlayer) -> TTTPos {
            let mut pos = [None; 9];
            for (i, player) in v.iter().enumerate() {
                pos[i] = *player;
            }
            TTTPos {
                board: pos,
                whose_turn,
            }
        }
    }

    #[test]
    fn all_moves_are_possible() {
        let pos = TTTPos::new();
        let states = pos.possible_next_states();
        assert_eq!(states.len(), 9);
        for state in states {
            assert_eq!(state.board.iter().filter(|&&p| p.is_some()).count(), 1);
        }
    }

    #[test]
    fn one_possible_move() {
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                None,
                Some(TTTPlayer::O),
            ],
            TTTPlayer::X,
        );
        let states = pos.possible_next_states();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].board.iter().filter(|&&p| p.is_some()).count(), 9)
    }

    #[test]
    fn first_row_complete_is_terminal() {
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::X),
                Some(TTTPlayer::X),
                Some(TTTPlayer::X),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            TTTPlayer::O,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::X)));
    }

    #[test]
    fn first_col_complete_is_terminal() {
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::X),
                None,
                None,
                Some(TTTPlayer::X),
                None,
                None,
                Some(TTTPlayer::X),
                None,
                None,
            ],
            TTTPlayer::O,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::X)));
    }

    #[test]
    fn diagonal_complete_is_terminal() {
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::O),
                None,
                None,
                None,
                Some(TTTPlayer::O),
                None,
                None,
                None,
                Some(TTTPlayer::O),
            ],
            TTTPlayer::X,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::O)));
    }

    #[test]
    fn anti_diagonal_complete_is_terminal() {
        let pos = TTTPos::from_slice(
            &[
                None,
                None,
                Some(TTTPlayer::O),
                None,
                Some(TTTPlayer::O),
                None,
                Some(TTTPlayer::O),
                None,
                None,
            ],
            TTTPlayer::X,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::O)));
    }

    #[test]
    fn player_o_wins() {
        // OX.
        // XOX
        // .XO
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                None,
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                None,
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
            ],
            TTTPlayer::X,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::O)));
    }
}
