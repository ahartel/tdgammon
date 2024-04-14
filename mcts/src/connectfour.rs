use std::fmt::{self, Debug, Formatter};

use itertools::Itertools;

use crate::searchtree::Node;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum C4Player {
    Yellow,
    Red,
}

impl C4Player {
    fn other(&self) -> C4Player {
        match self {
            C4Player::Red => C4Player::Yellow,
            C4Player::Yellow => C4Player::Red,
        }
    }
}

#[derive(PartialEq, Debug, Eq, Hash)]
pub enum C4Result {
    Draw,
    Win(C4Player),
}

#[derive(Clone, Eq, Hash, PartialEq)]
/// A Connnect Four game state
/// Index for the bottom left is 0, bottom right is 6 and for the top right is 41
pub struct C4State {
    board: [Option<C4Player>; 42],
    pub whose_turn: C4Player,
}

impl C4State {
    pub fn new() -> C4State {
        C4State {
            board: [None; 42],
            whose_turn: C4Player::Yellow,
        }
    }

    fn with_index(&self, idx: usize) -> Result<C4State, ()> {
        if idx >= 6 * 7 || self.board[idx].is_some() {
            return Err(());
        }
        let mut pos = self.board.clone();
        pos[idx] = Some(self.whose_turn);
        Ok(C4State {
            board: pos,
            whose_turn: self.whose_turn.other(),
        })
    }
}

impl Debug for C4State {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for row in (0..6).rev() {
            for col in 0..7 {
                match self.board[row * 7 + col] {
                    Some(C4Player::Yellow) => write!(f, "Y")?,
                    Some(C4Player::Red) => write!(f, "R")?,
                    None => write!(f, ".")?,
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Node for C4State {
    fn possible_next_states(&self) -> Vec<C4State> {
        (0..7)
            .into_iter()
            .filter_map(|col| {
                if self.board[5 * 7 + col].is_none() {
                    let max_row: usize = self
                        .board
                        .iter()
                        .skip(col)
                        .step_by(7)
                        .enumerate()
                        .find_map(|(i, f)| if f.is_none() { Some(i) } else { None })
                        .unwrap_or(col);
                    Some(self.with_index(max_row * 7 + col).unwrap())
                } else {
                    None
                }
            })
            .collect()
    }

    type Winner = C4Result;

    /// Reasons for a win:
    /// - 4 in a row horizontally
    /// - 4 in a row vertically
    /// - 4 in a row diagonally
    ///
    /// A draw only happens if the board is full and none of the above hold.
    /// Otherwise, the game is still ongoing
    fn is_terminal(&self) -> Option<Self::Winner> {
        for row in 0..6 {
            if let Some(winner) = self
                .board
                .iter()
                .skip(row * 7)
                .take(7)
                .tuple_windows()
                .filter_map(|(a, b, c, d)| {
                    if a.is_some() && a == b && b == c && c == d {
                        Some(a.unwrap())
                    } else {
                        None
                    }
                })
                .next()
            {
                return Some(C4Result::Win(winner));
            }
        }
        for col in 0..7 {
            if let Some(winner) = self
                .board
                .iter()
                .skip(col)
                .step_by(7)
                .tuple_windows()
                .filter_map(|(a, b, c, d)| {
                    if a.is_some() && a == b && b == c && c == d {
                        Some(a.unwrap())
                    } else {
                        None
                    }
                })
                .next()
            {
                return Some(C4Result::Win(winner));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    impl C4State {
        pub fn from_slice(v: &[Option<C4Player>], whose_turn: C4Player) -> C4State {
            let mut pos = [None; 42];
            for (i, player) in v.iter().enumerate() {
                pos[i] = *player;
            }
            C4State {
                board: pos,
                whose_turn,
            }
        }
    }

    #[test]
    fn can_print() {
        let pos = C4State::from_slice(
            &[
                None,
                None,
                None,
                Some(C4Player::Yellow),
                Some(C4Player::Red),
            ],
            C4Player::Yellow,
        );
        dbg!(pos);
    }

    #[test]
    fn can_find_possible_next_states() {
        let pos = C4State::from_slice(
            &[
                None,
                None,
                None,
                Some(C4Player::Yellow),
                Some(C4Player::Red),
            ],
            C4Player::Yellow,
        );
        let next_states = pos.possible_next_states();
        assert_eq!(next_states.len(), 7);
    }

    #[test]
    fn can_find_possible_next_states_if_first_row_is_full() {
        let pos = C4State::from_slice(
            &[
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
            ],
            C4Player::Red,
        );
        let next_states = pos.possible_next_states();
        assert_eq!(next_states.len(), 7);
    }

    #[test]
    fn only_one_possible_next_state() {
        let pos = C4State::from_slice(&vec![Some(C4Player::Yellow); 41], C4Player::Red);
        let next_states = pos.possible_next_states();
        assert_eq!(next_states.len(), 1);
    }

    #[test]
    fn four_in_a_row_horizontally_is_terminal() {
        let pos = C4State::from_slice(
            &[
                Some(C4Player::Yellow),
                Some(C4Player::Yellow),
                Some(C4Player::Yellow),
                Some(C4Player::Yellow),
            ],
            C4Player::Red,
        );
        assert_eq!(pos.is_terminal(), Some(C4Result::Win(C4Player::Yellow)));
    }

    #[test]
    fn four_in_the_second_row_horizontally_is_terminal() {
        let pos = C4State::from_slice(
            &[
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Red),
                Some(C4Player::Red),
                Some(C4Player::Red),
            ],
            C4Player::Red,
        );
        assert_eq!(pos.is_terminal(), Some(C4Result::Win(C4Player::Red)));
    }

    #[test]
    fn four_vertically_is_terminal() {
        let pos = C4State::from_slice(
            &[
                // Row 1
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                // Row 2
                Some(C4Player::Yellow),
                None,
                None,
                None,
                None,
                None,
                None,
                // Row 3
                Some(C4Player::Yellow),
                None,
                None,
                None,
                None,
                None,
                None,
                // Row 4
                Some(C4Player::Yellow),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            C4Player::Red,
        );
        assert_eq!(pos.is_terminal(), Some(C4Result::Win(C4Player::Yellow)));
    }

    #[test]
    fn four_vertically_in_last_column_is_terminal() {
        let pos = C4State::from_slice(
            &[
                // Row 1
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Yellow),
                Some(C4Player::Red),
                Some(C4Player::Red),
                // Row 2
                None,
                None,
                None,
                None,
                None,
                None,
                Some(C4Player::Yellow),
                // Row 3
                None,
                None,
                None,
                None,
                None,
                None,
                Some(C4Player::Red),
                // Row 4
                None,
                None,
                None,
                None,
                None,
                None,
                Some(C4Player::Red),
                // Row 5
                None,
                None,
                None,
                None,
                None,
                None,
                Some(C4Player::Red),
                // Row 6
                None,
                None,
                None,
                None,
                None,
                None,
                Some(C4Player::Red),
            ],
            C4Player::Red,
        );
        assert_eq!(pos.is_terminal(), Some(C4Result::Win(C4Player::Red)));
    }
}
