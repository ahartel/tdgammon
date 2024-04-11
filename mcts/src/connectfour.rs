use std::fmt::{self, Debug, Formatter};

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

impl C4State {}

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
        todo!()
        // self.board
        //     .iter()
        //     .enumerate()
        //     .filter_map(|(i, &player)| {
        //         if player.is_none() {
        //             Some(self.with_index(i).unwrap())
        //         } else {
        //             None
        //         }
        //     })
        //     .collect()
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
}
