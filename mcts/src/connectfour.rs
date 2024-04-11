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
/// A Tic Tac Toe game state
pub struct TTTPos {
    board: [Option<C4Player>; 42],
    pub whose_turn: C4Player,
}
