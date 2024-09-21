#[derive(Debug)]
pub enum BoardError {
    NoStoneFound,
    DistanceNotInDice,
    MustMoveFromPoint,
    InvalidMove,
    CannotParseMove,
}

pub const WHITE_SQUARE: char = 'ø';
pub const BLACK_SQUARE: char = 'o';

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Actor {
    White,
    Black,
}

// 24 -> hit off the board
const POINT: usize = 24;
// 25 -> born off
const BEARING_TABLE: usize = 25;

pub struct BackgammonBoard {
    white: [u8; 26],
    black: [u8; 26],
}

#[derive(Debug, PartialEq)]
pub struct Move {
    actor: Actor,
    from: usize,
    to: usize,
}

#[derive(Debug, PartialEq)]
pub struct Dice([Option<u8>; 4]);
impl Dice {
    pub fn contains(&self, distance: &u8) -> bool {
        self.0
            .iter()
            .any(|x| x.is_some() && x.unwrap() == *distance)
    }

    pub fn num_moves(&self) -> usize {
        self.0.iter().filter(|x| x.is_some()).count()
    }

    pub fn remove(&mut self, distance: &u8) -> Option<()> {
        for i in 0..4 {
            if self.0[i] == Some(*distance) {
                self.0[i] = None;
                return Some(());
            }
        }
        None
    }

    pub fn new(one: u8, two: u8) -> Self {
        if one == two {
            Dice([Some(one), Some(one), Some(two), Some(two)])
        } else {
            Dice([Some(one), Some(two), None, None])
        }
    }

    fn values(&self) -> Vec<u8> {
        self.0.iter().filter_map(|x| x.map(|x| x.clone())).collect()
    }
}

impl BackgammonBoard {
    fn new() -> Self {
        let mut initial_white = [0; 26];
        initial_white[0] = 2;
        initial_white[11] = 5;
        initial_white[16] = 3;
        initial_white[18] = 5;

        let mut initial_black = [0; 26];
        initial_black[5] = 5;
        initial_black[7] = 3;
        initial_black[12] = 5;
        initial_black[23] = 2;
        BackgammonBoard {
            white: initial_white,
            black: initial_black,
        }
    }

    fn apply_move(
        &mut self,
        Move { actor, from, to }: Move,
        dice: &Dice,
    ) -> Result<(Option<u8>, bool), BoardError> {
        match actor {
            Actor::White => {
                if self.white[BEARING_TABLE] == 15 {
                    Ok((None, true))
                } else if self.white[POINT] > 0 {
                    if from == POINT && self.black[to] <= 1 {
                        if self.black[to] == 1 {
                            self.white[from] -= 1;
                            self.white[to] += 1;
                            self.black[to] -= 1;
                            self.black[POINT] += 1;
                        } else {
                            self.white[from] -= 1;
                            self.white[to] += 1;
                        }
                        Ok((Some((to + 1) as u8), false))
                    } else {
                        Err(BoardError::MustMoveFromPoint)
                    }
                } else if self.white[from] > 0 && self.black[to] <= 1 {
                    if to <= from || !dice.contains(&((to - from) as u8)) {
                        Err(BoardError::DistanceNotInDice)
                    } else if self.black[to] == 1 {
                        self.white[from] -= 1;
                        self.white[to] += 1;
                        self.black[to] -= 1;
                        self.black[POINT] += 1;
                        Ok((Some((to - from) as u8), false))
                    } else {
                        self.white[from] -= 1;
                        self.white[to] += 1;
                        Ok((Some((to - from) as u8), false))
                    }
                } else {
                    Err(BoardError::NoStoneFound)
                }
            }
            Actor::Black => {
                if self.black[BEARING_TABLE] == 15 {
                    Ok((None, true))
                } else if self.black[POINT] > 0 {
                    if from == POINT && self.white[to] <= 1 {
                        if self.white[to] == 1 {
                            self.black[from] -= 1;
                            self.black[to] += 1;
                            self.white[to] -= 1;
                            self.white[POINT] += 1;
                        } else {
                            self.black[from] -= 1;
                            self.black[to] += 1;
                        }
                        Ok((Some((25 - to) as u8), false))
                    } else {
                        Err(BoardError::MustMoveFromPoint)
                    }
                } else if self.black[from] > 0 && self.white[to] <= 1 {
                    if from <= to || !dice.contains(&((from - to) as u8)) {
                        Err(BoardError::DistanceNotInDice)
                    } else if self.white[to] == 1 {
                        self.black[from] -= 1;
                        self.black[to] += 1;
                        self.white[to] -= 1;
                        self.white[POINT] += 1;
                        Ok((Some((from - to) as u8), false))
                    } else {
                        self.black[from] -= 1;
                        self.black[to] += 1;
                        Ok((Some((from - to) as u8), false))
                    }
                } else {
                    Err(BoardError::NoStoneFound)
                }
            }
        }
    }

    fn possible_next_moves(&self, dice: Dice, actor: Actor) -> Vec<Move> {
        match actor {
            Actor::White => {
                let mut moves = vec![];
                for idx in 0..=POINT {
                    if self.white[idx] > 0 {
                        for value in dice.values() {
                            if idx < (POINT - value as usize)
                                && self.black[idx + value as usize] <= 1
                            {
                                moves.push(Move {
                                    actor,
                                    from: idx,
                                    to: idx + value as usize,
                                });
                            }
                        }
                    }
                }
                moves
            }
            Actor::Black => {
                let mut moves = vec![];
                for idx in 0..=POINT {
                    if self.black[idx] > 0 {
                        for value in dice.values() {
                            if idx > value as usize && self.white[idx - value as usize] <= 1 {
                                moves.push(Move {
                                    actor,
                                    from: idx,
                                    to: idx - value as usize,
                                });
                            }
                        }
                    }
                }
                moves
            }
        }
    }
}

pub struct BoardIO {
    board: BackgammonBoard,
}

impl BoardIO {
    pub fn new() -> Self {
        BoardIO {
            board: BackgammonBoard::new(),
        }
    }

    pub fn print(&self) {
        for idx in 12..24 {
            print!("{:02} ", idx + 1);
        }
        println!("BT");
        for idx in 12..24 {
            if self.board.white[idx] > 0 {
                print!(" {} ", WHITE_SQUARE);
            } else if self.board.black[idx] > 0 {
                print!(" {} ", BLACK_SQUARE);
            } else {
                print!(" . ");
            }
        }
        println!(" {}", WHITE_SQUARE);
        for idx in 12..24 {
            if self.board.white[idx] > 0 {
                print!(" {} ", self.board.white[idx]);
            } else if self.board.black[idx] > 0 {
                print!(" {} ", self.board.black[idx]);
            } else {
                print!(" . ");
            }
        }
        println!(" {}", self.board.black[BEARING_TABLE]);
        println!("");
        println!(
            "            {}: {}  {}: {}",
            BLACK_SQUARE, self.board.black[POINT], WHITE_SQUARE, self.board.white[POINT]
        );
        println!("");
        for idx in (0..12).rev() {
            if self.board.white[idx] > 0 {
                print!(" {} ", self.board.white[idx]);
            } else if self.board.black[idx] > 0 {
                print!(" {} ", self.board.black[idx]);
            } else {
                print!(" . ");
            }
        }
        println!(" {}", self.board.white[BEARING_TABLE]);
        for idx in (0..12).rev() {
            if self.board.white[idx] > 0 {
                print!(" {} ", WHITE_SQUARE);
            } else if self.board.black[idx] > 0 {
                print!(" {} ", BLACK_SQUARE);
            } else {
                print!(" . ");
            }
        }
        println!(" {}", BLACK_SQUARE);
        for idx in (0..12).rev() {
            print!("{:02} ", idx + 1);
        }
        println!("BT");
    }

    pub fn read_and_apply_move(
        &mut self,
        input: String,
        actor: Actor,
        dice: &Dice,
    ) -> Result<(Option<u8>, bool), BoardError> {
        let mov: Result<Vec<usize>, _> = input
            .trim()
            .split(",")
            .map(|v| v.parse())
            .collect::<Result<_, _>>();
        match mov {
            Ok(mov) => {
                if mov.len() == 2 {
                    self.board.apply_move(
                        Move {
                            actor,
                            from: mov[0] - 1,
                            to: mov[1] - 1,
                        },
                        dice,
                    )
                } else if mov.len() == 1 {
                    self.board.apply_move(
                        Move {
                            actor,
                            from: POINT,
                            to: mov[0] - 1,
                        },
                        dice,
                    )
                } else {
                    Err(BoardError::InvalidMove)
                }
            }
            Err(_e) => Err(BoardError::CannotParseMove),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backgammon_board::{BackgammonBoard, Dice, Move, POINT};

    use super::Actor;

    impl BackgammonBoard {
        fn empty() -> Self {
            let initial_white = [0; 26];
            let initial_black = [0; 26];

            BackgammonBoard {
                white: initial_white,
                black: initial_black,
            }
        }

        fn with_piece(mut self, actor: Actor, point: usize) -> Self {
            match actor {
                Actor::White => self.white[point] += 1,
                Actor::Black => self.black[point] += 1,
            }
            self
        }

        fn with_pieces(mut self, positions: Vec<(Actor, usize)>) -> Self {
            for (actor, point) in positions {
                self = self.with_piece(actor, point);
            }
            self
        }
    }

    #[test]
    fn two_possible_next_moves_for_white() {
        let board = BackgammonBoard::empty().with_piece(Actor::White, 0);
        let dice = Dice([Some(1), Some(2), None, None]);
        let moves = board.possible_next_moves(dice, Actor::White);
        assert_eq!(
            moves,
            vec![
                Move {
                    actor: Actor::White,
                    from: 0,
                    to: 1
                },
                Move {
                    actor: Actor::White,
                    from: 0,
                    to: 2
                }
            ]
        );
    }

    #[test]
    fn black_on_point_blocked_by_white() {
        let board = BackgammonBoard::empty().with_pieces(vec![
            (Actor::White, 23),
            (Actor::White, 23),
            (Actor::White, 22),
            (Actor::White, 22),
            (Actor::Black, POINT),
        ]);
        let dice = Dice([Some(1), Some(2), None, None]);
        let moves = board.possible_next_moves(dice, Actor::Black);
        assert_eq!(moves, vec![]);
    }

    #[test]
    fn black_blocked_by_white() {
        let board = BackgammonBoard::empty().with_pieces(vec![
            (Actor::White, 1),
            (Actor::White, 1),
            (Actor::White, 2),
            (Actor::White, 2),
            (Actor::Black, 6),
        ]);
        let dice = Dice([Some(4), Some(4), Some(4), Some(4)]);
        let moves = board.possible_next_moves(dice, Actor::Black);
        assert_eq!(moves, vec![]);
    }

    #[test]
    fn black_can_move_normally() {
        let board = BackgammonBoard::empty().with_pieces(vec![
            (Actor::Black, 4),
            (Actor::Black, 9),
            (Actor::Black, 10),
        ]);
        let dice = Dice([Some(3), Some(5), None, None]);
        let moves = board.possible_next_moves(dice, Actor::Black);
        assert_eq!(moves.len(), 5);
    }

    #[test]
    fn white_can_move_normally() {
        let board = BackgammonBoard::empty().with_pieces(vec![
            (Actor::White, 21),
            (Actor::White, 12),
            (Actor::White, 17),
            (Actor::Black, 23),
            (Actor::Black, 20),
            (Actor::Black, 20),
        ]);
        let dice = Dice([Some(3), Some(5), None, None]);
        let moves = board.possible_next_moves(dice, Actor::White);
        assert_eq!(moves.len(), 3);
    }
}
