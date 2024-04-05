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

#[derive(PartialEq)]
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
        actor: Actor,
        dice: &[u8],
        from: usize,
        to: usize,
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
        dice: &[u8],
    ) -> Result<(Option<u8>, bool), BoardError> {
        let mov: Result<Vec<usize>, _> = input
            .trim()
            .split(",")
            .map(|v| v.parse())
            .collect::<Result<_, _>>();
        match mov {
            Ok(mov) => {
                if mov.len() == 2 {
                    self.board.apply_move(actor, dice, mov[0] - 1, mov[1] - 1)
                } else if mov.len() == 1 {
                    self.board.apply_move(actor, dice, POINT, mov[0] - 1)
                } else {
                    Err(BoardError::InvalidMove)
                }
            }
            Err(_e) => Err(BoardError::CannotParseMove),
        }
    }
}
