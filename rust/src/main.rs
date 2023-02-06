use backgammon_board::BackgammonBoard;
use rand::{rngs::ThreadRng, Rng};
use std::{
    io::Error,
    io::{self, stdout, Write},
    num::ParseIntError,
};

mod backgammon_board;

struct PairOfDice {
    rng: ThreadRng,
}

impl PairOfDice {
    fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
    fn throw_dice(&mut self) -> (u8, u8) {
        (self.rng.gen_range(1..=6), self.rng.gen_range(1..=6))
    }
}

#[derive(Debug)]
enum GameError {
    NeedDice(State),
    CannotParseInput(Error),
    CannotParseMove(ParseIntError),
    InvalidMove(State),
}

#[derive(Debug)]
enum State {
    Initial,
    ThrowDice,
    Move(((u8, u8), u8)),
}

impl State {
    fn next(self, dice: Option<(u8, u8)>) -> Result<Self, GameError> {
        match self {
            State::Initial => Ok(State::ThrowDice),
            State::ThrowDice => Ok(State::Move(match dice {
                Some(dice) => {
                    if dice.0 == dice.1 {
                        (dice, 4)
                    } else {
                        (dice, 2)
                    }
                }
                None => return Err(GameError::NeedDice(self)),
            })),
            State::Move((dice, n)) => {
                if n - 1 == 0 {
                    Ok(State::Initial)
                } else {
                    Ok(State::Move((dice, n - 1)))
                }
            }
        }
    }

    fn as_prompt(&self) -> String {
        match self {
            State::Initial => "I ".to_string(),
            State::ThrowDice => "D".to_string(),
            State::Move((_, n)) => format!("M{n}"),
        }
    }
}

fn main() -> ! {
    let mut dice = PairOfDice::new();
    let mut board = BackgammonBoard::new();
    let mut state = State::Initial;

    loop {
        let mut input = String::new();
        print!("{}>: ", state.as_prompt());
        stdout().flush().unwrap();
        let new_state = match io::stdin().read_line(&mut input) {
            Ok(_n) => match state {
                State::Initial => {
                    println!("");
                    board.print();
                    state.next(None)
                }
                State::ThrowDice => {
                    let dice = dice.throw_dice();
                    println!("{:?}", &dice);
                    state.next(Some(dice))
                }
                State::Move((dice, _)) => {
                    let mov: Result<Vec<usize>, _> = input
                        .trim()
                        .split(",")
                        .map(|v| v.parse())
                        .collect::<Result<_, _>>();
                    match mov {
                        Ok(mov) => match board.apply_move(dice, mov[0], mov[1]) {
                            Err(_) => Err(GameError::InvalidMove(state)),
                            Ok(_) => state.next(None),
                        },
                        Err(e) => Err(GameError::CannotParseMove(e)),
                    }
                }
            },
            Err(error) => Err(GameError::CannotParseInput(error)),
        };
        match new_state {
            Ok(new_state) => state = new_state,
            Err(error) => {
                println!("error: {:?}", error);
                match error {
                    GameError::NeedDice(old_state) => state = old_state,
                    GameError::InvalidMove(old_state) => state = old_state,
                    _ => state = State::Initial,
                }
            }
        }
    }
}
