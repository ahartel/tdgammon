use backgammon_board::{Actor, BoardIO, Dice, BLACK_SQUARE, WHITE_SQUARE};
use rand::{rngs::ThreadRng, Rng};
use std::io::{self, stdout, Write};

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
    fn throw_dice(&mut self) -> Dice {
        Dice::new(self.rng.gen_range(1..=6), self.rng.gen_range(1..=6))
    }
}

#[derive(Debug)]
enum GameError {
    NeedDice(State),
    CannotParseInput,
    InvalidMove(State),
    InvalidDieToRemove,
}

#[derive(Debug)]
enum State {
    Initial,
    ThrowDiceWhite,
    MoveWhite(Dice),
    ThrowDiceBlack,
    MoveBlack(Dice),
}

impl State {
    fn next(self, dice: Option<Dice>, remove_die: Option<u8>) -> Result<Self, GameError> {
        match self {
            State::Initial => Ok(State::ThrowDiceWhite),
            State::ThrowDiceWhite => Ok(State::MoveWhite(match dice {
                Some(dice) => dice,
                None => return Err(GameError::NeedDice(self)),
            })),
            State::MoveWhite(mut dice) => {
                if remove_die.is_none() {
                    return Err(GameError::InvalidDieToRemove);
                }
                let remove_die = remove_die.unwrap();
                if dice.contains(&remove_die) {
                    if dice.num_moves() == 1 {
                        Ok(State::ThrowDiceBlack)
                    } else {
                        dice.remove(&remove_die).unwrap();
                        Ok(State::MoveWhite(dice))
                    }
                } else {
                    Err(GameError::InvalidDieToRemove)
                }
            }
            State::ThrowDiceBlack => Ok(State::MoveBlack(match dice {
                Some(dice) => dice,
                None => return Err(GameError::NeedDice(self)),
            })),
            State::MoveBlack(mut dice) => {
                println!("Dice: {dice:?}");
                if remove_die.is_none() {
                    return Err(GameError::InvalidDieToRemove);
                }
                let remove_die = remove_die.unwrap();
                println!("Remove_die: {remove_die}");
                if dice.contains(&remove_die) {
                    if dice.num_moves() == 1 {
                        Ok(State::ThrowDiceWhite)
                    } else {
                        dice.remove(&remove_die).unwrap();
                        Ok(State::MoveBlack(dice))
                    }
                } else {
                    Err(GameError::InvalidDieToRemove)
                }
            }
        }
    }

    fn as_prompt(&self) -> String {
        match self {
            State::Initial => "I 🎉".to_string(),
            State::ThrowDiceWhite => format!("D {}", WHITE_SQUARE),
            State::MoveWhite(d) => format!("M{}{WHITE_SQUARE}", d.num_moves()),
            State::ThrowDiceBlack => format!("D {}", BLACK_SQUARE),
            State::MoveBlack(d) => format!("M{}{BLACK_SQUARE}", d.num_moves()),
        }
    }
}

fn main() -> ! {
    let mut dice = PairOfDice::new();
    let mut board = BoardIO::new();
    let mut state = State::Initial;

    loop {
        let mut input = String::new();
        print!("{}>: ", state.as_prompt());
        stdout().flush().unwrap();
        let new_state = match io::stdin().read_line(&mut input) {
            Ok(_n) => match state {
                State::Initial => state.next(None, None),
                State::ThrowDiceWhite => {
                    let dice = dice.throw_dice();
                    println!("{:?}", &dice);
                    state.next(Some(dice), None)
                }
                State::MoveWhite(ref dice) => {
                    match board.read_and_apply_move(input, Actor::White, dice) {
                        Err(_) => Err(GameError::InvalidMove(state)),
                        Ok((remove_die, _)) => state.next(None, remove_die),
                    }
                }
                State::ThrowDiceBlack => {
                    let dice = dice.throw_dice();
                    println!("{:?}", &dice);
                    state.next(Some(dice), None)
                }
                State::MoveBlack(ref dice) => {
                    match board.read_and_apply_move(input, Actor::Black, dice) {
                        Err(_) => Err(GameError::InvalidMove(state)),
                        Ok((remove_die, _)) => state.next(None, remove_die),
                    }
                }
            },
            Err(_error) => Err(GameError::CannotParseInput),
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
        println!("");
        board.print();
    }
}
