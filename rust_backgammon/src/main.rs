use backgammon_board::{Actor, BoardIO, Dice, BLACK_SQUARE, WHITE_SQUARE};
use rand::{rngs::ThreadRng, Rng};
use std::io::{self, stdout, Write};

use crate::backgammon_board::BoardError;

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
    NeedDice(GameState),
    CannotParseInput,
    InvalidMove(GameState),
    InvalidDieToRemove(GameState),
}

#[derive(Debug)]
enum GameState {
    Initial,
    ThrowDiceWhite,
    MoveWhite(Dice),
    ThrowDiceBlack,
    MoveBlack(Dice),
}

impl GameState {
    fn next(self, dice: Option<Dice>, remove_die: Option<u8>) -> Result<Self, GameError> {
        match self {
            GameState::Initial => Ok(GameState::ThrowDiceWhite),
            GameState::ThrowDiceWhite => Ok(GameState::MoveWhite(match dice {
                Some(dice) => dice,
                None => return Err(GameError::NeedDice(self)),
            })),
            GameState::MoveWhite(mut dice) => {
                if remove_die.is_none() {
                    return Err(GameError::InvalidDieToRemove(GameState::MoveWhite(dice)));
                }
                let remove_die = remove_die.unwrap();
                if dice.contains(&remove_die) {
                    if dice.num_moves() == 1 {
                        Ok(GameState::ThrowDiceBlack)
                    } else {
                        dice.remove(&remove_die).unwrap();
                        Ok(GameState::MoveWhite(dice))
                    }
                } else {
                    Err(GameError::InvalidDieToRemove(GameState::MoveWhite(dice)))
                }
            }
            GameState::ThrowDiceBlack => Ok(GameState::MoveBlack(match dice {
                Some(dice) => dice,
                None => return Err(GameError::NeedDice(self)),
            })),
            GameState::MoveBlack(mut dice) => {
                println!("Dice: {dice:?}");
                if remove_die.is_none() {
                    return Err(GameError::InvalidDieToRemove(GameState::MoveBlack(dice)));
                }
                let remove_die = remove_die.unwrap();
                println!("Remove_die: {remove_die}");
                if dice.contains(&remove_die) {
                    if dice.num_moves() == 1 {
                        Ok(GameState::ThrowDiceWhite)
                    } else {
                        dice.remove(&remove_die).unwrap();
                        Ok(GameState::MoveBlack(dice))
                    }
                } else {
                    Err(GameError::InvalidDieToRemove(GameState::MoveBlack(dice)))
                }
            }
        }
    }

    fn as_prompt(&self) -> String {
        match self {
            GameState::Initial => "I 🎉".to_string(),
            GameState::ThrowDiceWhite => format!("D {}", WHITE_SQUARE),
            GameState::MoveWhite(d) => format!("M{}{WHITE_SQUARE}", d.num_moves()),
            GameState::ThrowDiceBlack => format!("D {}", BLACK_SQUARE),
            GameState::MoveBlack(d) => format!("M{}{BLACK_SQUARE}", d.num_moves()),
        }
    }
}

fn main() -> ! {
    let mut dice = PairOfDice::new();
    let mut board = BoardIO::new();
    let mut state = GameState::Initial;

    loop {
        let mut input = String::new();
        print!("{}>: ", state.as_prompt());
        stdout().flush().unwrap();
        let new_state = match io::stdin().read_line(&mut input) {
            Ok(_n) => match state {
                GameState::Initial => state.next(None, None),
                GameState::ThrowDiceWhite => {
                    let dice = dice.throw_dice();
                    println!("{:?}", &dice);
                    state.next(Some(dice), None)
                }
                GameState::MoveWhite(ref dice) => {
                    match board.read_and_apply_move(input, Actor::White, dice) {
                        Err(e) => match e {
                            BoardError::NoPossibleMoves => Ok(GameState::ThrowDiceBlack),
                            _ => Err(GameError::InvalidMove(state)),
                        },
                        Ok((remove_die, _)) => state.next(None, remove_die),
                    }
                }
                GameState::ThrowDiceBlack => {
                    let dice = dice.throw_dice();
                    println!("{:?}", &dice);
                    state.next(Some(dice), None)
                }
                GameState::MoveBlack(ref dice) => {
                    match board.read_and_apply_move(input, Actor::Black, dice) {
                        Err(e) => match e {
                            BoardError::NoPossibleMoves => Ok(GameState::ThrowDiceWhite),
                            _ => Err(GameError::InvalidMove(state)),
                        },
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
                    GameError::InvalidDieToRemove(old_state) => state = old_state,
                    _ => state = GameState::Initial,
                }
            }
        }
        match state {
            GameState::MoveWhite(ref dice) => {
                board.print_possible_next_moves(Actor::White, dice);
            }
            GameState::MoveBlack(ref dice) => {
                board.print_possible_next_moves(Actor::Black, dice);
            }
            _ => {}
        }
        println!("");
        board.print();
        println!("");
    }
}
