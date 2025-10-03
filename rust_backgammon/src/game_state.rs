use crate::backgammon_board::{BLACK_SQUARE, Dice, WHITE_SQUARE};
use rand::{Rng, rngs::ThreadRng};

pub struct PairOfDice {
    rng: ThreadRng,
}

impl PairOfDice {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    pub fn throw_dice(&mut self) -> Dice {
        Dice::new(self.rng.gen_range(1..=6), self.rng.gen_range(1..=6))
    }
}

#[derive(Debug)]
pub enum GameError {
    NeedDice(GameState),
    CannotParseInput,
    InvalidMove(GameState),
    InvalidDieToRemove(GameState),
}

#[derive(Debug)]
pub enum GameState {
    Initial,
    ThrowDiceWhite,
    MoveWhite(Dice),
    ThrowDiceBlack,
    MoveBlack(Dice),
}

impl GameState {
    pub fn next(self, dice: Option<Dice>, remove_die: Option<u8>) -> Result<Self, GameError> {
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

    pub fn as_prompt(&self) -> String {
        match self {
            GameState::Initial => "I 🎉".to_string(),
            GameState::ThrowDiceWhite => format!("D {}", WHITE_SQUARE),
            GameState::MoveWhite(d) => format!("M{}{WHITE_SQUARE}", d.num_moves()),
            GameState::ThrowDiceBlack => format!("D {}", BLACK_SQUARE),
            GameState::MoveBlack(d) => format!("M{}{BLACK_SQUARE}", d.num_moves()),
        }
    }
}
