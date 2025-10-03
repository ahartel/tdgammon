use std::io::{self, Write, stdout};

use backgammon::backgammon_board::BoardError;
use backgammon::backgammon_board::{Actor, BoardIO};
use backgammon::game_state::{GameError, GameState, PairOfDice};

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
