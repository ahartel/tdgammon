use std::collections::HashMap;

use tictactoe_mcts::{
    connectfour::{C4Player, C4Result, C4State},
    searchtree::{Node, SearchTree},
    tictactoe::{TTTPlayer, TTTPos, TTTResult},
};
use tqdm::tqdm;

fn main() {
    evaluate_tictactoe_performance();
    evaluate_connectfour_performance();
}

fn evaluate_tictactoe_performance() {
    println!("TIC TAC TOE");
    println!("===========");
    let mut results = HashMap::new();
    let num_games = 1000;
    for _ in tqdm(0..num_games) {
        let final_pos = tictactoe_mcts_vs_random_player();
        results
            .entry(final_pos.is_terminal().unwrap())
            .and_modify(|e| *e += 1)
            .or_insert(1);
    }
    println!(
        "Player X (random) won {}% of the games",
        results.get(&TTTResult::Win(TTTPlayer::X)).unwrap_or(&0) * 100 / num_games
    );
    println!(
        "Player O (MCTS) won {}% of the games",
        results.get(&TTTResult::Win(TTTPlayer::O)).unwrap_or(&0) * 100 / num_games
    );
    println!(
        "{}% of the games were draws",
        results.get(&TTTResult::Draw).unwrap_or(&0) * 100 / num_games
    );
}

fn evaluate_connectfour_performance() {
    println!("Connect 4");
    println!("=========");
    let mut results = HashMap::new();
    let num_games = 1000;
    for _ in tqdm(0..num_games) {
        let final_pos = connectfour_mcts_vs_random_player();
        results
            .entry(final_pos.is_terminal().unwrap())
            .and_modify(|e| *e += 1)
            .or_insert(1);
    }
    println!(
        "Player Y (random) won {}% of the games",
        results.get(&C4Result::Win(C4Player::Yellow)).unwrap_or(&0) * 100 / num_games
    );
    println!(
        "Player R (MCTS) won {}% of the games",
        results.get(&C4Result::Win(C4Player::Red)).unwrap_or(&0) * 100 / num_games
    );
    println!(
        "{}% of the games were draws",
        results.get(&C4Result::Draw).unwrap_or(&0) * 100 / num_games
    );
}

fn simulate_game<P: Node>(tree: &mut SearchTree<P>, start_state: &P) -> Option<P::Winner> {
    let mut current_pos = start_state.to_owned();
    while let Some(child) = tree.random_next_state(&current_pos) {
        if let Some(result) = child.is_terminal() {
            return Some(result);
        }
        current_pos = child;
    }
    return None;
}

fn tictactoe_mcts_vs_random_player() -> TTTPos {
    let root_pos = TTTPos::new();
    let mut tree = SearchTree::new(root_pos.clone());
    let mut current_pos = root_pos;
    while !current_pos.is_terminal().is_some() {
        match current_pos.whose_turn {
            TTTPlayer::X => {
                let next_state = tree.random_next_state(&current_pos).unwrap();
                current_pos = next_state;
            }
            TTTPlayer::O => {
                for _ in 0..50 {
                    let leaf = tree.find_most_valuable_leaf(&current_pos).to_owned();
                    if leaf.is_terminal().is_some() {
                        continue;
                    }
                    let new_children = leaf.possible_next_states();
                    tree.add_children(&leaf, new_children);
                    let simulated_child = tree.random_child(&leaf).unwrap();
                    let winner = simulate_game(&mut tree, &simulated_child);
                    tree.add_visit(
                        &simulated_child,
                        match winner {
                            Some(TTTResult::Win(TTTPlayer::X)) => -1.0,
                            Some(TTTResult::Win(TTTPlayer::O)) => 1.0,
                            _ => 0.5,
                        },
                    );
                }
                current_pos = tree.best_child(&current_pos).unwrap();
            }
        }
    }
    current_pos
}

fn connectfour_mcts_vs_random_player() -> C4State {
    let root_pos = C4State::new();
    let mut tree = SearchTree::new(root_pos.clone());
    let mut current_pos = root_pos;
    while !current_pos.is_terminal().is_some() {
        // dbg!(&current_pos);
        match current_pos.whose_turn {
            C4Player::Yellow => {
                let next_state = tree.random_next_state(&current_pos).unwrap();
                current_pos = next_state;
            }
            C4Player::Red => {
                for _ in 0..75 {
                    let leaf = tree.find_most_valuable_leaf(&current_pos).to_owned();
                    if leaf.is_terminal().is_some() {
                        continue;
                    }
                    let new_children = leaf.possible_next_states();
                    tree.add_children(&leaf, new_children);
                    let simulated_child = tree.random_child(&leaf).unwrap();
                    let winner = simulate_game(&mut tree, &simulated_child);
                    tree.add_visit(
                        &simulated_child,
                        match winner {
                            Some(C4Result::Win(C4Player::Yellow)) => -1.0,
                            Some(C4Result::Win(C4Player::Red)) => 1.0,
                            _ => 0.5,
                        },
                    );
                }
                current_pos = tree.best_child(&current_pos).unwrap();
            }
        }
    }
    // dbg!(&current_pos);
    current_pos
}
