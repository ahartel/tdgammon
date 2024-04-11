use std::collections::HashMap;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;

use rand::Rng;
use tqdm::tqdm;

fn main() {
    let mut results = HashMap::new();
    let num_games = 1000;
    for _ in tqdm(0..num_games) {
        let final_pos = play_against_random_player();
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
        "Player O (random) won {}% of the games",
        results.get(&TTTResult::Win(TTTPlayer::O)).unwrap_or(&0) * 100 / num_games
    );
    println!(
        "{}% of the games were draws",
        results.get(&TTTResult::Draw).unwrap_or(&0) * 100 / num_games
    );
}

fn simulate_game(tree: &mut SearchTree<TTTPos>, start_state: &TTTPos) -> Option<TTTResult> {
    let mut current_pos = start_state.to_owned();
    while let Some(child) = tree.random_next_state(&current_pos) {
        if let Some(result) = child.is_terminal() {
            return Some(result);
        }
        current_pos = child;
    }
    return None;
}

fn play_against_random_player() -> TTTPos {
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
                for _ in 0..30 {
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

trait Node: Eq + Hash + Clone {
    fn possible_next_states(&self) -> Vec<Self>;
}

#[derive(Debug, Copy, Clone)]
/// How many times a node has been visited and how many points it has accumulated.
struct Score {
    points: f32,
    visits: i32,
}

impl Score {
    fn zero() -> Score {
        Score {
            points: 0.0,
            visits: 0,
        }
    }

    fn as_f32(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        self.points / self.visits as f32
    }
}

#[derive(Debug)]
struct SearchTree<N: Node> {
    _root: N,
    children: HashMap<N, Vec<N>>,
    scores: HashMap<N, Score>,
    parents: HashMap<N, N>,
}

impl<N: Node> SearchTree<N> {
    fn new(root: N) -> SearchTree<N> {
        SearchTree {
            _root: root,
            children: HashMap::new(),
            scores: HashMap::new(),
            parents: HashMap::new(),
        }
    }

    fn children(&self, pos: &N) -> Option<&Vec<N>> {
        self.children.get(pos)
    }

    fn add_children(&mut self, pos: &N, children: Vec<N>) {
        for child in &children {
            self.parents.insert(child.clone(), pos.clone());
        }
        self.children.insert(pos.clone(), children);
    }

    fn random_child(&self, pos: &N) -> Option<N> {
        let children = self.children(pos)?;
        let mut rng = rand::thread_rng();
        Some(children[rng.gen_range(0..children.len())].clone())
    }

    fn best_child(&self, pos: &N) -> Option<N> {
        let children = self.children(pos)?;
        children
            .iter()
            .max_by_key(|child| (self.score(child).as_f32() * 1000.0) as i32)
            .map(|child| child.clone())
    }

    fn random_next_state(&self, pos: &N) -> Option<N> {
        let possible_states = pos.possible_next_states();
        if possible_states.is_empty() {
            return None;
        }
        let mut rng = rand::thread_rng();
        Some(possible_states[rng.gen_range(0..possible_states.len())].clone())
    }

    fn score(&self, pos: &N) -> Score {
        self.scores.get(pos).copied().unwrap_or(Score::zero())
    }

    fn add_visit(&mut self, pos: &N, points: f32) {
        let mut current_pos = Some(pos);
        while let Some(pos) = current_pos {
            let entry = self.scores.entry(pos.clone()).or_insert(Score::zero());
            entry.points += points;
            entry.visits += 1;
            current_pos = self.parents.get(pos);
        }
    }

    fn find_most_valuable_leaf<'a>(&'a self, pos: &'a N) -> &'a N {
        let mut current_pos = pos;
        let mut rng = rand::thread_rng();
        loop {
            match self.children(current_pos) {
                Some(children) => {
                    if children.is_empty() {
                        return current_pos;
                    } else {
                        current_pos = &children[rng.gen_range(0..children.len())];
                    }
                }
                None => return current_pos,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum TTTPlayer {
    X,
    O,
}

impl TTTPlayer {
    fn other(&self) -> TTTPlayer {
        match self {
            TTTPlayer::X => TTTPlayer::O,
            TTTPlayer::O => TTTPlayer::X,
        }
    }
}

#[derive(PartialEq, Debug, Eq, Hash)]
enum TTTResult {
    Draw,
    Win(TTTPlayer),
}

#[derive(Clone, Eq, Hash, PartialEq)]
/// A Tic Tac Toe game state
struct TTTPos {
    board: [Option<TTTPlayer>; 9],
    whose_turn: TTTPlayer,
}

impl TTTPos {
    fn new() -> TTTPos {
        TTTPos {
            board: [None; 9],
            whose_turn: TTTPlayer::X,
        }
    }

    fn with_index(&self, idx: usize) -> Result<TTTPos, ()> {
        if idx >= 9 {
            return Err(());
        }
        let mut pos = self.board.clone();
        pos[idx] = Some(self.whose_turn);
        Ok(TTTPos {
            board: pos,
            whose_turn: self.whose_turn.other(),
        })
    }

    fn is_terminal(&self) -> Option<TTTResult> {
        if self
            .board
            .iter()
            .take(3)
            .all(|&player| player.is_some() && player == self.board[0])
        {
            return Some(TTTResult::Win(self.board[0].unwrap()));
        } else if self
            .board
            .iter()
            .skip(3)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[3])
        {
            return Some(TTTResult::Win(self.board[3].unwrap()));
        } else if self
            .board
            .iter()
            .skip(6)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[6])
        {
            return Some(TTTResult::Win(self.board[6].unwrap()));
        } else if self
            .board
            .iter()
            .step_by(3)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[0])
        {
            return Some(TTTResult::Win(self.board[0].unwrap()));
        } else if self
            .board
            .iter()
            .skip(1)
            .step_by(3)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[1])
        {
            return Some(TTTResult::Win(self.board[1].unwrap()));
        } else if self
            .board
            .iter()
            .skip(2)
            .step_by(3)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[2])
        {
            return Some(TTTResult::Win(self.board[2].unwrap()));
        } else if self
            .board
            .iter()
            .step_by(4)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[0])
        {
            return Some(TTTResult::Win(self.board[0].unwrap()));
        } else if self
            .board
            .iter()
            .skip(2)
            .step_by(2)
            .take(3)
            .all(|&player| player.is_some() && player == self.board[2])
        {
            return Some(TTTResult::Win(self.board[2].unwrap()));
        }
        if self.board.iter().all(|&player| player.is_some()) {
            return Some(TTTResult::Draw);
        }
        None
    }
}

impl Debug for TTTPos {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, &player) in self.board.iter().enumerate() {
            if i % 3 == 0 {
                writeln!(f)?;
            }
            match player {
                Some(TTTPlayer::X) => write!(f, "X")?,
                Some(TTTPlayer::O) => write!(f, "O")?,
                None => write!(f, ".")?,
            }
        }
        Ok(())
    }
}

impl Node for TTTPos {
    fn possible_next_states(&self) -> Vec<TTTPos> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, &player)| {
                if player.is_none() {
                    Some(self.with_index(i).unwrap())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl TTTPos {
        fn from_slice(v: &[Option<TTTPlayer>], whose_turn: TTTPlayer) -> TTTPos {
            let mut pos = [None; 9];
            for (i, player) in v.iter().enumerate() {
                pos[i] = *player;
            }
            TTTPos {
                board: pos,
                whose_turn,
            }
        }
    }

    #[test]
    fn empty_tree_has_no_expanded_children() {
        let root_pos = TTTPos::new();
        let tree = SearchTree::new(root_pos.clone());
        assert!(tree.children(&root_pos).is_none());
    }

    #[test]
    fn can_add_children_to_tree() {
        let root_pos = TTTPos::new();
        let mut tree = SearchTree::new(root_pos.clone());
        let new_children = root_pos.possible_next_states();
        tree.add_children(&root_pos, new_children.clone());
        assert_eq!(tree.children(&root_pos).unwrap().len(), new_children.len());
    }

    #[test]
    fn all_moves_are_possible() {
        let pos = TTTPos::new();
        let states = pos.possible_next_states();
        assert_eq!(states.len(), 9);
        for state in states {
            assert_eq!(state.board.iter().filter(|&&p| p.is_some()).count(), 1);
        }
    }

    #[test]
    fn one_possible_move() {
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                None,
                Some(TTTPlayer::O),
            ],
            TTTPlayer::X,
        );
        let states = pos.possible_next_states();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].board.iter().filter(|&&p| p.is_some()).count(), 9)
    }

    #[test]
    fn find_most_valuable_leaf_in_root_only_tree() {
        let root_pos = TTTPos::new();
        let tree = SearchTree::new(root_pos.clone());
        let leaf = tree.find_most_valuable_leaf(&root_pos);
        assert_eq!(leaf, &root_pos);
    }

    #[test]
    fn find_most_valuable_leaf_in_tree_with_children() {
        let root_pos = TTTPos::new();
        let mut tree = SearchTree::new(root_pos.clone());
        let new_children = root_pos.possible_next_states();
        tree.add_children(&root_pos, new_children.clone());
        let leaf = tree.find_most_valuable_leaf(&root_pos);
        assert!(new_children.contains(leaf));
    }

    #[test]
    fn first_row_complete_is_terminal() {
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::X),
                Some(TTTPlayer::X),
                Some(TTTPlayer::X),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            TTTPlayer::O,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::X)));
    }

    #[test]
    fn first_col_complete_is_terminal() {
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::X),
                None,
                None,
                Some(TTTPlayer::X),
                None,
                None,
                Some(TTTPlayer::X),
                None,
                None,
            ],
            TTTPlayer::O,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::X)));
    }

    #[test]
    fn diagonal_complete_is_terminal() {
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::O),
                None,
                None,
                None,
                Some(TTTPlayer::O),
                None,
                None,
                None,
                Some(TTTPlayer::O),
            ],
            TTTPlayer::X,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::O)));
    }

    #[test]
    fn anti_diagonal_complete_is_terminal() {
        let pos = TTTPos::from_slice(
            &[
                None,
                None,
                Some(TTTPlayer::O),
                None,
                Some(TTTPlayer::O),
                None,
                Some(TTTPlayer::O),
                None,
                None,
            ],
            TTTPlayer::X,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::O)));
    }

    #[test]
    fn player_o_wins() {
        // OX.
        // XOX
        // .XO
        let pos = TTTPos::from_slice(
            &[
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                None,
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
                Some(TTTPlayer::X),
                None,
                Some(TTTPlayer::X),
                Some(TTTPlayer::O),
            ],
            TTTPlayer::X,
        );
        assert_eq!(pos.is_terminal(), Some(TTTResult::Win(TTTPlayer::O)));
    }

    #[test]
    fn score_of_root_is_zero() {
        let root_pos = TTTPos::new();
        let tree = SearchTree::new(root_pos.clone());
        let score = tree.score(&root_pos);
        assert_eq!(score.as_f32(), 0.0);
    }

    #[test]
    fn can_set_score_for_node() {
        let root_pos = TTTPos::new();
        let mut tree = SearchTree::new(root_pos.clone());
        tree.add_visit(&root_pos, 0.0);
        let score = tree.score(&root_pos);
        assert_eq!(score.as_f32(), 0.0);
    }

    #[test]
    fn can_set_successful_score_for_node() {
        let root_pos = TTTPos::new();
        let mut tree = SearchTree::new(root_pos.clone());
        tree.add_visit(&root_pos, 1.0);
        let score = tree.score(&root_pos);
        assert_eq!(score.as_f32(), 1.0);
    }
}
