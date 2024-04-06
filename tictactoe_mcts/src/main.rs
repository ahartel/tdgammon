use std::collections::HashMap;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;

use rand::Rng;

fn main() {}

trait Node: Eq + Hash + Clone {
    fn possible_next_states(&self, whose_turn: Player) -> Vec<Self>;
}

#[derive(Debug, Copy, Clone)]
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
    root: N,
    children: HashMap<N, Vec<N>>,
    scores: HashMap<N, Score>,
}

impl<N: Node> SearchTree<N> {
    fn new(root: N) -> SearchTree<N> {
        SearchTree {
            root,
            children: HashMap::new(),
            scores: HashMap::new(),
        }
    }

    fn inner(&self) -> &N {
        &self.root
    }

    fn children(&self, pos: &N) -> Option<&Vec<N>> {
        self.children.get(pos)
    }

    fn add_children(&mut self, pos: &N, children: Vec<N>) {
        self.children.insert(pos.clone(), children);
    }

    fn random_next_state(&self, pos: &N, whose_turn: Player) -> Option<N> {
        let possible_states = pos.possible_next_states(whose_turn);
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
        let mut entry = self.scores.entry(pos.clone()).or_insert(Score::zero());
        entry.points += points;
        entry.visits += 1;
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
enum Player {
    X,
    O,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct TTTPos([Option<Player>; 9]);

impl TTTPos {
    fn new() -> TTTPos {
        TTTPos([None; 9])
    }

    fn from_index_and_player(idx: usize, player: Player) -> Result<TTTPos, ()> {
        if idx >= 9 {
            return Err(());
        }
        let mut pos = [None; 9];
        pos[idx] = Some(player);
        Ok(TTTPos(pos))
    }

    fn with_index_and_player(&self, idx: usize, player: Player) -> Result<TTTPos, ()> {
        if idx >= 9 {
            return Err(());
        }
        let mut pos = self.0.clone();
        pos[idx] = Some(player);
        Ok(TTTPos(pos))
    }

    fn is_terminal(&self) -> bool {
        self.0
            .iter()
            .take(3)
            .all(|&player| player.is_some() && player == self.0[0])
            || self
                .0
                .iter()
                .skip(3)
                .take(3)
                .all(|&player| player.is_some() && player == self.0[3])
            || self
                .0
                .iter()
                .skip(6)
                .take(3)
                .all(|&player| player.is_some() && player == self.0[6])
            || self
                .0
                .iter()
                .step_by(3)
                .take(3)
                .all(|&player| player.is_some() && player == self.0[0])
            || self
                .0
                .iter()
                .skip(1)
                .step_by(3)
                .take(3)
                .all(|&player| player.is_some() && player == self.0[1])
            || self
                .0
                .iter()
                .skip(2)
                .step_by(3)
                .take(3)
                .all(|&player| player.is_some() && player == self.0[2])
            || self
                .0
                .iter()
                .step_by(4)
                .take(3)
                .all(|&player| player.is_some() && player == self.0[0])
            || self
                .0
                .iter()
                .skip(2)
                .step_by(2)
                .take(3)
                .all(|&player| player.is_some() && player == self.0[2])
    }
}

impl Debug for TTTPos {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, &player) in self.0.iter().enumerate() {
            if i % 3 == 0 {
                writeln!(f)?;
            }
            match player {
                Some(Player::X) => write!(f, "X")?,
                Some(Player::O) => write!(f, "O")?,
                None => write!(f, ".")?,
            }
        }
        Ok(())
    }
}

impl Node for TTTPos {
    fn possible_next_states(&self, whose_turn: Player) -> Vec<TTTPos> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, &player)| {
                if player.is_none() {
                    Some(self.with_index_and_player(i, whose_turn).unwrap())
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
        fn from_slice(v: &[Option<Player>]) -> TTTPos {
            let mut pos = [None; 9];
            for (i, player) in v.iter().enumerate() {
                pos[i] = *player;
            }
            TTTPos(pos)
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
        let new_children = root_pos.possible_next_states(Player::O);
        tree.add_children(&root_pos, new_children.clone());
        assert_eq!(tree.children(&root_pos).unwrap().len(), new_children.len());
    }

    #[test]
    fn all_moves_are_possible() {
        let pos = TTTPos::new();
        let states = pos.possible_next_states(Player::X);
        assert_eq!(states.len(), 9);
        for state in states {
            assert_eq!(state.0.iter().filter(|&&p| p.is_some()).count(), 1);
        }
    }

    #[test]
    fn one_possible_move() {
        let pos = TTTPos::from_slice(&[
            Some(Player::X),
            Some(Player::O),
            Some(Player::X),
            Some(Player::O),
            Some(Player::X),
            Some(Player::O),
            Some(Player::O),
            None,
            Some(Player::O),
        ]);
        let states = pos.possible_next_states(Player::X);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].0.iter().filter(|&&p| p.is_some()).count(), 9)
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
        let new_children = root_pos.possible_next_states(Player::O);
        tree.add_children(&root_pos, new_children.clone());
        let leaf = tree.find_most_valuable_leaf(&root_pos);
        assert!(new_children.contains(leaf));
    }

    #[test]
    fn first_row_complete_is_terminal() {
        let pos = TTTPos::from_slice(&[
            Some(Player::X),
            Some(Player::X),
            Some(Player::X),
            None,
            None,
            None,
            None,
            None,
            None,
        ]);
        assert!(pos.is_terminal());
    }

    #[test]
    fn first_col_complete_is_terminal() {
        let pos = TTTPos::from_slice(&[
            Some(Player::X),
            None,
            None,
            Some(Player::X),
            None,
            None,
            Some(Player::X),
            None,
            None,
        ]);
        assert!(pos.is_terminal());
    }

    #[test]
    fn diagonal_complete_is_terminal() {
        let pos = TTTPos::from_slice(&[
            Some(Player::O),
            None,
            None,
            None,
            Some(Player::O),
            None,
            None,
            None,
            Some(Player::O),
        ]);
        assert!(pos.is_terminal());
    }

    #[test]
    fn anti_diagonal_complete_is_terminal() {
        let pos = TTTPos::from_slice(&[
            None,
            None,
            Some(Player::O),
            None,
            Some(Player::O),
            None,
            Some(Player::O),
            None,
            None,
        ]);
        assert!(pos.is_terminal());
    }

    #[test]
    fn can_simulate_game() {
        let root_pos = TTTPos::new();
        let tree = SearchTree::new(root_pos.clone());
        let mut current_pos = root_pos;
        let mut whose_turn = Player::X;
        while let Some(child) = tree.random_next_state(&current_pos, whose_turn) {
            dbg!(&child);
            if child.is_terminal() {
                break;
            }
            current_pos = child;
            whose_turn = match whose_turn {
                Player::X => Player::O,
                Player::O => Player::X,
            };
        }
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
