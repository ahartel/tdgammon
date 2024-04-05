use std::collections::HashMap;
use std::hash::Hash;

fn main() {}

#[derive(Debug)]
struct SearchTree<N: Eq + Hash> {
    root: N,
    children: HashMap<N, Vec<N>>,
}

impl<N: Eq + Hash + Clone> SearchTree<N> {
    fn new(root: N) -> SearchTree<N> {
        SearchTree {
            root,
            children: HashMap::new(),
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
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum Player {
    X,
    O,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
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

    fn possible_moves(&self, whose_turn: Player) -> Vec<TTTPos> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, &player)| {
                if player.is_none() {
                    Some(TTTPos::from_index_and_player(i, whose_turn).unwrap())
                } else {
                    None
                }
            })
            .collect()
    }
}

fn find_most_valuable_leaf<'a, N: Eq + Hash + Clone>(tree: &'a SearchTree<N>, pos: &'a N) -> &'a N {
    let mut current_pos = pos;
    loop {
        match tree.children(current_pos) {
            Some(children) => {
                if children.is_empty() {
                    return current_pos;
                } else {
                    current_pos = &children[0];
                }
            }
            None => return current_pos,
        }
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
        let new_children = root_pos.possible_moves(Player::O);
        tree.add_children(&root_pos, new_children.clone());
        assert_eq!(tree.children(&root_pos).unwrap().len(), new_children.len());
    }

    #[test]
    fn all_moves_are_possible() {
        let pos = TTTPos::new();
        let moves = pos.possible_moves(Player::X);
        assert_eq!(moves.len(), 9);
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
        let moves = pos.possible_moves(Player::X);
        assert_eq!(moves.len(), 1);
    }

    #[test]
    fn find_most_valuable_leaf_in_root_only_tree() {
        let root_pos = TTTPos::new();
        let tree = SearchTree::new(root_pos.clone());
        let leaf = find_most_valuable_leaf(&tree, &root_pos);
        assert_eq!(leaf, &root_pos);
    }

    #[test]
    fn find_most_valuable_leaf_in_tree_with_children() {
        let root_pos = TTTPos::new();
        let mut tree = SearchTree::new(root_pos.clone());
        let new_children = root_pos.possible_moves(Player::O);
        tree.add_children(&root_pos, new_children.clone());
        let leaf = find_most_valuable_leaf(&tree, &root_pos);
        assert_eq!(leaf, &new_children[0]);
    }
}
