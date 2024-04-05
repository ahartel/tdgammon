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

    fn children(&self, pos: &N) -> Option<Vec<N>> {
        self.children.get(pos).cloned()
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
        vec![
            TTTPos::from_index_and_player(0, whose_turn).unwrap(),
            TTTPos::from_index_and_player(1, whose_turn).unwrap(),
            TTTPos::from_index_and_player(2, whose_turn).unwrap(),
            TTTPos::from_index_and_player(3, whose_turn).unwrap(),
            TTTPos::from_index_and_player(4, whose_turn).unwrap(),
            TTTPos::from_index_and_player(5, whose_turn).unwrap(),
            TTTPos::from_index_and_player(6, whose_turn).unwrap(),
            TTTPos::from_index_and_player(7, whose_turn).unwrap(),
            TTTPos::from_index_and_player(8, whose_turn).unwrap(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
