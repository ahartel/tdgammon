pub mod searchtree;
pub mod tictactoe;
pub mod connectfour;

#[cfg(test)]
mod tests {
    use crate::{
        searchtree::{Node, SearchTree},
        tictactoe::TTTPos,
    };

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
