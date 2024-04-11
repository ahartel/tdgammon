use std::{collections::HashMap, hash::Hash};

use rand::Rng;

pub trait Node: Eq + Hash + Clone {
    fn possible_next_states(&self) -> Vec<Self>;
}

#[derive(Debug, Copy, Clone)]
/// How many times a node has been visited and how many points it has accumulated.
pub struct Score {
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

    pub fn as_f32(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        self.points / self.visits as f32
    }
}

#[derive(Debug)]
pub struct SearchTree<N: Node> {
    _root: N,
    children: HashMap<N, Vec<N>>,
    scores: HashMap<N, Score>,
    parents: HashMap<N, N>,
}

impl<N: Node> SearchTree<N> {
    pub fn new(root: N) -> SearchTree<N> {
        SearchTree {
            _root: root,
            children: HashMap::new(),
            scores: HashMap::new(),
            parents: HashMap::new(),
        }
    }

    pub fn children(&self, pos: &N) -> Option<&Vec<N>> {
        self.children.get(pos)
    }

    pub fn add_children(&mut self, pos: &N, children: Vec<N>) {
        for child in &children {
            self.parents.insert(child.clone(), pos.clone());
        }
        self.children.insert(pos.clone(), children);
    }

    pub fn random_child(&self, pos: &N) -> Option<N> {
        let children = self.children(pos)?;
        let mut rng = rand::thread_rng();
        Some(children[rng.gen_range(0..children.len())].clone())
    }

    pub fn best_child(&self, pos: &N) -> Option<N> {
        let children = self.children(pos)?;
        children
            .iter()
            .max_by_key(|child| (self.score(child).as_f32() * 1000.0) as i32)
            .map(|child| child.clone())
    }

    pub fn random_next_state(&self, pos: &N) -> Option<N> {
        let possible_states = pos.possible_next_states();
        if possible_states.is_empty() {
            return None;
        }
        let mut rng = rand::thread_rng();
        Some(possible_states[rng.gen_range(0..possible_states.len())].clone())
    }

    pub fn score(&self, pos: &N) -> Score {
        self.scores.get(pos).copied().unwrap_or(Score::zero())
    }

    pub fn add_visit(&mut self, pos: &N, points: f32) {
        let mut current_pos = Some(pos);
        while let Some(pos) = current_pos {
            let entry = self.scores.entry(pos.clone()).or_insert(Score::zero());
            entry.points += points;
            entry.visits += 1;
            current_pos = self.parents.get(pos);
        }
    }

    pub fn find_most_valuable_leaf<'a>(&'a self, pos: &'a N) -> &'a N {
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
