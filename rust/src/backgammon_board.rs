pub enum BoardError {
    NoStoneFound,
}

pub struct BackgammonBoard {
    white: [u8; 26],
    black: [u8; 26],
}

impl BackgammonBoard {
    pub fn new() -> Self {
        let mut initial_white = [0; 26];
        initial_white[0] = 2;
        initial_white[11] = 5;
        initial_white[16] = 3;
        initial_white[18] = 5;

        let mut initial_black = [0; 26];
        initial_black[5] = 5;
        initial_black[7] = 3;
        initial_black[12] = 5;
        initial_black[23] = 2;
        BackgammonBoard {
            white: initial_white,
            black: initial_black,
        }
    }

    pub fn print(&self) {
        let white_square: char = 248.into();
        let black_square: char = 111.into();
        for idx in 12..24 {
            print!("{idx:02} ");
        }
        println!("");
        for idx in 12..24 {
            if self.white[idx] > 0 {
                print!(" {} ", white_square);
            } else if self.black[idx] > 0 {
                print!(" {} ", black_square);
            } else {
                print!(" . ");
            }
        }
        println!("");
        for idx in 12..24 {
            if self.white[idx] > 0 {
                print!(" {} ", self.white[idx]);
            } else if self.black[idx] > 0 {
                print!(" {} ", self.black[idx]);
            } else {
                print!(" . ");
            }
        }
        println!("");
        println!("");
        for idx in (0..12).rev() {
            if self.white[idx] > 0 {
                print!(" {} ", self.white[idx]);
            } else if self.black[idx] > 0 {
                print!(" {} ", self.black[idx]);
            } else {
                print!(" . ");
            }
        }
        println!("");
        for idx in (0..12).rev() {
            if self.white[idx] > 0 {
                print!(" {} ", white_square);
            } else if self.black[idx] > 0 {
                print!(" {} ", black_square);
            } else {
                print!(" . ");
            }
        }
        println!("");
        for idx in (0..12).rev() {
            print!("{idx:02} ");
        }
        println!("");
    }

    pub fn apply_move(
        &mut self,
        _dice: (u8, u8),
        from: usize,
        to: usize,
    ) -> Result<(), BoardError> {
        if self.white[from] > 0 && self.black[to] <= 1 {
            self.white[from] -= 1;
            self.white[to] += 1;
            Ok(())
        } else if self.black[from] > 0 && self.white[to] <= 1 {
            self.black[from] -= 1;
            self.black[to] += 1;
            Ok(())
        } else {
            Err(BoardError::NoStoneFound)
        }
    }
}
