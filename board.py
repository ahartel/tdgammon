import sys


class Board:
    NUM_FIELDS = 24

    def __init__(self):
        # bar is field 25
        # off is field 0
        self.whites = [0 for _ in range(self.NUM_FIELDS + 2)]
        self.blacks = [0 for _ in range(self.NUM_FIELDS + 2)]
        self.init_with_starting_position(self.whites)
        self.init_with_starting_position(self.blacks)

    @staticmethod
    def init_with_starting_position(fields):
        fields[24] = 2
        fields[13] = 5
        fields[8] = 3
        fields[6] = 5

    def print(self):
        self.print_number_row(13, 19, 25, 1)
        for row in range(5):
            self.print_quadrant_row(row, 13, 19, 1)
            sys.stdout.write("|")
            self.print_quadrant_row(row, 19, 25, 1)
            sys.stdout.write("\n")
        print("=" * 26)
        for row in range(5):
            self.print_quadrant_row(row, 12, 6, -1)
            sys.stdout.write("|")
            self.print_quadrant_row(row, 6, 0, -1)
            sys.stdout.write("\n")
        self.print_number_row(12, 6, 0, -1)

    @staticmethod
    def print_number_row(start, mid, end, step):
        for i in range(start, mid, step):
            sys.stdout.write("{:02d}".format(i))
        sys.stdout.write(" ")
        for i in range(mid, end, step):
            sys.stdout.write("{:02d}".format(i))
        sys.stdout.write("\n")

    def print_quadrant_row(self, row, start, stop, step):
        for field in range(start, stop, step):
            if self.whites[field] > row:
                # sys.stdout.write("⬤")
                sys.stdout.write(" G")
            elif self.blacks[25-field] > row:
                # sys.stdout.write("⚪")
                sys.stdout.write(" O")
            else:
                sys.stdout.write(" _")

    @staticmethod
    def pos_bits(pos):
        bits = [0, 0, 0, 0]

        if pos >= 1:
            bits[0] = 1
        if pos >= 2:
            bits[1] = 1
        if pos >= 3:
            bits[2] = 1
        if pos > 3:
            bits[3] = (float(pos) - 3.0) / 2.0
        return bits

    def prepare_inputs(self):
        inputs = []
        for white in self.whites[1:25]:
            inputs.extend(self.pos_bits(white))
        for black in self.blacks[1:25]:
            inputs.extend(self.pos_bits(white))
        inputs.append(float(self.whites[25]) / 2.0)
        inputs.append(float(self.blacks[25]) / 2.0)
        inputs.append(float(self.whites[0]) / 15.0)
        inputs.append(float(self.blacks[0]) / 15.0)
        return inputs
