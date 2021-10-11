

class Grid:
    def __init__(self, width, height):
        self.__width = width
        self.__height = height
        self.__start_pos = (0, 0)
        self.__goal_pos = (width-1, height-1)

    def give_reward(self, location):
        assert((0, 0) <= location < (self.__width, self.__height))
        if location == self.__goal_pos:
            return 1.0
        else:
            return 0.0

    def get_start_position(self):
        return self.__start_pos

    def get_goal_position(self):
        return self.__goal_pos


def main():
    width = 10
    height = 10
    grid = Grid(width, height)
    for x in range(width):
        for y in range(height):
            reward = grid.give_reward((x, y))
            if (x, y) == (width-1, height-1):
                assert(reward == 1.0)
            else:
                assert(reward == 0.0)


if __name__ == '__main__':
    main()
