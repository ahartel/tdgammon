import sys

import numpy as np
import numpy.random
import pygame
import random

from MLP import MLP
from grid import Grid


class GridGame:
    def __init__(self, width, height):
        self.mlp = MLP(weights=[np.random.uniform(0.0, 0.1, (1, width * height))],
                       biases=[np.zeros(1)],
                       learning_rate=0.01)
        self.grid = Grid(width, height)
        self.step = 1
        self.width = width
        self.height = height
        self.pos = self.grid.get_start_position()
        self.path = [(self.pos, False)]
        self.eligibility_trace = np.zeros((1, width * height))
        self.last_pos_value = 0
        self.__lambda = 0.5

    def get_weights(self):
        return self.mlp.get_weights()[0]

    def get_eligibility_trace(self):
        return self.eligibility_trace

    def increment_step(self):
        if self.step < len(self.path):
            self.step += 1
            self.pos, random_step = self.path[self.step-1]
            print("Step redone, {}, {}".format(self.step, len(self.path)))
            return self.pos, random_step
        elif self.pos != self.grid.get_goal_position():
            new_positions = [
                (self.pos[0] + 0, self.pos[1] + 1),
                (self.pos[0] + 0, self.pos[1] - 1),
                (self.pos[0] - 1, self.pos[1] + 0),
                (self.pos[0] + 1, self.pos[1] + 0),
            ]
            rand = random.random()
            do_random_move = rand < 0.5
            next_post = None
            if not do_random_move:
                predicted_rewards = [0.0 for _ in range(4)]
                for idx, new_pos in enumerate(new_positions):
                    if 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height:
                        inputs = np.zeros(self.width * self.height)
                        inputs[new_pos[0] + self.width * new_pos[1]] = 1.0
                        predicted_rewards[idx] = self.mlp.run_input(inputs, save_inputs_and_activations=False)
                max_idx = np.argmax(predicted_rewards)
                next_pos = new_positions[max_idx]
            else:
                next_pos = (-1, -1)
                while not (0 <= next_pos[0] < self.width and 0 <= next_pos[1] < self.height):
                    next_pos = random.choice(new_positions)
            assert (0 <= next_pos[0] < self.width and 0 <= next_pos[1] < self.height)
            # calculate the gradient based on last state
            current_gradient = self.mlp.gradient()[0][0]
            # run the input for the gradient of the next run
            # and for the value estimate
            inputs = np.zeros(self.width * self.height)
            inputs[next_pos[0] + self.width * next_pos[1]] = 1.0
            this_pos_value = self.mlp.run_input(inputs, save_inputs_and_activations=True)
            self.eligibility_trace = self.__lambda * self.eligibility_trace + current_gradient
            delta = self.grid.give_reward(next_pos) + this_pos_value - self.last_pos_value
            self.mlp.add_to_weights([self.eligibility_trace * delta])
            self.last_pos_value = this_pos_value
            self.pos = next_pos
            self.path.append((next_pos, do_random_move))
            self.step += 1
            print("Step done, {}, {}".format(self.step, len(self.path)))
            return next_pos, do_random_move
        else:
            return self.pos, False

    def decrement_step(self):
        if not self.step > 1:
            return self.pos, False
        self.step -= 1
        self.pos, random_move = self.path[self.step-1]
        print("Step undone, {}".format(self.step))
        return self.pos, random_move

    def get_current_position(self):
        return self.pos


def draw_weights_on_grid(screen, weights, width, height, field_size, font):
    for x in range(width):
        for y in range(height):
            pos = x + y * width
            rect = pygame.Rect(x * field_size,
                               y * field_size,
                               field_size/2,
                               field_size)
            alpha = int(255 * weights[0][pos] * 10.0)
            # print(alpha)
            pygame.draw.rect(screen, (0, 0, alpha), rect, border_radius=10)
            text_surface = font.render("({}, {})".format(x, y), False, (255, 255, 255))
            screen.blit(text_surface, (x * field_size, y * field_size))
            text_surface = font.render("{:.5f}".format(weights[0][pos]), False, (255, 255, 255))
            screen.blit(text_surface, (x * field_size, (y + 0.5) * field_size))


def draw_eligibility_on_grid(screen, trace, width, height, field_size, font):
    for x in range(width):
        for y in range(height):
            pos = x + y * width
            rect = pygame.Rect((x + 0.5) * field_size,
                               y * field_size,
                               field_size/2,
                               field_size)
            alpha = int(255 * trace[0][pos] * 1.0)
            # print(alpha)
            pygame.draw.rect(screen, (0, alpha, alpha), rect, border_radius=10)
            text_surface = font.render("{:.5f}".format(trace[0][pos]), False, (255, 255, 255))
            screen.blit(text_surface, ((x + 0.5) * field_size, (y + 0.5) * field_size))


def draw_pos_on_grid(screen, pos, random_move, field_size, font):
    location_x = (pos[0] + 0.25) * field_size
    location_y = (pos[1] + 0.25) * field_size
    rect = pygame.Rect(location_x, location_y, field_size/4, field_size/4)
    color = (255, 0, 0) if random_move else (0, 255, 0)
    pygame.draw.rect(screen, color, rect, border_radius=10)
    text_surface = font.render("R" if random_move else "G", False, (255, 255, 255))
    screen.blit(text_surface, (location_x, location_y))


def main():
    num_columns = 5
    num_rows = 5
    field_size = 100
    random.seed(42)
    numpy.random.seed(42)
    pygame.init()
    screen = pygame.display.set_mode((num_columns * field_size, num_rows * field_size))
    clock = pygame.time.Clock()
    game = GridGame(num_columns, num_rows)
    weights = game.get_weights()
    pos = game.get_current_position()
    random_move = False
    pygame.font.init()  # you have to call this at the start,
    # if you want to use this module.
    font = pygame.font.SysFont('Comic Sans MS', 12)
    eligibility_trace = game.get_eligibility_trace()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    pos, random_move = game.decrement_step()
                if event.key == pygame.K_RIGHT:
                    pos, random_move = game.increment_step()
                    eligibility_trace = game.get_eligibility_trace()

        draw_eligibility_on_grid(screen, eligibility_trace, num_columns, num_rows, field_size, font)
        draw_weights_on_grid(screen, weights, num_columns, num_rows, field_size, font)
        draw_pos_on_grid(screen, pos, random_move, field_size, font)

        pygame.display.update()
        clock.tick(60)


if __name__ == '__main__':
    main()
