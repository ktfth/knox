import os
import gym as g
import random
from MAMEToolkit.sf_environment import Environment

import tensorflow as tf

def main(argv):
    roms_path = "roms/"
    env = Environment("env1", roms_path)

    env.start()
    while True:
        move_action = random.randint(0, 8)
        attack_action = random.randint(0, 9)
        frames, reward, round_done, stage_done, game_done = env.step(move_action, attack_action)
        if game_done:
            env.new_game()
        elif stage_done:
            env.next_stage()
        elif round_done:
            env.next_round()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
