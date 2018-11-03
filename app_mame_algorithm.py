import os
import gym as g
import random
from MAMEToolkit.sf_environment import Environment

import tensorflow as tf

import warnings as ignite ; ignite.simplefilter('ignore')

K = tf.keras.backend

from app_parser import parser

from policy_gradient_builder import PolicyGradientBuilder

from agent_proxy import AgentProxy

from dqn_flyweight import DQNFlyweight

from reinforcement_learning import ReinforcementLearning

def main(argv):
    roms_path = "roms/"
    env = Environment("env1", roms_path)

    policy_gradient = PolicyGradientBuilder(100800, 100800, False)

    rl = AgentProxy(env, 100800)
    dqn = DQNFlyweight(agent=rl)
    net = ReinforcementLearning(rl)

    env.start()
    while True:
        move_action = random.randint(0, 8)
        p_move_action = rl.action_space_down_sample(move_action)
        steps_move_action = net.steps_action(p_move_action)
        attack_action = random.randint(0, 9)
        p_attack_action = rl.action_space_down_sample(attack_action)
        steps_attack_action = net.steps_action(p_attack_action)
        #frames, reward, round_done, stage_done, game_done = env.step(move_action, attack_action)
        frames, reward, \
        round_done, stage_done, \
        game_done = policy_gradient.learn(steps_move_action, steps_attack_action)
        if game_done:
            env.new_game()
        elif stage_done:
            env.next_stage()
        elif round_done:
            env.next_round()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
