from brain import PolicyGradient
from env import Env
import numpy as np

n_actions = 2
interval = 0.01
stage_len = 10
epoches = 10000

if __name__ == "__main__":

    brain = PolicyGradient(n_actions, stage_len)
    e = Env(stage_len, interval)
    brain.build_net()
    done = False

    for epoch in range(epoches):
        if (epoch % 500 == 1) or epoch < 5 or epoch > 3000:
            e.render = True
        else:
            e.render = False
        e.reset()
        while not done:
            s = e.status()
            action = brain.choose_action(s)
            r, done, _ = e.move(action)
            brain.store_transition(s, action, r)
        done = False
        brain.learn()
