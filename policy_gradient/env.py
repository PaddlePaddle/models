import time
import sys
import numpy as np


class Env():
    def __init__(self, stage_len, interval):
        self.stage_len = stage_len
        self.end = self.stage_len - 1
        self.position = 0
        self.interval = interval
        self.step = 0
        self.epoch = -1
        self.render = False

    def reset(self):
        self.end = self.stage_len - 1
        self.position = 0
        self.epoch += 1
        self.step = 0
        if self.render:
            self.draw(True)

    def status(self):
        s = np.zeros([self.stage_len]).astype("float32")
        s[self.position] = 1
        return s

    def move(self, action):
        self.step += 1
        reward = 0.0
        done = False
        if action == 0:
            self.position = max(0, self.position - 1)
        else:
            self.position = min(self.end, self.position + 1)
        if self.render:
            self.draw()
        if self.position == self.end:
            reward = 1.0
            done = True
        return reward, done, self.status()

    def draw(self, new_line=False):
        if new_line:
            print ""
        else:
            print "\r",
        for i in range(self.stage_len):
            if i == self.position:
                sys.stdout.write("O")
            else:
                sys.stdout.write("-")
        sys.stdout.write("    epoch: %d; steps: %d" % (self.epoch, self.step))
        sys.stdout.flush()
        time.sleep(self.interval)
