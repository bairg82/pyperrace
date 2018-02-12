from collections import deque
import random
import numpy as np
""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123, save_dir = './experience/'):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.save_dir = save_dir
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size, get_all = False):
        batch = []

        if get_all:
            batch = self.buffer
        else:
            if self.count < batch_size:
                batch = random.sample(self.buffer, self.count)
            else:
                batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def save(self, policy = 'all', number = 1):
        # based on this:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez.html#numpy.savez

        # open file

        # save all
        if policy == 'all':
            batch = self.buffer

        # save only best, number
        # if policy = 'best'
        #TODO: impelemnting best selection

        # select items
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        # save to file
        np.savez((self.save_dir+'/experience.npz'), s=s_batch, a=a_batch, r=r_batch, t=t_batch, s2=s2_batch,)

    def load(self, file='./experience/tf_ddpg/experience.npz'):
        # based on this:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez.html#numpy.savez

        added = 0

        # loading experience from file
        try:
            npzfile = np.load(file)
        except:
            # do nothing
            pass

        if npzfile != None:
            s = npzfile['s']
            a = npzfile['a']
            r = npzfile['r']
            t = npzfile['t']
            s2 = npzfile['s2']

            # this might be slow
            # adding items by one
            for i in range(len(s)):
                self.add(s[i], a[i], r[i], t[i], s2[i])
                added += 1

        return added

    def clear(self):
        self.buffer.clear()
        self.count = 0