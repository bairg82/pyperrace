import numpy as np
import matplotlib.pyplot as plt

from environment import PaperRaceEnv
from replay_buffer import ReplayBuffer

import random as rnd

import matplotlib.image as mpimg

trk_col = np.array([99, 99, 99]) # pálya színe (szürke)

#sections = np.array([[350,  60, 350, 100],
#                     [560, 130, 535, 165]])
#                     [348, 354, 348, 326]])
#                     [ 35, 200,  70, 200],
#                     [250,  60, 250, 100]])

#sections = np.array([[273, 125, 273, 64],
#                     [327, 125, 327, 65],
#                     [394, 157, 440, 102]])
#                     [348, 354, 348, 326]])
#                     [ 35, 200,  70, 200],
#                     [250,  60, 250, 100]])

#sections = np.array([[394, 157, 440, 102],
#                     [331, 212, 331, 267]])

"""
sections = np.array([[273, 125, 273, 64],
                     [333, 125, 333, 64],
                     [394, 157, 440, 102],
                     [370, 195, 430, 240],
                     [331, 212, 331, 267],
                     [220, 300, 280, 300],
                     [240, 400, 330, 380]])
# [190, 125, 190, 64]])
"""
#sections = np.array([[273, 125, 273, 64],  # [333, 125, 333, 64],[394, 157, 440, 102],[240, 400, 330, 380]])
#                     [80, 250, 180, 250]])
"""
# palya5.bmp-hez:
sections = np.array([[670, 310, 670, 130],  # [333, 125, 333, 64],[394, 157, 440, 102],
                     [1250, 680, 1250, 550]])
"""
# h1.bmp-hez:
sections = np.array([[200, 220, 200, 50],  # [333, 125, 333, 64],[394, 157, 440, 102],
                     [400, 1250, 500, 1400]])

# start_line = np.array([32, 393, 32, 425]) # sigmoid alakú pálya

env = PaperRaceEnv('h1.bmp', trk_col, 'GG1.bmp', sections, random_init=False) # paperrace környezet létrehozása
mem_size = 100 # a memória mérete, amiből a batch-be válogatunk
batch_size = 10 # batch mérete, ami a tanítási adatokat tartalmazza
episodes = 1000 # hányszor fusson a tanítás
random_seed = 123

ref_spd = 20 #referencia sebesseg, rewardhoz, [pixel/lepes]

s_dim = 4 #állapottér dimenziója
a_dim = 1 #action space dimenzója

replay_buffer = ReplayBuffer(int(mem_size), int(random_seed))

draw = True

rndlep = int(input('Random input? 1=yes'))
if rndlep == 1:
    random = True
else:
    random = False

env.gg_pic = mpimg.imread('GG_rally.bmp')

for ep in range(episodes):
    env.reset()
    print("================EP.: ", ep) # epizód számának kiírása
    if draw: # ha rajzolunk
        plt.clf()
        env.draw_track()
    v = np.array(env.starting_spd)  # az elején a sebesség a startvonalra meroleges
    # ezt könnyen megváltoztatja, tulajdonképen csak arra jó, hogy nem 0
    pos = np.array(env.starting_pos)  # kezdőpozíció beállítása
    #print("envtest start pos, v:", pos, v)
    reward = 0
    epreward = 0
    ref_dist = 0
    end = False
    color = (0, 0, 1)
    step = 0

    while not end:
        step = step + 1

        if random:
            #action = int(np.random.randint(-180, 180, size=1))
            lepestol = rnd.uniform(0, env.ref_actions.size)
            #print(lepestol, env.ref_actions.size)
            if (lepestol < step) and (step < env.ref_actions.size):
                action = int(np.random.normal(env.ref_actions[step], 30, size=1))
                print("range  rand action: ", action, "-------------")
            else:
                action = env.ref_actions[step-1]# int(np.random.randint(-180, 180, size=1))
                print("ref action: ", action, "-------------")
        else:
            action = int(input('Give inut (-180..180 number)'))
            print("manual action: ", action, "-------------")

        gg_action = env.gg_action(action)  # action-höz tartozó vektor lekérése
        v_new, pos_new, reward, end = env.step(gg_action, v, pos, draw, color)
        t_diff = env.get_time_diff(pos, pos_new, reward, end)
        s = [v[0], v[1], pos[0], pos[1]]
        s2 = [v_new[0], v_new[1], pos_new[0], pos_new[1]]
        a = action
        # a ref tavolsag legyen hogy adott ref_spd sebesseggel adott reward eltelt ido alatt, (reward most nekunk
        # tulkepp az eltelt idot adja, negativban) meddig kellett volna eljutni
        # ref_dist = (-ref_spd * reward) + ref_dist
        # print("ref_dist: ", ref_dist)
        # print("curr_dist: ", curr_dist)
        # r = curr_dist - ref_dist
        r = t_diff
        terminal = end

        #print(s)
        #print(s2)
        epreward = epreward + r
        print("reward: ", r)
        #print("Section: ", section_nr)

        replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
                          terminal, np.reshape(s2, (s_dim,)))

        if draw:
            plt.pause(0.001)
            plt.draw()

        v = v_new
        pos = pos_new

    if replay_buffer.size() > int(batch_size):
        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(batch_size))
        #TODO: valami olvashatóbb formátumban kiiratni. pl. táblázat
        #print('batch: ', s_batch, a_batch, r_batch, t_batch, s2_batch)

    print("Eprew.: ", epreward)


