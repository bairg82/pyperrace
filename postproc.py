import numpy as np
import matplotlib.pyplot as plt

from environment import PaperRaceEnv

from collections import OrderedDict

import matplotlib.image as mpimg

trk_col = np.array([99, 99, 99]) # pálya színe (szürke)

# h1.bmp-hez:
sections = np.array([[200, 220, 200, 50],  # [333, 125, 333, 64],[394, 157, 440, 102],
                     [350, 1150, 400, 1400]])

env = PaperRaceEnv('h1.bmp', trk_col, 'GG1.bmp', sections, random_init=False) # paperrace környezet létrehozása
mem_size = 100 # a memória mérete, amiből a batch-be válogatunk
batch_size = 10 # batch mérete, ami a tanítási adatokat tartalmazza
episodes = 1000 # hányszor fusson a tanítás
random_seed = 123

ref_spd = 20 #referencia sebesseg, rewardhoz, [pixel/lepes]

s_dim = 4 #állapottér dimenziója
a_dim = 1 #action space dimenzója
"""
linestyles_odict = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
"""
linestyles_odict = OrderedDict(
    [('solid',               (0, ())),
     ('densely dotted',      (0, (1, 1))),

     ('densely dashed',      (0, (5, 1))),
     ('dashed',              (0, (5, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotted',          (0, (5, 3, 1, 3))),

     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('dotted', (0, (1, 5)))])

linestyles = list(linestyles_odict.items())

hum_act = (
    [70, -30, -70, -180, -180, -180, -110, -110, -110, -90, -110, -100, -90, -90, -80, -80, -80, -80, -40, -40, -30,
     -30, -20, -20, -20, -20, -10], \
    [90, -90, -50, 100, -180, -180, -180, -160, -150, -140, -110, -100, -90, -90, -80, -70, -70, -70, -60, -40, -20,
     -20, -20, -20, -10, -10, -10], \
    [-90, 90, 50, -100, -180, -180, -180, -160, -150, -140, -110, -100, -90, -90, -80, -70, -70, -70, -40, -30, -20,
     -20, -20, -20, -20, -20], \
    [90, -40, -10, -180, -180, -180, -175, -165, -150, -120, -100, -90, -70, -70, -70, -70, -70, -70, -70, -60, -50,
     -40, -40, -30, -30, -20], \
    [-180, -150, 150, 150, 90, -90, 0, -90, 130, -120, -120, -110, -100, -100, -100, -100, -100, -90, -80, -80, -60,
     -40, -40, -40, -20, -20, -20, -20, -20], \
    [-180, -150, -90, 90, 90, 120, 120, -130, -120, -120, -120, -100, -110, -100, -100, -100, -100, -90, -80, -70,
     -40, -40, -20, -20, -20, -20, -20, -20, -20], \
    [0, -180, -180, -170, 120, 120, -130, -130, -130, -100, 100, 100, 100, -110, -110, -110, -110, -110, -100, -100,
     100, 100, -100, -100, -120, -100, -100, 90, 80, -60, 60, -30, -30, -20, -30, -30, -40, -40, -50, -30],
    [100])

for ep in range(episodes):
    env.reset()
    print("================EP.: ", ep) # epizód számának kiírása

    plt.clf()
    env.draw_track()

    reward = 0
    epreward = 0
    ref_dist = 0
    end = False
    color = (0, 0, 1)

    env.gg_pic = mpimg.imread('GG1_gokart.bmp')
    # az emberi lepesek kirajzolgatása
    for ii in range(len(hum_act)):
        print(ii)
        v = np.array(env.starting_spd)
        pos = np.array(env.starting_pos)

        cur_ep_actions = np.array(hum_act[ii])
        for jj in range(cur_ep_actions.size):
            action = cur_ep_actions[jj]
            gg_action = env.gg_action(action)  # action-höz tartozó vektor lekérése
            v_new, pos_new, reward, end, section_nr = env.step(gg_action, v, pos, False, color)
            t_diff = env.get_time_diff(pos, pos_new, reward, end)
            s = [v[0], v[1], pos[0], pos[1]]
            s2 = [v_new[0], v_new[1], pos_new[0], pos_new[1]]
            a = action
            r = t_diff
            terminal = end
            epreward = epreward + r
            print("reward: ", r)

            X = np.array([pos[0], pos_new[0]])
            Y = np.array([pos[1], pos_new[1]])
            plt.plot(X, Y, linestyle=linestyles[ii][1], color='black')
            plt.pause(0.001)
            plt.draw()

            v = v_new
            pos = pos_new
        print("Eprew.", ii, ": ", epreward)



