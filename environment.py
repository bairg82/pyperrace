"Ez a HPC-s verzio³"

OnHPC = True

use_matplotlib = True

if OnHPC:
    # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib
    #cannot use interactive backend
    import matplotlib as mpl
    mpl.use('Agg')

if use_matplotlib:
    import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import numpy as np
from random import randint
from skimage.morphology import disk
from skimage.color import rgb2gray
from math import sqrt
import os
import pickle
from scipy.spatial.distance import cdist
import time

import tracks
import cars

class PaperRaceEnv:
    """ez az osztály biztosítja a tanuláshoz a környezetet"""

    def __init__(self, track_name, car_name, random_init, \
                 ref_calc='default',\
                 save_env_ref_buffer_dir = './env_ref_buffer', \
                 save_env_ref_buffer_name = 'env_ref_buffer_1', \
                 load_env_ref_buffer='', \
                 load_all_env_ref_buffer_dir='',):

        self.car_name = car_name
        self.track_name = track_name

        trk_pic_file, self.trk_col, self.track_inside_color, self.track_outside_color, self.start_line, self.end_line, self.sections, self.ref_actions = tracks.get_track_params(self.track_name)

        if ref_calc == 'default':
            self.set_car('Touring')

        self.use_ref_time = False

        # 0-ázza a start poz-tól való távolságot a reward fv-hez
        self.steps = 0  # az eddig megtett lépések száma

        # a palya hanyad reszet sikerult teljesiteni
        self.game_pos_reward = 0
        # mennyi az eltelt ido
        self.game_time = 0
        # referenciahoz kepest szamitott reward, regi
        self.game_ref_reward = 0
        # teljes reward palya megtetele es ido alapjan
        self.game_reward = 0

        self.step_time = 0
        self.last_step_time = 0
        self.last_step_ref_time_diff = 0
        self.step_ref_time_diff = 0
        self.step_pos_reward = 0
        self.step_reward = 0

        self.pos_last = 0
        self.v_last = 0
        self.end = False
        self.finish = False

        self.section_nr = 0

        self.curr_dist_in = 0
        self.curr_pos_in = 0
        self.curr_dist_out = 0
        self.curr_pos_out = 0

        # buffer a már lekért és kiszámolt referenciákra, hogy gyorsabb legyen a lekérés
        self.ref_buffer_dir = save_env_ref_buffer_dir
        self.ref_buffer_name = save_env_ref_buffer_name

        # referencia buffer inicializalas
        self.ref_buffer = {}
        self.ref_buffer_unsaved = 0
        # self.ref_buffer_load(load_env_ref_buffer, load_all_dir=load_all_env_ref_buffer_dir)

        # beolvassa a pályát
        self.trk_pic = mpimg.imread(trk_pic_file)
        self.trk = rgb2gray(self.trk_pic)  # szürkeárnyalatosban dolgozunk

        self.players = []
        self.players.append(self.Player())
        self.player = self.getplayer()

        self.col_in = rgb2gray(np.reshape(np.array(self.track_inside_color), (1, 1, 3)))[0, 0]
        self.col_out = rgb2gray(np.reshape(np.array(self.track_outside_color), (1, 1, 3)))[0, 0]

        # Ha be van kapcsolva az autó véletlen pozícióból való indítása, random szakaszból indulunk
        self.random_init = random_init

        # A kezdo pozicio a startvonal fele, es onnan 1-1 pixellel "arrebb" Azert hogy ne legyen a startvonal es a
        # kezdeti sebesseg metszo.
        # ezen a ponton section_nr = 0, az elso szakasz a listaban (sections) a startvonal
        # A startvonalra meroleges iranyvektor:
        e_start_x = int(np.floor((self.start_line[0] - self.start_line[2])))
        e_start_y = int(np.floor((self.start_line[1] - self.start_line[3])))
        self.e_start_spd = np.array([e_start_y, -e_start_x]) / np.linalg.norm(np.array([e_start_y, -e_start_x]))

        # A startvonal közepe:
        self.start_x = int(np.floor((self.start_line[0] + self.start_line[2]) / 2))
        self.start_y = int(np.floor((self.start_line[1] + self.start_line[3]) / 2))
        # A kezdő pozíció, a startvonal közepétől, a startvonalra merőleges irányba egy picit eltolva:
        self.starting_pos = np.array([self.start_x, self.start_y]) + np.array([int(self.e_start_spd[0] * 10), int(self.e_start_spd[1] * 10)])

        self.pos = self.starting_pos

        #a kezdo sebesseget a startvonalra merolegesre akarjuk:
        self.starting_spd = self.e_start_spd * 150
        self.v = self.starting_spd

        self.gg_actions = None # az action-ökhöz tartozó vektor értékeit cash-eli a legelajén és ebben tárolja

        # van egy ref... fv. Ahhoz hog az jol mukodjon, kellenek mindig egy "előző" lépés ref adatai. Ezek:
        self.prev_dist_in = 0
        self.prev_dist_out = 0
        self.prev_pos_in = np.array([0, 0])
        self.prev_pos_out = np.array([0, 0])

        self.dists_in, self.dists_in_pos = self.__get_dists_in(False) # a kezdőponttól való "távolságot" tárolja a reward fv-hez
        self.dists_out, self.dists_out_pos = self.__get_dists_out(False) # a kezdőponttól való "távolságot" tárolja

        self.dist_in_max = len(self.dists_in_pos)
        self.dist_out_max = len(self.dists_out_pos)

        # a referencia megallapitashoz meg ugye nem lehet referenciat hasznalni
        self.use_ref_time = False

        # ehhez van egy init, ami eloallitja a belso iv menten mert elorehaladast minden lepesben
        self.ref_dist, self.ref_steps = self.__get_ref_dicts(self.ref_actions)

        #mostmar hasznalhatjuk referencia szerinti diff szamolasokat
        self.use_ref_time = True

        # refference is made now switched to game car
        self.set_car(car_name)


    def reset(self, drawing = False):
        """ha vmiért vége egy menetnek, meghívódik"""
        # 0-ázza a start poz-tól való távolságot a reward fv-hez
        self.steps = 0  # az eddig megtett lépések száma

        # a palya hanyad reszet sikerult teljesiteni
        self.game_pos_reward = 0
        # mennyi az eltelt ido
        self.game_time = 0
        # referenciahoz kepest szamitott reward, regi
        self.game_ref_reward = 0
        # teljes reward palya megtetele es ido alapjan
        self.game_reward = 0

        self.step_time = 0
        self.last_step_time = 0
        self.last_step_ref_time_diff = 0
        self.step_ref_time_diff = 0
        self.step_pos_reward = 0
        self.step_reward = 0

        self.pos_last = 0
        self.v_last = 0
        self.end = False
        self.finish = False

        self.section_nr = 0

        self.curr_dist_in = 0
        self.curr_pos_in = 0
        self.curr_dist_out = 0
        self.curr_pos_out = 0

        """
        # ha a random indítás be van kapcsolva, akkor új kezdő pozíciót választ
        if self.random_init:
            self.starting_pos = self.track_indices[randint(0, len(self.track_indices) - 1)]
            self.prev_dist = self.get_ref_time(self.starting_pos)
        """
        if self.random_init:
            self.section_nr = randint(0, len(self.sections) - 2)
        else:
            self.section_nr = 0 # kezdetben a 0. szakabol indul a jatek
        # print("SectNr: ", self.section_nr)

        self.game_reward = 0
        #drawing
        if drawing:
            self.draw_clear()
            self.draw_track()

    def calc_game_reward(self):
        # calculating gmae raward based on position if not completed
        # if completed then reward based in time
        if self.game_pos_reward >= 0.995:
            # if completed the reward is the reciproc of time -> better time means higher points
            self.game_reward = 0.0 + 1000 / self.game_time
        else:
            # if not completed -100 point is the minimal point and 0 if finished track
            self.game_reward = -100.0 + self.game_pos_reward*100.0

    def calc_game_ref_time_reward(self, remaining_ref_time):
        self.game_ref_reward += self.step_ref_time_diff
        if self.end:
            self.game_ref_reward = remaining_ref_time * -2.0

    def calc_game_position(self):
        # if finish is reached dist maxes can be updated
        if self.finish:
            self.dist_in_max = self.curr_dist_in
            self.dist_out_max = self.curr_dist_out

        distinrate = self.curr_dist_in/self.dist_in_max
        distoutrate = self.curr_dist_out/self.dist_out_max
        return min(distinrate, distoutrate)

    def getplayer(self, name = 'default'):
        for player in self.players:
            if player.name == name:
                return player

    def get_ref_actions(self):
        # passing track action examples
        return tracks.get_ref_actions(self.track_name, self.car_name)

    def get_random_ref_actions(self):
        curr_ref_actions = tracks.get_ref_actions(self.track_name, self.car_name)
        # csak egy referencia lepessor van
        if isinstance(curr_ref_actions, (list, tuple)):
            return curr_ref_actions[int(np.random.uniform(0, int(curr_ref_actions.shape[0]), 1))]
        else:
            return curr_ref_actions

    # it draws the track to a current plot
    def draw_track(self):
        # pálya kirajzolása
        if use_matplotlib:
            plt.imshow(self.trk_pic)
        self.draw_section(self.start_line, color='orange')
        self.draw_section(self.end_line, color='orange')

    # it draws all section from self section
    def draw_sections(self):
        # Szakaszok kirajzolása
        for i in range(len(self.sections)):
                X = np.array([self.sections[i][0], self.sections[i][2]])
                Y = np.array([self.sections[i][1], self.sections[i][3]])
                self.draw_section_wpoints(X, Y, color='blue')

    # it draws all section from self section
    def draw_sections(self, sections, color):
        # Szakaszok kirajzolása
        for i in range(len(sections)):
            section = sections[i]
            self.draw_section(section, color=color)

    # it draws all section from self section
    def draw_section(self, section, color):
        # Szakaszok kirajzolása
        X = np.array([section[0], section[2]])
        Y = np.array([section[1], section[3]])
        self.draw_section_wpoints(X, Y, color=color)

    # it clears current plot
    def draw_clear(self):
        if use_matplotlib:
            plt.clf()

    # it draws a section to current plot
    def draw_section_wpoints(self, X, Y, color):
        if use_matplotlib:
            plt.plot(X, Y, color=color)
            plt.pause(0.001)
            plt.draw()

    # save current figure to file
    def draw_save(self, path = './img/', name = 'figure', count = '', extension = '.png'):
        if use_matplotlib:
            plt.savefig(path + name + count + extension)

    # draw info to current plot
    def draw_info(self, X, Y, text):
        if use_matplotlib:
                plt.text(X, Y, text)

    def update_state(self):

        # meghivjuk a sectionpass fuggvenyt, hogy megkapjuk szakitottunk-e at szakaszt, es ha igen melyiket,
        # es az elmozdulas hanyad reszenel

        # saving variables
        self.last_step_time = self.step_time

        crosses, self.step_time, section_nr, start, self.finish = self.sectionpass(self.pos_last, self.v)

        if self.finish:
            finish_pos = self.pos_last + self.v * self.step_time

        # ha szektor hatart nem ert akkor az ido a lepesido
        if not crosses:
            self.step_time = 1

        # megnezzuk palyan van-e es ha lemegya kkor kint vagy bent:
        step_on_track, inside, outside = self.is_on_track(self.pos)

        # ha nem ment le elvileg 1, ha lement a palya szeleig eso resz, ha celbaert a celig megtett ido
        self.game_time += self.step_time

        # ===================
        # Lépések:
        # ===================

        # Ha lemegy a palyarol:
        if not step_on_track:
            last_pos = self.calc_last_point(self.pos_last, self.pos)
            self.curr_dist_in, self.curr_pos_in, self.curr_dist_out, self.curr_pos_out = self.get_pos_ref_on_side(
                last_pos)

            self.end = True
            # print
            # ha atszakit egy szakaszhatart, es ez az utolso is, tehat pont celbaert es ugy esett le a palyarol:
            if self.finish:
                print("\033[91m {}\033[00m" .format("\nCELBAERT KI"))
            else:
                print("\033[91m {}\033[00m".format("\nLEMENT"))

        # Ha nem ment ki a palyarol:
        else:
            # ha a 0. szakaszt, azaz startvonalat szakit at (nem visszafordult hanem eleve visszafele indul):
            if (start):
                # szamoljunk megtett palyar a kezdo poziciohoz
                self.curr_dist_in, self.curr_pos_in, self.curr_dist_out, self.curr_pos_out = self.get_pos_ref_on_side(self.starting_pos)
                print("\nVISSZAKEZD")
                self.end = True

            # ha atszakit egy szakaszhatart, es ez az utolso is, tehat pont celbaert:
            elif self.finish:
                self.curr_dist_in, self.curr_pos_in, self.curr_dist_out, self.curr_pos_out = self.get_pos_ref_on_side(finish_pos)
                print("\033[92m {}\033[00m".format("\nCELBAERT BE"))
                self.end = True
            # ha barmi miatt az autó megáll, sebessege az alábbinál kisebb, akkor vége
            elif sqrt(self.v[0] ** 2 + self.v[1] ** 2) < 1:
                # mivel nem haladt semmit elore az elozo lepes dist-jei maradhatnak
                self.end = True
            else:
                # igy mar lehet megtett palyat szelet nezni
                self.curr_dist_in, self.curr_pos_in, self.curr_dist_out, self.curr_pos_out = self.get_pos_ref_on_side(
                    self.pos)
                self.end = False

        # updatating rewards
        # position based part
        last_game_pos_reward = self.game_pos_reward
        self.game_pos_reward = self.calc_game_position()
        self.step_pos_reward = self.game_pos_reward - last_game_pos_reward
        self.calc_game_reward()

        # ha nagyon lassan vagy hatrafele halad szinten legyen vege (egy jo lepes 4-6% ot halad egyenesben
        if self.step_pos_reward < 0.001:
            print("\033[92m {}\033[00m".format("\nVege: tul lassu, vagy hatrafele ment!"))
            self.end = True

        # -------- ! ------ modify self.end before this

        if self.end:
            print('End of game!')
            print('Reward: ' + '% 3.3f' % self.game_reward + ' Time: ' + '% 3.3f' % self.game_time + \
                  ' ref_time: ' + '% 3.3f' % self.game_ref_reward)

        self.calc_step_reward()

        if self.use_ref_time:
           self.calc_ref_time_reward()

    def calc_step_reward(self):
        # lepes reward a megtett ut, ha lemegy az -100, ha celbaert akkor azert megkapja a jatek pontot
        if self.end:
            if self.finish:
                # step reward on finish is distance travelled and time reward
                # self.step_reward = self.step_pos_reward * 100.0 + self.game_reward
                self.step_reward = self.step_pos_reward * 100.0 - 100.0
            else:
                # ha lement, tul lassu, visszakezd stb. akkor amit meg a palyan megtett plusz a buntetes
                # kell a palyan megtett mert ket rossz kozul igy el lehet donteni melyik volt kevesbe rossz -> tanulas
                self.step_reward = self.step_pos_reward - 100.0
        else:
            self.step_reward = self.step_pos_reward * 100.0

    def calc_ref_time_reward(self):
        # ref time based part
        self.last_step_ref_time_diff = self.step_ref_time_diff
        self.step_ref_time_diff, remaining_ref_time = self.get_time_diff(self.pos_last, self.pos, self.step_time)
        self.calc_game_ref_time_reward(remaining_ref_time)

    def calc_step(self, gg_action):
        #------------------
        # game phisics

        # az aktuális sebesség irányvektora:
        e1_spd_old = self.v / np.linalg.norm(self.v)
        e2_spd_old = np.array([-1 * e1_spd_old[1], e1_spd_old[0]])

        # a sebessegvaltozas lokalisban
        gg_action = np.asmatrix(gg_action)

        # a sebesseg valtozás globálisban:
        spd_chn_glb = np.round(np.column_stack((e1_spd_old, e2_spd_old)) * gg_action.transpose())

        # az új sebességvektor globalisban:
        spd_new = self.v + np.ravel(spd_chn_glb)

        # az uj pozicio globalisban:
        pos_new = self.pos + spd_new
        # print("PN:", pos_new, "PO:", pos_old)

        self.pos_last = self.pos
        self.v_last = self.v
        self.pos = pos_new
        self.v = spd_new
        #------------------

    def draw_step(self, draw_text='reward', info_text_X = 1300, info_text_Y = 1000):
        curr_dist_in_old, pos_temp_in_old, curr_dist_out_old, pos_temp_out_old = self.get_pos_ref_on_side(self.pos_last)
        # szakasz hatar
        X = np.array([pos_temp_in_old[0], pos_temp_out_old[0]])
        Y = np.array([pos_temp_in_old[1], pos_temp_out_old[1]])
        self.draw_section_wpoints(X, Y, color='magenta')

        X = np.array([self.pos_last[0], self.pos[0]])
        Y = np.array([self.pos_last[1], self.pos[1]])
        self.draw_section_wpoints(X, Y, self.player.color)
        if draw_text == 'time':
            self.draw_info(info_text_X, info_text_Y, 'time:' + str(int(self.game_ref_reward)))
        if draw_text == 'little_time' or draw_text == 'little_reward':
            # a szakasz felezopontjara meroleges vektoron d tavolsagra szoveg kiirasa
            d = 10
            tmp1 = (X[1] - X[0]) * 0.5
            tmp2 = (Y[1] - Y[0]) * 0.5
            h = d / max(sqrt(tmp1 ** 2 + tmp2 ** 2), 0.01)
            text_X = X[0] + tmp1 - tmp2 * h
            text_Y = Y[0] + tmp2 + tmp1 * h
            if draw_text == 'little_time':
                self.draw_info(text_X, text_Y, str('%.3f' % self.step_ref_time_diff))
            elif draw_text == 'little_reward':
                self.draw_info(text_X, text_Y, str('% 2.1f' % (self.step_pos_reward * 100)))
        else:
            self.draw_info(info_text_X, info_text_Y, draw_text)

    def step(self, action, draw, draw_text='reward', draw_info_X = 1300, draw_info_Y = 1000, player='default'):

        # change player if necessary
        if player != self.player.name:
            self.player = self.getplayer(player)

            print('\n  --' + self.player.name + ': ')
            print('    ' + str(action), end='')
        else:
            print(str(action) + ', ', end='')
        # print("\033[93m {}\033[00m".format("        -------ref action:"), a)

        #action = spd_chn

        action = max(min(action, 180), -180)

        gg_action = self.gg_action(action)

        self.calc_step(gg_action)

        self.update_state()

        # Ha akarjuk, akkor itt rajzoljuk ki az aktualis lepes abrajat (lehet maskor kene)
        if draw: # kirajzolja az autót
            self.draw_step(draw_text, draw_info_X, draw_info_Y)

        return self.v, self.pos, self.step_reward, self.step_pos_reward

    def getstate(self):
        return self.end, self.step_time, self.step_ref_time_diff, self.game_reward, self.game_ref_reward


    def is_on_track(self, pos):
        """ a pálya színe és a pozíciónk pixelének színe alapján visszaadja, hogy rajta vagyunk -e a pályán, illetve kint
           vagy bent csusztunk le rola... Lehetne tuti okosabban mint ahogy most van."""

        # meg kell nezni egyatalan a palya kepen belul van-e a pos
        # print(pos)
        if int(pos[1]) > (self.trk_pic.shape[0] - 1) or int(pos[0]) > (self.trk_pic.shape[1] - 1):
            inside = True
            outside = True
            ontrack = False
        else:
            inside = False
            outside = False
            ontrack = True

            if np.array_equal(self.trk_pic[int(pos[1]), int(pos[0])], self.track_inside_color):
                inside = True
            if np.array_equal(self.trk_pic[int(pos[1]), int(pos[0])], self.track_outside_color):
                outside = True
            if inside or outside:
                ontrack = False

            """
            if pos[0] > np.shape(self.trk_pic)[1] or pos[1] > np.shape(self.trk_pic)[0] or pos[0] < 0 or pos[1] < 0 or np.isnan(pos[0]) or np.isnan(pos[1]):
                ontrack = False
            else:
                ontrack = np.array_equal(self.trk_pic[int(pos[1]), int(pos[0])], self.trk_col)
            """

        return ontrack, inside, outside

    def set_car(self, car_name):
        file = cars.get_car_params(car_name)
        self.gg_pic = mpimg.imread(file)
        self.car_name = car_name

    def get_ref_actions(self):
        return np.array(tracks.get_ref_actions(self.track_name, self.car_name))

    def gg_action(self, action):
        # az action-ökhöz tartozó vektor értékek
        # első futáskor cash-eljúk

        if self.gg_actions is None:
            self.gg_actions = [None] * 361 # -180..180-ig, fokonként megnézzük a sugarat.
            for act in range(-180, 181):
                if -180 <= act <= 180:
                    # a GGpic 41x41-es B&W bmp. A közepétől nézzük, meddig fehér. (A közepén,
                    # csak hogy látszódjon, van egy fekete pont!
                    xsrt, ysrt = 21, 21
                    r = 1
                    pix_in_gg = True
                    x, y = xsrt, ysrt
                    while pix_in_gg:
                        # lépjünk az Act irányba +1 pixelnyit, mik x és y ekkor:
                        #rad = np.pi / 4 * (act + 3)
                        rad = (act+180) * np.pi / 180
                        y = ysrt + round(np.sin(rad) * r)
                        x = xsrt + round(np.cos(rad) * r)
                        r = r + 1

                        # GG-n belül vagyunk-e még?
                        pix_in_gg = np.array_equal(self.gg_pic[int(x - 1), int(y - 1)], [255, 255, 255, 255])

                    self.gg_actions[act - 1] = (-(x - xsrt), y - ysrt)
                else:
                    self.gg_actions[act - 1] = (0, 0)

        return self.gg_actions[action - 1]

        # jatek inditasa
    def start_game(self, player='last'):
        print('New game started!')
        # change player if necessary
        if (player != self.player.name and player != 'last'):
            self.player = self.getplayer(player)
        print('  --' + self.player.name + ': ')
        print('    ', end='')
        # kezdeti sebeesseg, ahogy a kornyezet adja
        self.v = np.array(self.starting_spd)

        # sebesség mellé a kezdeti poz. is kell. Ez a kezdőpozíció beállítása:
        self.pos = np.array(self.starting_pos)

        self.curr_dist_in, self.curr_pos_in, self.curr_dist_out, self.curr_pos_out = self.get_pos_ref_on_side(self.pos)

        return self.pos, self.v

    # give section with 2 points, a point and a speed vector, if it goes through this sections returns true
    def check_if_crossed(self, pos, spd, section):
        v1y = section[2] - section[0]
        v1z = section[3] - section[1]
        v2y = spd[0]
        v2z = spd[1]

        p1y = section[0]
        p1z = section[1]
        p2y = pos[0]
        p2z = pos[1]

        t1=0
        t2=0

        cross=False

        # mielott vizsgaljuk a metszeseket, gyorsan ki kell zarni, ha a parhuzamosak a vizsgalt szakaszok
        # ilyenkor 0-val osztas lenne a kepletekben
        if -v1y * v2z + v1z * v2y == 0:
            cross = False
        # Amugy mehetnek a vizsgalatok
        else:
            """
            t2 azt mondja hogy a p1 pontbol v1 iranyba indulva v1 hosszanak hanyadat kell megtenni hogy elerjunk a 
            metszespontig. Ha t2=1 epp v2vegpontjanal van a metszespopnt. t1,ugyanez csak p1 es v2-vel.
            """
            t2 = (-v1y * p1z + v1y * p2z + v1z * p1y - v1z * p2y) / (-v1y * v2z + v1z * v2y)
            t1 = (p1y * v2z - p2y * v2z - v2y * p1z + v2y * p2z) / (-v1y * v2z + v1z * v2y)

            """
            Annak eldontese hogy akkor az egyenesek metszespontja az most a
            szakaszokon belulre esik-e: Ha mindket t, t1 es t2 is kisebb mint 1 és
            nagyobb mint 0
            """
            cross = (0 < t1) and (t1 < 1) and (0 < t2) and (t2 < 1)

        return cross, t2

    def sectionpass(self, pos, spd):
        """
                Ha a Pos - ból húzott Spd vektor metsz egy szakaszt(Szakasz(!),nem egynes) akkor crosses = 1 - et ad vissza(true)
                A t2 az az ertek ami mgmondja hogy a Spd vektor hanyadánál metszi az adott szakaszhatart. Ha t2 = 1 akkor a Spd
                vektor eppenhogy eleri a celvonalat.

                Ezt az egeszet nezi a kornyezet, azaz a palya definialasakor kapott osszes szakaszra (sectionlist) Ha a
                pillanatnyi pos-bol huzott spd barmely section-t jelzo szakaszt metszi, visszaadja hogy:
                crosses = True, azaz hogy tortent szakasz metszes.
                t2 = annyi amennyi, azaz hogy spd hanyadanal metszette
                section_nr = ahanyadik szakaszt metszettuk epp.
                """
        """
        keplethez kello idediglenes ertekek. p1, es p2 pontokkal valamint v1 es v2 iranyvektorokkal adott egyenesek metszespontjat
        nezzuk, ugy hogy a celvonal egyik pontjabol a masikba mutat a v1, a v2 pedig a sebesseg, p2pedig a pozicio
        """
        section_nr = 0
        t2 = 0
        ret_t2 = 0
        sc_cross = False
        start, start_t2 = self.check_if_crossed(pos, spd, self.start_line)
        end, end_t2 = self.check_if_crossed(pos, spd, self.end_line)

        # ha vannak belso szekciok
        if self.sections != []:
            for i in range(self.sections.size[0]):
                sc_cross, sc_t2 = self.check_if_crossed(pos, spd, self.start_line)
                if sc_cross:
                    section_nr = i
                    ret_t2 = t2

        crosses = start or end or sc_cross

        # nem teljesen jo ha tobb mindent atszakit, de azt nem hasznaljuk meg
        if end:
            ret_t2 = end_t2
        elif start:
            ret_t2 = start_t2

        return crosses, ret_t2, section_nr, start, end

    def normalize_data(self, data_orig):
        """
        a háló könnyebben, tanul, ha az értékek +-1 közé esnek, ezért normalizáljuk őket
        pozícióból kivonjuk a pálya méretének a felét, majd elosztjuk a pálya méretével
        """

        n_rows = data_orig.shape[0]
        data = np.zeros((n_rows, 4))
        sizeX = np.shape(self.trk_pic)[1]
        sizeY = np.shape(self.trk_pic)[0]
        data[:, 0] = (data_orig[:, 0] - sizeX / 2) / sizeX
        data[:, 2] = (data_orig[:, 2] - sizeX / 2) / sizeX
        data[:, 1] = (data_orig[:, 1] - sizeY / 2) / sizeY
        data[:, 3] = (data_orig[:, 3] - sizeY / 2) / sizeY

        return data

    def pixel_in_track(self, x, y, color):
        if self.trk[x, y] == color:
            # inside colored pixel found
            return np.array(x, y)
        else:
            return False


    def get_pos_ref_on_side(self, position):
        # https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
        # from scipy.spatial.distance import cdist
        # cdist(XA, XB, metric='euclidean', p=2, V=None, VI=None, w=None)

        # ha a posiciot mar egyszer kiszamoltuk
        curr_dist_in = cdist([position], self.dists_in_pos).argmin()
        pos_in = self.dists_in_pos[curr_dist_in]

        curr_dist_out = cdist([position], self.dists_out_pos).argmin()
        pos_out = self.dists_out_pos[curr_dist_out]

        return curr_dist_in, pos_in, curr_dist_out, pos_out

    # TODO it is very slow
    # TODO it is used many times
    # there is a better way
    # https://stackoverflow.com/questions/307445/finding-closest-non-black-pixel-in-an-image-fast
    def get_pos_ref_on_side_old(self, position):
        starttime = time.time()

        """Ref adatokat ado fuggveny.
        posision csak palyan levo pont lehet, ha enm akkor hibat fog adni.
        Nincs vizsgálat erre
        Input tehat:
        position: a palya egy adott pontja

        Output:
        belso iv menten megtett ut,kulso iv menten megtett ut, belso iv referencia pontja, es kulso iv ref pontja"""

        # ha a posiciot mar egyszer kiszamoltuk
        if tuple(position) in self.ref_buffer:
            curr_dist_in, pos_in, curr_dist_out, pos_out = self.ref_buffer[tuple(position)]
            return curr_dist_in, pos_in, curr_dist_out, pos_out

        # az algoritmus működik, hogy az aktuális pozícióban egyre nagyobb négyzetet rajzol, aminek a pixelein végig
        # végig megy ezt addig csinálja amíg nem talál egyet és a talált pont távolságánál 2 szer nagyobb a szélesség
        #
        # akkor azt a pixelt kikeresi a dist_dict-ből, majd megnezi ehhez mennyi a ref sebesseggel mennyi ido
        # jar

        # konvertálás -> np array
        posisiton_np = np.array(position, dtype='int32')

        # pos_new-el egy vonalban levo belso pont meghatarozasa---------
        # tmp_in = [0]
        inside_pixels = []
        inside_pixel_distances = []
        outside_pixels = []
        outside_pixel_distances = []
        # this is the box edge distance from the pixel
        scan_dist = 0
        search_succesful = False
        search_terminated = False
        top_edge_reached = False
        bottom_edge_reached = False
        right_edge_reached = False
        left_edge_reached = False

        best_inside_pixel_found = False
        best_outside_pixel_found = False

        while ((not search_succesful) and (not search_terminated)):
            scan_dist = scan_dist + 1 # növeljük a boxot
            left_edge = int(position[0]) - scan_dist
            right_edge = int(position[0]) + scan_dist
            top_edge = int(position[1]) - scan_dist
            bottom_edge = int(position[1]) + scan_dist

            # if an edge is reached we wont search there, because tahat pixels doesn exist
            # and edges are modified (max or min) to work in other edge search
            if left_edge < 0:
                left_edge_reached = True
                left_edge = 0
            if right_edge >= self.trk.shape[1]:
                right_edge_reached = True
                right_edge = self.trk.shape[1]- 1
            if top_edge < 0:
                top_edge_reached = True
                top_edge = 0
            if bottom_edge >= self.trk.shape[0]:
                bottom_edge_reached = True
                bottom_edge = self.trk.shape[0] - 1

            # all edge is reached search is terminated
            if (left_edge_reached and right_edge_reached and top_edge_reached and bottom_edge_reached):
                search_terminated = True

            # left edge pixel column
            if not left_edge_reached:
                for i in range(top_edge, bottom_edge):
                    # inside colored pixel found
                    if self.trk[i, left_edge] == self.col_in:
                        inside_pixels.append([left_edge, i])
                        inside_pixel_distances.append(sqrt((left_edge - position[0])**2 + (i - position[1])**2))
                    # outside colored pixel found
                    if self.trk[i, left_edge] == self.col_out:
                        outside_pixels.append([left_edge, i])
                        outside_pixel_distances.append(sqrt((left_edge - position[0])**2 + (i - position[1])**2))

            # right edge pixel column
            if not right_edge_reached:
                for i in range(top_edge, bottom_edge):
                    # inside colored pixel found
                    if self.trk[i, right_edge] == self.col_in:
                        inside_pixels.append([right_edge, i])
                        inside_pixel_distances.append(sqrt((right_edge - position[0])**2 + (i - position[1])**2))
                    # outside colored pixel found
                    if self.trk[i, right_edge] == self.col_out:
                        outside_pixels.append([right_edge, i])
                        outside_pixel_distances.append(sqrt((right_edge - position[0])**2 + (i - position[1])**2))

            # top edge pixel row
            if not top_edge_reached:
                for i in range(left_edge, right_edge):
                    # inside colored pixel found
                    if self.trk[top_edge, i] == self.col_in:
                        inside_pixels.append([i, top_edge])
                        inside_pixel_distances.append(sqrt((i - position[0])**2 + (top_edge - position[1])**2))
                    # outside colored pixel found
                    if self.trk[top_edge, i] == self.col_out:
                        outside_pixels.append([i, top_edge])
                        outside_pixel_distances.append(sqrt((i - position[0])**2 + (top_edge - position[1])**2))

            # bottom edge pixel row
            if not bottom_edge_reached:
                for i in range(left_edge, right_edge):
                    # inside colored pixel found
                    if self.trk[bottom_edge, i] == self.col_in:
                        inside_pixels.append([i, bottom_edge])
                        inside_pixel_distances.append(sqrt((i - position[0])**2 + (bottom_edge - position[1])**2))
                    # outside colored pixel found
                    if self.trk[bottom_edge, i] == self.col_out:
                        outside_pixels.append([i, bottom_edge])
                        outside_pixel_distances.append(sqrt((i - position[0])**2 + (bottom_edge - position[1])**2))

            # all pixels investigated on perimeter

            #best inside track pixel is reached
            if len(inside_pixels) > 0:
                if min(inside_pixel_distances) < scan_dist:
                    best_inside_pixel_found = True
                    pos_in = inside_pixels[inside_pixel_distances.index(min(inside_pixel_distances))]
            #best outside track edge is reached
            if len(outside_pixels) > 0:
                if min(outside_pixel_distances) < scan_dist:
                    best_outside_pixel_found = True
                    pos_out = outside_pixels[outside_pixel_distances.index(min(outside_pixel_distances))]

                    # check if goal is reached
            if best_inside_pixel_found and best_outside_pixel_found == True:
                search_succesful = True
        # end of while -> pixel search

        # A kapott belso es kulso pontokrol megnezni milyen messze vannak a starttol:--------------------------

        # Ha kozel vagyunk a falhoz, 1 sugaru r adodik, es egy pixelnyivel mindig pont mas key-t ker mint ami letezik.
        # Ezt  elkerulendo, megnezzuk hogy amivel meg akarjuk hivni az valoban benne van-e a dictekben.
        # tehat ha a dict-ek tartalmazzak pos_in es pos_out-ot:
        if tuple(pos_in) in self.dists_in and tuple(pos_out) in self.dists_out:
            curr_dist_in = self.dists_in[tuple(pos_in)] # a dist_dict-ből lekérjük a start-tól való távolságát
            curr_dist_out = self.dists_out[tuple(pos_out)] # a dist_dict-ből lekérjük a start-tól való távolságát
        else:
            raise Exception(pos_in, pos_out, 'out of track, no border found error')

        # rakjuk el a bufferba
        self.ref_buffer[tuple(position)] = [curr_dist_in, pos_in, curr_dist_out, pos_out]
        self.ref_buffer_unsaved += 1
        # save env ref buffer if 1000 new reference exists
        if self.ref_buffer_unsaved >= 1000:
            self.ref_buffer_unsave = 0
            self.ref_buffer_save()
        print('dists: ' + str(starttime -time.time()))
        return curr_dist_in, pos_in, curr_dist_out, pos_out

    def ref_buffer_save(self):

        file = open(self.ref_buffer_dir + '/' + self.ref_buffer_name, "wb")
        for key, value in self.ref_buffer.items():
            pickle.dump([key[0], key[1], value[0], value[1][0],value[1][1], value[2], value[3][0], value[3][1]], file)
        file.close()


    def ref_buffer_load(self, file_name='', load_all_dir = ''):
        try:
            # no directory is specified then load file with exect path
            if load_all_dir == '':
                self.ref_buffer_fill(file_name)
            # load all file in defined directory if specified
            else:
                for tmp_file_name in os.listdir(load_all_dir):
                    self.ref_buffer_fill(load_all_dir + '/' + tmp_file_name)
        except:
            print('wrong file name or directory')

    def ref_buffer_fill(self, file_name):
        try:
            with open(file_name, 'rb') as file:
                while True:
                    obj = pickle.load(file)
                    if (not obj):
                        break
                    key = [int(obj[0]), int(obj[1])]
                    value = [obj[2], [obj[3], obj[4]], obj[5], [obj[6], obj[7]]]
                    self.ref_buffer[tuple(key)] = value
        except:
            # no ref buffer
            pass
    """
    def get_reward(self, pos_old, pos_new, step_nr):
        ""Reard ado fuggveny. Egy adott lepeshez (pos_old - pos new) ad jutalmat. Eredetileg az volt hogy -1 azaz mint
        mint eltelt idő. Most megnezzuk mivan ha egy referencia lepessorhoz kepest a nyert vagy veszetett ido lesz.
        kb. mint a delta_time channel a MOTEC-ben""

        # Fent az env initbe kell egy referencia lepessor. Actionok, egy vektorban...vagy akarhogy.
        # Az Actionokhoz tudjuk a pos-okat minden lepesben
        # Es tudjuk a dist_in es dist_outokat is minden lepeshez (a lepes egy timestep elvileg)
        # A fentiek alapjan pl.:Look-up table szeruen tudunk barmilyen dist-hez lepest (idot) rendelni

        # Megnezzuk hogy a pos_old es a pos_new milyen dist_old es dist_new-hez tartozik (in vagy out, vagy atlag...)

        # Ehez a dist_old es dist new-hoz megnezzuk hogy a referencia lepessor mennyi ido alatt jutott el ezek lesznek
        # step_old es step_new.

        # A step_old es step_new kulonbsege azt adja hogy azt a tavot, szakaszt, amit a jelenlegi pos_old, pos_new
        # megad, azt a ref lepessor, mennyi ido tette meg. A jelenlegi az 1 ido, hiszen egy lepes. A ketto kulonbsege
        # adja majd pillanatnyi rewardot.



        return reward
    """

    def get_time_diff(self, pos_old, pos_new, act_rew):
        """Reward ado fuggveny. Egy adott lepeshez (pos_old - pos new) ad jutalmat. Eredetileg az volt hogy -1 azaz mint
        mint eltelt idő. Most megnezzuk mivan ha egy referencia lepessorhoz kepest a nyert vagy veszetett ido lesz.
        kb. mint a delta_time channel a MOTEC-ben

        pre_dist_in, az aktualis lepessor, elozo pozicio belso iv menti tavolsaga,
        curr_dist_in, az aktualis lepessor, jelenlegi pozicio belso iv menti tavolsaga
        act_rew: az aktualis lepessor mennyi ido alatt ert e tavolsagok kozott

        visszater a rew_dt, ami azt adja, hogy a referencia lepessorhoz kepest ez mennyivel tobb ido"""

        pre_dist_in, pre_pos_in, pre_dist_out, pre_pos_out = self.get_pos_ref_on_side(pos_old)

        # amennyi ido (lepes) alatt a ref_actionok, a pre_dist-ből a curr_dist-be eljutottak--------------------------
        # look-up szerűen lesz. Először a bemenetek:
        x = self.ref_dist
        y = self.ref_steps

        xvals = np.array([pre_dist_in, self.curr_dist_in])
        # print("elozo es aktualis tav:", xvals)

        # ezekre a tavolsagokra a referencia lepessor ennyi ido alaptt jutott el
        yinterp = np.interp(xvals, self.ref_dist, self.ref_steps, 0)
        # print("ref ennyi ido alatt jutott ezekre:", yinterp)

        # tehat ezt a negyasau lepest a referencia ennyi ido alatt tette meg ugyanezt a palya tavot(- legyen, hogy a kisebb ido legyen a magasabb
        # reward) :
        ref_step_time = yinterp[1] - yinterp[0]
        # print("a ref. ezen lepesnyi ideje:", ref_delta)

        # az atualis lepesben az eltelt ido nyilvan -1, illetve ha ido-bunti van akkor ennel tobb, eppen a reward
        # print("elozo es aktualis lepes kozott eltelt ido:", act_rew)

        # amenyivel az aktualis ebben a lepesben jobb, azaz kevesebb ido alatt tette meg ezt a elmozdulat, mint a ref
        # lepessor, az:
        # ha mint a referencia akkor pozitiv,
        rew_dt = ref_step_time - act_rew
        #print("az aktualis, ebben a lepesben megtett tavot ennyivel kevesebb ido alatt tette meg mint a ref. (ha (-) akkor meg több):", rew_dt)
        # a kieses helyetol a ref lepessorral, hatra levo ido:
        remain_time = self.ref_steps[-1] - yinterp[1]

        return -rew_dt, remain_time

    def calc_last_point(self, pos_old, pos_new):
        # a sebessegvektor iranyaban egyre nagyobb vektorral vizsgalja, hogy mar kint van-e
        # ha igen az utolso elotti lepes meg bent van ezzel a posicioval ter vissza

        tmp_dist = 1.0
        v_x = float(pos_new[0] - pos_old[0])
        v_y = float(pos_new[1] - pos_old[1])
        v_abs = sqrt(v_x ** 2 + v_y ** 2)
        dist_x = 0
        dist_y = 0
        while tmp_dist < v_abs:
            dist_x = int(tmp_dist * v_x / v_abs)
            dist_y = int(tmp_dist * v_y / v_abs)
            ontrack, inside, outside = self.is_on_track(np.array([pos_old[0] + dist_x, pos_old[1] + dist_y]))
            if (ontrack is False):
                break
            tmp_dist += 1.0
        # az egyel korabbi tavolsag ertekhez tartozo lesz meg belül
        tmp_dist -= 1.0
        try:
            dist_x = int(tmp_dist * v_x / v_abs)
            dist_y = int(tmp_dist * v_y / v_abs)
        except:
            print('track side calculation error')
            dist_x = 0
            dist_y = 0

        # ez az utolso palyan levo pozicio
        last_pos = np.array([pos_old[0] + dist_x, pos_old[1] + dist_y])
        return last_pos

    def __get_ref_dicts(self, ref_actions):

        # Fent az env initbe kell egy referencia lepessor. Actionok, egy vektorban...vagy akarhogy.
        # Az Actionokhoz tudjuk a pos-okat minden lepesben

        steps_nr = range(0, len(ref_actions))

        ref_steps = np.zeros(len(ref_actions))
        ref_dist = np.zeros(len(ref_actions))


        self.draw_clear()
        self.draw_track()

        for i in steps_nr:
            #nye = input('Give input')
            action = self.ref_actions[i]
            v_new, pos_new, step_reward, reward = self.step(action, draw=True, draw_text='')
            curr_dist_in, pos_in, curr_dist_out, pos_out = self.get_pos_ref_on_side(pos_new)
            ref_dist[i] = curr_dist_in
            ref_steps[i] = self.game_time

        print(ref_dist)
        print(ref_steps)

        return ref_dist, ref_steps

    def __get_dists_in(self, draw=False):
        """
        "feltérképezi" a pályát a reward fv-hez
        a start pontban addig növel egy korongot, amíg a korong a pálya egy belső pixelét (piros) nem fedi
        ekkor végigmegy a belső rész szélén és eltárolja a távolságokat a kezdőponttól úgy,
        hogy közvetlenül a pálya széle mellett menjen
        úgy kell elképzelni, mint a labirintusban a falkövető szabályt

        :return: dictionary, ami (pálya belső pontja, távolság) párokat tartalmaz
        """

        dist_dict_in = {} # dictionary, (pálya belső pontja, távolság) párokat tartalmaz
        dist_points = []
        # a generalashoz a start pozicio alapbol startvonal kozepe lenne. De valahogy a startvonal kozeleben a dist az
        # szar tud lenni ezert az algoritmus kezdo pontjat a startvonal kicsit visszabbra tesszuk.
        #
        start_point = np.array([self.start_x, self.start_y]) - np.array([int(self.e_start_spd[0] * 10), int(self.e_start_spd[1] * 10)])
        #trk = rgb2gray(self.trk_pic) # szürkeárnyalatosban dolgozunk
        #col_in = rgb2gray(np.reshape(np.array(self.track_inside_color), (1, 1, 3)))[0, 0]
        tmp = [0]
        r = 0 # a korong sugarát 0-ra állítjuk
        while not np.any(tmp): # amíg nincs belső pont fedésünk
            r = r + 1 # növeljük a sugarat
            mask = disk(r) # létrehozzuk a korongot (egy mátrixban 0-ák és egyesek)
            tmp = self.trk[start_point[1] - r:start_point[1] + r + 1, start_point[0] - r:start_point[0] + r + 1] # kivágunk
            # a képből egy kezdőpont kp-ú, ugyanekkora részt
            tmp = np.multiply(mask, tmp) # maszkoljuk a koronggal
            tmp[tmp != self.col_in] = 0 # a kororngon ami nem piros azt 0-ázzuk

        indices = [p[0] for p in np.nonzero(tmp)] #az első olyan pixel koordinátái, ami piros
        offset = [indices[1] - r, indices[0] - r] # eltoljuk, hogy a kp-tól megkapjuk a relatív távolságvektorát
        # (a mátrixban ugye a kp nem (0, 0) (easter egg bagoly) indexű, hanem középen van a sugáral le és jobbra eltolva)
        start_point = np.array(start_point + offset) # majd a kp-hoz hozzáadva megkapjuk a képen a pozícióját az első referenciapontnak
        dist = 0
        dist_dict_in[tuple(start_point)] = dist # ennek 0 a távolsága a kp-tól
        JOBB, FEL, BAL, LE = [1, 0], [0, -1], [-1, 0], [0, 1]  # [x, y], tehát a mátrix indexelésekor fordítva, de a pozícióhoz hozzáadható azonnal
        dirs = [JOBB, FEL, BAL, LE]
        direction_idx = 0
        point = start_point
        if draw:
            self.draw_track()
        while True:
            dist += 1 # a távolságot növeli 1-gyel
            bal_ford = dirs[(direction_idx + 1) % 4] # a balra lévő pixel eléréséhez
            jobb_ford = dirs[(direction_idx - 1) % 4] # a jobbra lévő pixel eléréséhez
            if self.trk[point[1] + bal_ford[1], point[0] + bal_ford[0]] == self.col_in: # ha a tőlünk balra lévő pixel piros
                direction_idx = (direction_idx + 1) % 4 # akkor elfordulunk balra
                point = point + bal_ford
            elif self.trk[point[1] + dirs[direction_idx][1], point[0] + dirs[direction_idx][0]] == self.col_in: # ha az előttünk lévő pixel piros
                point = point + dirs[direction_idx] # akkor arra megyünk tovább
            else:
                direction_idx = (direction_idx - 1) % 4 # különben jobbra fordulunk
                point = point + jobb_ford

            dist_dict_in[tuple(point)] = dist # a pontot belerakjuk a dictionarybe
            dist_points.append(point)

            if draw:
                self.draw_section([point[0]], [point[1]], 'y')

            if np.array_equal(point, start_point): # ha visszaértünk az elejére, akkor leállunk
                break

        return dist_dict_in, dist_points

    def __get_dists_out(self, draw=False):
        """
        "feltérképezi" a pályát a reward fv-hez
        a start pontban addig növel egy korongot, amíg a korong a pálya egy belső pixelét (piros) nem fedi
        ekkor végigmegy a belső rész szélén és eltárolja a távolságokat a kezdőponttól úgy,
        hogy közvetlenül a pálya széle mellett menjen
        úgy kell elképzelni, mint a labirintusban a falkövető szabályt

        :return: dictionary, ami (pálya belső pontja, távolság) párokat tartalmaz
        """

        dist_dict_out = {} # dictionary, (pálya külső pontja, távolság) párokat tartalmaz
        dist_points = []
        # a generalashoz a start pozicio alapbol startvonal kozepe lenne. De valahogy a startvonal kozeleben a dist az
        # szar tud lenni ezert az algoritmus kezdo pontjat a startvonal kicsit visszabbra tesszuk.
        # (TODO: megerteni miert szarakodik a dist, es kijavitani)
        start_point = np.array([self.start_x, self.start_y])
        #- np.array([int(self.e_start_spd[0] * 10), int(self.e_start_spd[1] * 10)])
        #trk = rgb2gray(self.trk_pic) # szürkeárnyalatosban dolgozunk
        #col_out = rgb2gray(np.reshape(np.array(self.track_outside_color), (1, 1, 3)))[0, 0]
        tmp = [0]
        r = 0 # a korong sugarát 0-ra állítjuk
        while not np.any(tmp): # amíg nincs belső pont fedésünk
            r = r + 1 # növeljük a sugarat
            mask = disk(r) # létrehozzuk a korongot (egy mátrixban 0-ák és egyesek)
            tmp = self.trk[start_point[1] - r:start_point[1] + r + 1, start_point[0] - r:start_point[0] + r + 1] # kivágunk
            # a képből egy kezdőpont kp-ú, ugyanekkora részt
            tmp = np.multiply(mask, tmp) # maszkoljuk a koronggal
            #???? con_in-nek kellene lennie
            tmp[tmp != self.col_out] = 0 # a kororngon ami nem piros azt 0-ázzuk

        indices = [p[0] for p in np.nonzero(tmp)] #az első olyan pixel koordinátái, ami piros
        offset = [indices[1] - r, indices[0] - r] # eltoljuk, hogy a kp-tól megkapjuk a relatív távolságvektorát
        # (a mátrixban ugye a kp nem (0, 0) (easter egg bagoly) indexű, hanem középen van a sugáral le és jobbra eltolva)
        start_point = np.array(start_point + offset) # majd a kp-hoz hozzáadva megkapjuk a képen a pozícióját az első referenciapontnak
        dist = 0
        dist_dict_out[tuple(start_point)] = dist # ennek 0 a távolsága a kp-tól
        JOBB, FEL, BAL, LE = [1, 0], [0, -1], [-1, 0], [0, 1]  # [x, y], tehát a mátrix indexelésekor fordítva, de a pozícióhoz hozzáadható azonnal

        # INNENTOL KEZDVE A LENTI KOMMENTEK SZAROK!!! A KULSO IVEN MAS "FORGASIRANY" SZERINT KELL KORBEMENNI EZERT MEG
        # VANNAK MASITVA A dirs-benAZ IRANYOK SORRENDJE A __get_dist_in-hez kepest!!!
        dirs = [JOBB, LE, BAL, FEL]
        direction_idx = 0
        point = start_point
        if draw:
            self.draw_track()
        while True:
            dist += 1 # a távolságot növeli 1-gyel
            bal_ford = dirs[(direction_idx - 1) % 4] # a balra lévő pixel eléréséhez
            jobb_ford = dirs[(direction_idx + 1) % 4] # a jobbra lévő pixel eléréséhez
            if self.trk[point[1] + jobb_ford[1], point[0] + jobb_ford[0]] == self.col_out: # ha a tőlünk jobbra lévő pixel fehér
                direction_idx = (direction_idx + 1) % 4 # akkor elfordulunk jobbra
                point = point + jobb_ford
            elif self.trk[point[1] + dirs[direction_idx][1], point[0] + dirs[direction_idx][0]] == self.col_out: # ha az előttünk lévő pixel fehér
                point = point + dirs[direction_idx] # akkor arra megyünk tovább
            else:
                direction_idx = (direction_idx - 1) % 4 # különben jobbra fordulunk
                point = point + bal_ford

            if draw:
                self.draw_section([point[0]], [point[1]], 'y')

            # ha visszaértünk az elejére, akkor leállunk
            if np.array_equal(point, start_point): # ha visszaértünk az elejére, akkor leállunk
                break
            # ha lemegyünk a képről akkor is leállunk
            if (point[0] < 0 or point[1] < 0 or point[0] >= self.trk.shape[1] or point[1] >= self.trk.shape[0]):
                break

            dist_dict_out[tuple(point)] = dist # a pontot belerakjuk a dictionarybe
            dist_points.append(point)

        return dist_dict_out, dist_points

    def clean(self):
        # self.ref_buffer_save()
        del self.ref_buffer
        del self.players

    def new_player(self, name, color):
        new = self.Player(name, color)
        self.players.append(new)

    class Player(object):
        def __init__(self, name = 'default', color = (0, 0, 1)):
            self.name = name
            self.color = color

