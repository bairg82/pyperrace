"""
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""

import tensorflow as tf
import numpy as np
import os

"!!!Ez a HPC-s verziÃ³"

# OnHPC = True

# used_device = '/gpu:0'
used_device = '/cpu:0'

#import gym
from environment import PaperRaceEnv
#from gym import wrappers
import tflearn
import argparse
import pprint as pp

import Agent

from replay_buffer import ReplayBuffer



# ===========================
#   Agent Training
# ===========================

def get_steps_with_reference(env, deviation, step_count_from_start = 0):
    # if null it will be random

    # az emberi lepessorok kozul valasszunk egyet veletlenszeruen mint aktualis epizod lepessor:
    curr_ref_actions = env.get_random_ref_actions()
    size_curr_ref_actions = len(curr_ref_actions)

    if step_count_from_start == 0:
        actions_size = int(np.random.uniform(0, size_curr_ref_actions, 1))
    else:
        actions_size = min(abs(step_count_from_start), size_curr_ref_actions)

    actions = []
    for i in range(actions_size):
        actions.append(int(np.random.normal(curr_ref_actions[i], deviation, 1)))

    return actions, actions_size

def get_reference_episode(env, episode, max_episodes):
    ep_for_exp = np.array([0, 0.005,
                           1.15, 1.25,
                           1.35, 1.45]) * int(max_episodes)

    # Minden sor szam pedig hogy abban a fentiekben megadott intervallumokban mennyiről mennyire csökkenjen a szórás.
    deviations = np.array([0, 5,
                            10, 0,
                            20, 0])

    ref_episode = (episode in range(int(ep_for_exp[0]), int(ep_for_exp[1]))) or (
            episode in range(int(ep_for_exp[2]), int(ep_for_exp[3]))) or (
            episode in range(int(ep_for_exp[4]), int(ep_for_exp[5])))

    # a random lepesekhez a szoras:
    deviation = np.interp(episode, ep_for_exp, deviations)

    if ref_episode:
        actions, actions_size = get_steps_with_reference(env, deviation)
    else:
        actions = []
        actions_size = 0

    return ref_episode, actions, actions_size

def get_ref_step(step, max_steps, reference_steps, reference_step_size):
    # ha nem ért még véget az epizod, de mar a ref lepessor vege, akkor random lepkedunk
    if step < reference_step_size:
        a = reference_steps[step]
        player = 'reference'
    else:
        player = 'random'
        a = int(np.random.uniform(-180, 180, 1))

    return a, player

def play_train(env, agent, replay_buffer, max_episodes, max_episode_len, minibatch_size, \
               show_window, save_image_episodes, save_graph_episodes, actor_noise = 'not implemented'):

    player = 'agent'
    env.new_player(player, (1, 0, 0))

    # ====================
    # Indul egy epizod:
    # ====================

    for i in range(max_episodes):
        # alapállapotba hozzuk a környezetet
        env.reset()

        pos, v = env.start_game()

        # kezdeti teljes epzód alatt szerzett jutalom, és legjobb q étrék:
        ep_reward = 0

        # egy egy epizódon belül ennyi lépés van maximum:
        for j in range(max_episode_len):
            # state
            s = [v[0], v[1], pos[0], pos[1]]

            a = int(agent.actor.predict(np.reshape(s, (1, agent.state_dim))))

            v_new, pos_new, step_reward, pos_reward = env.step(a, draw, draw_text='little_reward', player=player)

            end, time, last_t_diff, game_pos_reward, game_ref_reward = env.getstate()

            r = step_reward
            if end:
                full_reward = game_pos_reward

            # megintcsak a kétfelől összemásolgatott kodok miatt, feleltessunkk meg egymasnak változókat:
            s2 = [v_new[0], v_new[1], pos_new[0], pos_new[1]]

            terminal = end

            #és akkor a megfeleltetett változókkal már lehet csinálni a replay memory-t:
            replay_buffer.add(np.reshape(s, (agent.state_dim,)), np.reshape(a, (agent.action_dim,)), r, terminal, \
                              np.reshape(s2, (agent.state_dim,)))

            if replay_buffer.size() > int(minibatch_size):  # and not rand_episode:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(minibatch_size)

                ep_ave_max_q_cum += agent.train(s_batch, a_batch, r_batch, t_batch, s2_batch)

            # Keep adding experience to the memory until there are at least minibatch size samples, És amig a
            # tanulas elejen a random lepkedos fazisban vagyunk.

            if terminal:
                break
            # end of if terminal

        #Ha egybol (J=0-nal vege)
        if j == 0:
            ep_ave_max_q = ep_ave_max_q_cum / float(1)
        else:
            ep_ave_max_q = ep_ave_max_q_cum / float(j)

        agent.update_summaries(full_reward, ep_ave_max_q, best_reward, i)

        # end of steps

def play_train_with_ref(env, agent, replay_buffer, max_episodes, max_episode_len, minibatch_size, \
               show_window, save_image_episodes, save_graph_episodes, actor_noise = 'not implemented'):
# Set up summary Ops
    # ----------------------------

    # az emberi jatekokat bele kell "keverni" majd, mint experience. Hogy a teljes tanitásra szánt epizodok alatt
    # mikor, az a lenti matrixban dol el. Minden sor egy szakaszt jelöl, amiben exploration van:

    use_refference_steps = True


    pre_train = False
    pre_trained = False
    step_train = True

    #Full random step probability
    rand_stp_normal = 0.01

    # store steps in it
    episode_steps = []

    best_reward = -100.0


    env.new_player('agent', (1, 0, 0))
    env.new_player('reference', (0, 0, 1))
    env.new_player('random', (1, 1, 0))

    # ====================
    # Indul egy epizod:
    # ====================

    for i in range(max_episodes):
        # ------------------kornyezet kirajzolasahoz---------------------------------
        # draw in this episode
        if i % save_image_episodes == 0 or show_window == 'allstep':
            draw = True
        else:
            draw = False

        # alapállapotba hozzuk a környezetet
        env.reset(draw)

        pos, v = env.start_game()

        # kezdeti teljes epzód alatt szerzett jutalom, és legjobb q étrék:
        ep_reward = 0
        ep_ave_max_q_cum = 0
        # legjobb elert reward

        lepesek = []

        # --------------- Refernce player at the begining
        # Actionok:
        # Ha i (epizod) abban atartományban van amikor emberi lepes alapu random epizodot akarunk akkor:
        reference_episode, reference_steps, reference_step_size = get_reference_episode(env, i, max_episodes)
        # manual reference negation
        reference_episode = False

        # egy egy epizódon belül ennyi lépés van maximum:
        for j in range(max_episode_len):

            # state
            s = [v[0], v[1], pos[0], pos[1]]

            # -------------Reference player
            if reference_episode:
                a, player = get_ref_step(j, max_episode_len, reference_steps, reference_step_size)

            # ------------ Random steps frequently random player
            elif np.random.uniform(0, 1, 1) < rand_stp_normal:
                player = 'random'
                a = int(np.random.uniform(-180, 180))

            # ------------ Agent playing
            else:
                a = int(agent.actor.predict(np.reshape(s, (1, agent.state_dim))))
                player = 'agent'

            v_new, pos_new, step_reward, pos_reward = env.step(a, draw, draw_text='little_reward', player=player)

            end, time, last_t_diff, game_pos_reward, game_ref_reward = env.getstate()

            # giving reward based on reference:
            reward_based_on = ''

            if reward_based_on == 'reference':
                full_reward = game_ref_reward
                if end:
                    r = game_ref_reward
                else:
                    r = last_t_diff
            else:
                r = step_reward
                if end:
                    full_reward = game_pos_reward

            # megintcsak a kétfelől összemásolgatott kodok miatt, feleltessunkk meg egymasnak változókat:
            s2 = [v_new[0], v_new[1], pos_new[0], pos_new[1]]
            terminal = end

            #és akkor a megfeleltetett változókkal már lehet csinálni a replay memory-t:
            replay_buffer.add(np.reshape(s, (agent.state_dim,)), np.reshape(a, (agent.action_dim,)), r, terminal, \
                              np.reshape(s2, (agent.state_dim,)))

            if step_train:
                if replay_buffer.size() > int(minibatch_size):  # and not rand_episode:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(minibatch_size)

                    ep_ave_max_q_cum += agent.train(s_batch, a_batch, r_batch, t_batch, s2_batch)

            # Keep adding experience to the memory until there are at least minibatch size samples, És amig a
            # tanulas elejen a random lepkedos fazisban vagyunk.

            lepesek.append(a)

            if terminal:
                break
            # end of if terminal

        #Ha egybol (J=0-nal vege)
        if j == 0:
            ep_ave_max_q = ep_ave_max_q_cum / float(1)
        else:
            ep_ave_max_q = ep_ave_max_q_cum / float(j)

        agent.update_summaries(full_reward, ep_ave_max_q, best_reward, i)

        # end of steps

        # episode lepes, info mentes, legjobb kigyujteshez
        if not reference_episode:
            episode_steps.append([i, full_reward, lepesek])

        print('| Reward: {:.3f} | Episode: {:d} | Qmax: {:.4f}'.format(full_reward, i, (ep_ave_max_q / float(j))))
        if pre_train:
            if not pre_trained:
                pre_trained = True
                for i in range(1000):
                    if replay_buffer.size() > int(minibatch_size):  # and not rand_episode:
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(minibatch_size)

                        ep_ave_max_q_cum += agent.train(s_batch, a_batch, r_batch, t_batch, s2_batch)

        # minden századik epizód után legyen mentés
        if i % save_graph_episodes == 0:
            agent.save(str(i))

            #legjobb mentese
            if len(episode_steps) > 0:
                best_reward = save_best_episode_steps(episode_steps)

        # minden x epizód után legyen kép mentés
        if i % save_image_episodes == 0:
            env.draw_info(400, \
                          1450, \
                          '| Reward: {:.3f} | Episode: {:d} | Qmax: {:.4f}'.format(full_reward, \
                                                                                   i, \
                                                                                   (ep_ave_max_q / float(j))))
            env.draw_save(name='e', count=str(i))


def save_best_episode_steps(episode_steps):
    # sorting
    sorted_list = sorted(episode_steps, key=lambda x: x[1])[-1:]
    print("best episode:")
    print(sorted_list)
    best_reward = sorted_list[0][1]
    # writing best to file
    with open('best.stp', 'w') as file:
        file.write(str(sorted_list[0][1]) + "\033")
        for k in range(len(sorted_list[0][2])):
            file.write(str(sorted_list[0][2][k]) + "\033")
    return best_reward

def main(args):
    # GG1.bmp is used for reward function
    env = PaperRaceEnv('h1', ref_calc = 'default', car_name='Gokart', random_init=False, \
                       save_env_ref_buffer_dir=args['save_env_ref_buffer_dir'],\
                       save_env_ref_buffer_name=args['save_env_ref_buffer_name'],\
                       load_env_ref_buffer=args['load_env_ref_buffer'],\
                       load_all_env_ref_buffer_dir=args['load_all_env_ref_buffer_dir'])

    # changing environment gg map
    # env.set_car('Gokart')

    state_dim = 4  # [vx, vy, posx, posy]
    action_dim = 1  # szam (fok) ami azt jelenti hogy a gg diagramon melikiranyba gyorsulunk
    action_bound = 180  # 0: egyenesen -180,180: fék, -90: jobbra kanyar
    # Ensure action bound is symmetric
    # assert (env.action_space.high == -env.action_space.low)

    agent = Agent.ActorCritic(state_dim, action_dim, action_bound, used_device, float(args['actor_lr']), \
                              float(args['critic_lr']), float(args['tau']), float(args['gamma']), \
                              log_dir= args['summary_dir'], network_dir=args['network_dir'])

    # setting random seed
    agent.set_random_seed(int(args['random_seed']))
    # also for the environment
    np.random.seed(int(args['random_seed']))

    # Initialize replay memory
    replay_buffer = ReplayBuffer(buffer_size = int(args['buffer_size']), \
                                 random_seed = int(args['random_seed']))

    replay_loaded = replay_buffer.load(load_file=args['load_experince_name'], \
                       load_all_dir=args['load_all_experince_dir'])
    print('replay loaded: ' + str(replay_loaded))

    play_train(env, agent, replay_buffer,\
          max_episodes=int(args['max_episodes']),\
          max_episode_len = int(args['max_episode_len']),\
          minibatch_size = int(args['minibatch_size']),\
          show_window=args['show_display'],\
          save_image_episodes = int(args['save_image_episodes']),\
          save_graph_episodes=int(args['save_graph_episodes']),\
          actor_noise = 'default')

    # replay_buffer.save(save_dir = args['save_experience_dir'], \
    #                   save_name = args['save_experience_name'])

    # cleaning
    replay_buffer.clear()
    env.clean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate',   default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.000005)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.998)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1500000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=32)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='pyperrace')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=12131)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=2000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=40)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results')
    parser.add_argument('--save-experience-dir', help='directory for saving experiences', default='./experience')
    parser.add_argument('--save-experience-name', help='name for saving experience file', default='experience.npz')
    parser.add_argument('--load-experince-name', help='loading experience setting', default='./experience/experience.npz')
    parser.add_argument('--load-all-experince-dir', help='loading all experience from this directory', default='./experience')
    parser.add_argument('--network-dir', help='saving networks to this folder', default='./network')
    parser.add_argument('--save-graph-episodes', help='save graph in every x epides', default=1000)
    parser.add_argument('--save-image-episodes', help='save image in every x epides', default=100)
    parser.add_argument('--show-display', help='show env in window', default='')



    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)