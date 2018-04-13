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
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0., name='reward')
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0., name='qmax')
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    max_episode_reward = tf.Variable(0., name='reward_max')
    tf.summary.scalar("Max episode reward Value", max_episode_reward)

    summary_vars = [episode_reward, episode_ave_max_q, max_episode_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, actor, critic, replay_buffer, max_episodes, max_episode_len, minibatch_size, show_window, save_image_episodes, actor_noise = 'not implemented'):
# Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    print("summaries built")

    sess.run(tf.global_variables_initializer())

    # to save all 100th
    saver = tf.train.Saver(max_to_keep=2)

    # writer = tf.summary.FileWriter((args['summary_dir']))
    writer = tf.summary.FileWriter(logdir = args['summary_dir'], graph = sess.graph)
    # writer.close()

    # Initialize target network weights
    actor.update_target_network()
    print("target actor initialised")
    critic.update_target_network()
    print("target critic initialised")

    # ----------------------------

    # nem minden epizodot fogunk kirajzolni, mert lassú. Lásd később
    # draws = 1

    # osszes tanulas alatt ennyiszer rajzolunk:

    # draw_config = 'allepisode'
    draws = save_image_episodes

    # draw_config = 'perxepisode'
    # draws = 100

    # draw_config = 'maxdrawsx'
    # draws_per_fullepisodes = max(1, max_episodes / draws_per_fullepisodes)

    # draws_config = 'drawnothing'
    # draws = 0

    # where to draw
    draw_where = {'window': True, 'file': True}

    # ----------------------------

    # az emberi jatekokat bele kell "keverni" majd, mint experience. Hogy a teljes tanitásra szánt epizodok alatt
    # mikor, az a lenti matrixban dol el. Minden sor egy szakaszt jelöl, amiben exploration van:

    ep_for_exp = np.array([0, 0.10,
                           0.15, 0.25,
                           0.35, 0.45]) * int(args['max_episodes'])

    # Minden sor szam pedig hogy abban a fentiekben megadott intervallumokban mennyiről mennyire csökkenjen a szórás.
    sig_for_exp = np.array([5, 0,
                            10, 0,
                            20, 0])

    #Jani véletlenszerű lépés tanulás közben arány
    rand_stp_normal = 0.001

    # store steps in it
    episode_steps = []

    best_reward = -100.0

    # ====================
    # Indul egy epizod:
    # ====================

    for i in range(max_episodes):
        # ------------------kornyezet kirajzolasahoz---------------------------------

        # draw in this episode
        if i % draws == 0 or show_window == 'allstep':
            draw = True
        else:
            draw = False

        # alapállapotba hozzuk a környezetet
        env.reset(draw)

        # kezdeti sebeesseg, ahogy a kornyezet adja
        v = np.array(env.starting_spd)

        # sebesség mellé a kezdeti poz. is kell. Ez a kezdőpozíció beállítása:
        pos = np.array(env.starting_pos)

        # kezdeti teljes epzód alatt szerzett jutalom, és legjobb q étrék:
        ep_reward = 0
        ep_ave_max_q = 0
        # legjobb elert reward


        # ---------------------------------------------------------------------------


        """
        # aztan kesobb, az epizodok elorehaladtaval, csokkeno valoszinuseggel, random lepesek
        rand_stp_for_exp = (max_episodes - (100 * i)) / max_episodes
        print("Random Step", rand_stp_for_exp)

        rand_stp_for_exp = (int(args['max_episodes']) - (100 * i)) / int(args['max_episodes'])
        print("Random Step: ", rand_stp_for_exp)
        
        #ennyiedik leestol kezdve random lesz a lepes:
        lepestol = rnd.uniform(0, env.ref_actions.size * (100*i) / max_episodes)
        """
        # az emberi lepessorok kozul valasszunk egyet veletlenszeruen mint aktualis epizod lepessor:
        #curr_ref_actions = np.array(env.hum_act[int(np.random.uniform(0, int(len(env.hum_act)), 1))])
        #TODO move it to environment
        curr_ref_actions = env.get_ref_actions()

        lepestol = np.random.uniform(0, curr_ref_actions.size * (100*i) / max_episodes, 1)




        # a random lepesekhez a szoras:
        szoras = np.interp(i, ep_for_exp, sig_for_exp)

        lepesek = []

        #egy egy epizódon belül ennyi lépés van maximum:
        for j in range(max_episode_len):
            step_color = (1, 0, 0)

            s = [v[0], v[1], pos[0], pos[1]] #az eredeti kodban s-be van gyujtve az ami a masikban pos és v

            rand_step = False
            # Az egy dolog hogy az elejen van egy darabig total random lepkedes, de utana is van hogy neha randomlep
            # Minnnel kesobb jarunk a tanulanal, annal kisebb valoszinuseggel. Eleinte meg naaagy valoszinuseggel.
            # Tovabab ugyanitt kezelve hogy mikor lejar a csokkeno valoszinusegu resz utana is meg neha rand legyen
            if (np.random.uniform(0, 1, 1) < rand_stp_normal):
                rand_step = True

            #Actionok:
            # Ha i (epizod) abban atartományban van amikor emberi lepes alapu random epizodot akarunk akkor:
            rand_episode = (i in range(int(ep_for_exp[0]), int(ep_for_exp[1]))) or (
                    i in range(int(ep_for_exp[2]), int(ep_for_exp[3]))) or (
                    i in range(int(ep_for_exp[4]), int(ep_for_exp[5])))
            if rand_episode:
                # ha nem ért még véget az epizod, de mar a ref lepessor vege, akkor random lepkedunk
                if j < curr_ref_actions.size:
                    a = int(np.random.normal(curr_ref_actions[j], szoras, 1))  # int(actor.predict(np.reshape(s, (1, actor.state_dim))))
                    print("\033[93m {}\033[00m".format("        -------ref action:"), a)
                    step_color = (0, 0, 1)
                else:
                    a = int(np.random.uniform(-180, 180, 1))
                    step_color = (0, 1, 0)
                    print("\033[92m {}\033[00m".format("        -------uni rand action:"), a)

                if (lepestol < j) and (j < curr_ref_actions.size):
                    a = int(np.random.normal(curr_ref_actions[j], 20, size=1))
                    step_color = (1, 0.5, 0)
                """
                 # a referencia lepessortol elterunk ha az aktualis lepes a kivant tartomanyba esik az epizodon belul
                if (lepestol < j) and (j < curr_ref_actions.size):
                    a = int(np.random.normal(curr_ref_actions[j], 10, 1))
                    print("\033[94m {}\033[00m".format("        -------ref norm rand action:"), a)
                else:#hanyadik lepestol lepunk random: (hogy az elejen meg lehetoleg a referenciat lepje)
                    if j < env.ref_actions.size:
                        a = env.ref_actions[j] # int(actor.predict(np.reshape(s, (1, actor.state_dim))))
                        step_color = (0, 0, 1)
                        print("\033[93m {}\033[00m" .format("        -------ref action:"), a)
                    else:
                        a = int(rnd.uniform(-180, 180))
                        print("\033[92m {}\033[00m".format("        -------uni rand action:"), a)
                """

            # Jani random lépés
            elif (rand_step is True):
                step_color = (1, 1, 0)
                a = int(np.random.uniform(-180, 180))
                print("\033[94m {}\033[00m".format("        -------full random action:"), a)
            else:
                # a = int(actor.predict(np.reshape(s, \
                #                                 (1, actor.state_dim))) + 0 * actor_noise()) + int(np.random.randint(-3, 3, size=1))
                a = int(actor.predict(np.reshape(s, \
                                                 (1, actor.state_dim)))) + int(np.random.randint(-3, 3, size=1))
                step_color = (1, 0, 0)
                print("Netwrk action:--------", a)

            a = max(min(a, 180), -180)
            gg_action = env.gg_action(a)  # action-höz tartozó vektor lekérése
            #általában ez a fenti két sor egymsor. csak nálunk most így van megírva a környezet, hogy így kell neki beadni az actiont
            #megnézzük mit mond a környezet az adott álapotban az adott action-ra:
            #s2, r, terminal, info = env.step(a)
            v_new, pos_new, reward, end, section_nr = env.step(gg_action, v, pos, draw, \
                                                               draw_text='little_reward',\
                                                               color=step_color)
            t_diff = env.get_time_diff(pos, pos_new, reward, end)
            #megintcsak a kétfelől összemásolgatott küdok miatt, feleltessünkk meg egymásnak változókat:
            s2 = [v_new[0], v_new[1], pos_new[0], pos_new[1]]
            r = t_diff
            terminal = end
            ep_reward += r

            #és akkor a megfeleltetett változókkal már lehet csinálni a replay memory-t:
            replay_buffer.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), r, terminal, \
                              np.reshape(s2, (actor.state_dim,)))

            # Keep adding experience to the memory until there are at least minibatch size samples, És amig a
            # tanulas elejen a random lepkedos fazisban vagyunk.
            if replay_buffer.size() > int(minibatch_size): # and not rand_episode:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(minibatch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (minibatch_size, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                #TODOdone: emiatt lassu, nem biztos
                # gradienseket ezzel kiolvassa a tensorflow graph-ból és visszamásolja
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            #a kovetkezo lepeshez uj s legyen egyenlo az aktualis es folytatjuk
            #s = s2
            v = v_new
            pos = pos_new

            lepesek.append(a)
            if terminal:
                #Ha egybol (J=0-nal vege)
                if j == 0:
                    j = 1
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: best_reward
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                break
            # end of if terminal
        # end of steps
        episode_steps.append([i, ep_reward, lepesek])

        print('| Reward: {:.3f} | Episode: {:d} | Qmax: {:.4f}'.format(ep_reward, i, (ep_ave_max_q / float(j))))

        # minden századik epizód után legyen mentés
        if i % args['save_graph_episodes'] == 0:
            saver.save(sess, args['network_dir'] + '/full_network_e' + str(i) + '.tfl')

            #legjobb mentese
            sorted_list = sorted(episode_steps, key=lambda x: x[1])[-1:]
            print("best episode:")
            print(sorted_list)
            best_reward = sorted_list[0][1]
            # writing best to file
            with open('best.stp', 'w') as file:
                file.write(str(sorted_list[0][1])+"\033")
                for k in range(len(sorted_list[0][2])):
                    file.write(str(sorted_list[0][2][k])+"\033")

        # minden x epizód után legyen kép mentés
        if draw:
            if draw_where['file']:
                env.draw_info(400, \
                              1450, \
                              '| Reward: {:.3f} | Episode: {:d} | Qmax: {:.4f}'.format(ep_reward, \
                                                                                       i, \
                                                                                       (ep_ave_max_q / float(j))))
                env.draw_save(name='e', count=str(i))


def main(args):
    with tf.Session() as sess:

        # GG1.bmp is used for reward function
        env = PaperRaceEnv('h1', ref_calc = 'default', car_name='Touring', random_init=False, \
                           save_env_ref_buffer_dir=args['save_env_ref_buffer_dir'],\
                           save_env_ref_buffer_name=args['save_env_ref_buffer_name'],\
                           load_env_ref_buffer=args['load_env_ref_buffer'],\
                           load_all_env_ref_buffer_dir=args['load_all_env_ref_buffer_dir'])

        # changing environment gg map
        env.set_car('Gokart')

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        #env.seed(int(args['random_seed']))

        state_dim = 4 #[vx, vy, posx, posy]
        action_dim = 1 #szam (fok) ami azt jelenti hogy a gg diagramon melikiranyba gyorsulunk
        action_bound = 180 #0: egyenesen -180,180: fék, -90: jobbra kanyar
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        actor = Agent.ActorNetwork(sess, used_device, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']))

        print("actor created")

        critic = Agent.CriticNetwork(sess, used_device, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        print("critic created")

        actor_noise = Agent.OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        print("actor noise created")

        # Initialize replay memory
        replay_buffer = ReplayBuffer(buffer_size = int(args['buffer_size']), \
                                     random_seed = int(args['random_seed']))

        replay_buffer.load(load_file=args['load_experince_name'], \
                           load_all_dir=args['load_all_experince_dir'])

        train(sess, env, actor, critic, replay_buffer,
              max_episodes=int(args['max_episodes']),
              max_episode_len = int(args['max_episode_len']),
              minibatch_size = int(args['minibatch_size']),
              show_window=args['show_display'],
              save_image_episodes = int(args['save_image_episodes']),
              actor_noise = actor_noise
              )

        replay_buffer.save(save_dir = args['save_experience_dir'], \
                           save_name = args['save_experience_name'])

        #cleaning
        replay_buffer.clear()
        env.clean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate',   default=0.0003)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.0005)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.998)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1500000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=32)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='pyperrace')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=12131)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=102)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=40)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results')
    parser.add_argument('--save-experience-dir', help='directory for saving experiences', default='./experience')
    parser.add_argument('--save-experience-name', help='name for saving experience file', default='experience.npz')
    parser.add_argument('--load-experince-name', help='loading experience setting', default='./experience/experience.npz')
    parser.add_argument('--load-all-experince-dir', help='loading all experience from this directory', default='./experience')
    parser.add_argument('--network-dir', help='saving networks to this folder', default='./network')
    parser.add_argument('--save-env-ref-buffer-dir', help='saving and loading ref buffer from this dir', default='./env_ref_buffer')
    parser.add_argument('--save-env-ref-buffer-name', help='saving and loading ref buffer from this dir', default='env_ref_buffer_1')
    parser.add_argument('--load-env-ref-buffer', help='load env buffer  from this folder', default='./env_ref_buffer/env_ref_buffer_1')
    parser.add_argument('--load-all-env-ref-buffer-dir', help='saving networks to this folder', default='./env_ref_buffer')
    parser.add_argument('--save-graph-episodes', help='save graph in every x epides', default=100)
    parser.add_argument('--save-image-episodes', help='save image in every x epides', default=1)
    parser.add_argument('--show-display', help='show env in window', default='allstep')


    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)