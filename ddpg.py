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
import random as rnd

from replay_buffer import ReplayBuffer


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        with tf.device(used_device):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.action_bound = action_bound
            self.learning_rate = learning_rate
            self.tau = tau

            self.sess = sess

            # Actor Network
            self.inputs, self.out, self.scaled_out = self.create_actor_network(scope = 'actor')

            self.network_params = tf.trainable_variables()

            # Target Network
            self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network(scope = 'actor_target')

            self.target_network_params = tf.trainable_variables()[
                                         len(self.network_params):]

            # Op for periodically updating target network with online network
            # weights
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim], name='actor_action_grad')

            # Combine the gradients here
            # TODOdone:  miért minus az action gradient?
            # http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
            self.actor_gradients = tf.gradients(
                self.scaled_out, self.network_params, -self.action_gradient, name='actor_grads')

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
                apply_gradients(zip(self.actor_gradients, self.network_params), name='actor_optimize')

            self.num_trainable_vars = len(
                self.network_params) + len(self.target_network_params)

            #initialise variables
            #init = tf.global_variables_initializer()
            #self.sess.run(init)

            #writer = tf.summary.FileWriter(args['summary_dir'], self.sess.graph)
            #writer.close()

    def create_actor_network(self, scope = 'actor'):
        with tf.name_scope(scope):
            inputs = tflearn.input_data(shape=[None, self.state_dim], name='actor_input')
            net1 = tflearn.fully_connected(inputs, 400, name='actor_fc1')
            net2 = tflearn.layers.normalization.batch_normalization(net1, name='actor_norm1')
            net3 = tflearn.activations.relu(net2)
            net4 = tflearn.fully_connected(net3, 300, name='actor_fc2')
            net5 = tflearn.layers.normalization.batch_normalization(net4, name='actor_norm2')
            net6 = tflearn.activations.relu(net5)
            """ 
            net = tflearn.fully_connected(net, 30)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            net = tflearn.fully_connected(net, 30)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            """
            # Final layer weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out = tflearn.fully_connected(
                net6, self.action_dim, activation='tanh', weights_init=w_init, name='actor_output')
            # Scale output to -action_bound to action_bound

            scaled_out = tf.multiply(out, self.action_bound)
            # scaled_out = np.sign(out)
            return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def save(self, path):
        self.saver.save(self.sess, path)

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        with tf.device(used_device):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.learning_rate = learning_rate
            self.tau = tau
            self.gamma = gamma

            self.sess = sess

            # Create the critic network

            #self.sess = tf.Session(graph = self.graph)

            self.inputs, self.action, self.out = self.create_critic_network(scope = 'critic')

            self.network_params = tf.trainable_variables()[num_actor_vars:]

            # Target Network
            self.target_inputs, self.target_action, self.target_out = self.create_critic_network(scope = 'critic_target')

            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

            # Op for periodically updating target network with online network
            # weights with regularization
            self.update_target_network_params = \
                [self.target_network_params[i].assign(
                    tf.add(tf.multiply(self.network_params[i], self.tau),
                           tf.multiply(self.target_network_params[i], 1. - self.tau, name='mult_params_'+str(i)),
                           name = 'add_params_'+str(i)))
                    for i in range(len(self.target_network_params))]

            # Network target (y_i)
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name='critc_predicted_q')

            # Define loss and optimization Op
            self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
            self.optimize = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)

            # Get the gradient of the net w.r.t. the action.
            # For each action in the minibatch (i.e., for each x in xs),
            # this will sum up the gradients of each critic output in the minibatch
            # w.r.t. that action. Each output is independent of all
            # actions except for one.
            self.action_grads = tf.gradients(self.out, self.action, name='critic_action_grads')

            #initialise variables
            #init = tf.global_variables_initializer()
            #self.sess.run(init)

            #writer = tf.summary.FileWriter(args['summary_dir'], self.sess.graph)
            #writer.close()

    def create_critic_network(self, scope = 'critic'):
        with tf.name_scope(scope):
            inputs = tflearn.input_data(shape=[None, self.state_dim], name='critic_input')
            net = tflearn.fully_connected(inputs, 400, name='critic_fc1')
            net = tflearn.layers.normalization.batch_normalization(net, name='critic_norm1')
            net = tflearn.activations.relu(net)
            t1 = tflearn.fully_connected(net, 300, name='critic_fc2')

            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            action = tflearn.input_data(shape=[None, self.action_dim], name='critic_action_input')
            t2 = tflearn.fully_connected(action, 300, name='critic_norm2')
            add_t2 = tf.add(tf.matmul(action, t2.W), t2.b, name='critic_t2_add')

            net = tflearn.activation(tf.add(tf.matmul(net, t1.W), add_t2), activation='relu', name='critic_relu')
            """
            net = tflearn.fully_connected(net, 90)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
    
            net = tflearn.fully_connected(net, 40)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
    
            net = tflearn.fully_connected(net, 20)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            """
            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out = tflearn.fully_connected(net, 1, weights_init=w_init, name='critic_output')
            # self.model = model = tflearn.DNN(out)
            return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        with tf.variable_scope('critic'):
            return self.sess.run([self.out, self.optimize], feed_dict={
                self.inputs: inputs,
                self.action: action,
                self.predicted_q_value: predicted_q_value
            })

    def predict(self, inputs, action):
        with tf.variable_scope('critic'):
            return self.sess.run(self.out, feed_dict={
                self.inputs: inputs,
                self.action: action
            })

    def predict_target(self, inputs, action):
        with tf.variable_scope('critic'):
            return self.sess.run(self.target_out, feed_dict={
                self.target_inputs: inputs,
                self.target_action: action
            })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.load(self.sess, path)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0., name='reward')
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0., name='qmax')
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise, replay_buffer):
    max_episodes = int(args['max_episodes'])
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    print("summaries built")

    sess.run(tf.global_variables_initializer())

    # to save all 100th
    saver = tf.train.Saver(max_to_keep=0)

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
    draws = 1

    # draw_config = 'perxepisode'
    # draws = 100

    # draw_config = 'maxdrawsx'
    # draws_per_fullepisodes = max(1, max_episodes / draws_per_fullepisodes)

    # draws_config = 'drawnothing'
    # draws = 0

    # where to draw
    draw_where = {'window': True, 'file': True}

    # ----------------------------

    # A koncepcio az lesz hogy generalunk egy 0-1 kozotti szamot, aminel majd kell random szamnak kisebbnek kell
    # lenni es akkor teljesul egy feltetel

    # de eloszor is a tanitasra szant epizodok elso valahany %-aban nagyon random lepked. Ilyenkor meg nem is kene
    # tanulni, csak tolteni fel az exerience memoryt
    rand_ep_for_exp = max_episodes * 0.02

    # egy masfele random baszakodas
    rand_ep_for_exp2 = range(int(0.01 * max_episodes), int(0.011 * max_episodes))
    rand_ep_for_exp3 = range(int(0.012 * max_episodes), int(0.013 * max_episodes))

    # Jani véletlenszerű lépés tanulás közben arány
    rand_stp_normal = 0.1
    # a minimum random amivel a teljes tanulas alatt neha random lep, megha mar a vegen is vagyunk:
    rand_stp_min = 0.001

    # ====================
    # Indul egy epizod:regrgterg
    # ====================

    for i in range(max_episodes):
        # alapállapotba hozzuk a környezetet
        env.reset()

        # kezdeti sebeesseg, ahogy a kornyezet adja
        v = np.array(env.starting_spd)

        # sebesség mellé a kezdeti poz. is kell. Ez a kezdőpozíció beállítása:
        pos = np.array(env.starting_pos)

        # kezdeti teljes epzód alatt szerzett jutalom, és legjobb q étrék:
        ep_reward = 0
        ep_ave_max_q = 0

        # ------------------kornyezet kirajzolasahoz---------------------------------

        # draw in this episode
        if i%draws == 0:
            draw = True
        else:
            draw = False

        # ---------------------------------------------------------------------------

        # drawing
        if draw:
            env.draw_clear()
            env.draw_track()

        # Exploration-joz: Ha mas nincs, ne veletlenszeruen lepkedjen
        rand_episode = False
        # random lesz egesz epizod, ha a tanulas elejen vagyunk:
        #rand_episode2 = i < rand_ep_for_exp
        #print("Random Episode")
        # !: később intézve hogy ilyenkor ne tanuljon, csak töltse a memoryt

        #rand_episode = (i in rand_ep_for_exp2) or (i in rand_ep_for_exp3)
        #print("Random Episode2:", rand_episode2)

        # aztan kesobb, az epizodok elorehaladtaval, csokkeno valoszinuseggel, random lepesek
        rand_stp_for_exp = (max_episodes - (100 * i)) / max_episodes
        print("Random Step", rand_stp_for_exp)

        #ennyiedik leestol kezdve random lesz a lepes:
        lepestol = rnd.uniform(0, env.ref_actions.size * (100*i) / max_episodes)

        #egy egy epizódon belül ennyi lépés van maximum:
        for j in range(int(args['max_episode_len'])):
            step_color = (1, 0, 0)

            s = [v[0], v[1], pos[0], pos[1]] #az eredeti kodban s-be van gyujtve az ami a masikban pos és v

            rand_step = False
            # Az egy dolog hogy az elejen van egy darabig total random lepkedes, de utana is van hogy neha randomlep
            # Minnnel kesobb jarunk a tanulanal, annal kisebb valoszinuseggel. Eleinte meg naaagy valoszinuseggel.
            # Tovabab ugyanitt kezelve hogy mikor lejar a csokkeno valoszinusegu resz utana is meg neha rand legyen
            if (rnd.uniform(0, 1) < rand_stp_normal):
                rand_step = True


            #Actionok:
            # Ha az adott felteltel teljesult korabban, es most egy random epizodban vagyunk, vagy nem random az epizod,
            # de a lepes random, na akkor randomot lepunk:
            if i < rand_ep_for_exp:

                if (lepestol < j) and (j < env.ref_actions.size):
                    a = int(np.random.normal(env.ref_actions[j], 20, size=1))
                    step_color = (1, 0.5, 0)
                    print("\033[94m {}\033[00m".format("        -------ref norm rand action:"), a)
                else:#hanyadik lepestol lepunk random: (hogy az elejen meg lehetoleg a referenciat lepje)
                    if j < env.ref_actions.size:
                        a = env.ref_actions[j] # int(actor.predict(np.reshape(s, (1, actor.state_dim))))
                        step_color = (0, 0, 1)
                        print("\033[93m {}\033[00m" .format("        -------ref action:"), a)
                    else:
                        a = int(rnd.uniform(-180, 180))
                        step_color = (0, 1, 0)
                        print("\033[92m {}\033[00m".format("        -------uni rand action:"), a)

            # A tanulasra szant epizodok soran nem sak az elejen hanem kesobb is legyen egy kis "belezavaras"
            #"""
            #elif (i in rand_ep_for_exp2) or (i in rand_ep_for_exp3): # or rand_step2:
             #   if j < env.ref_actions.size:
             #       a = int(rnd.uniform(-180, 180))
            # x       #a = int(np.random.normal(env.ref_actions[j], 120, size=1))
             ##       print("\033[94m {}\033[00m".format("        -------Rnd2 RndUni action:"), a)
             #   else:
             #       a = int(actor.predict(np.reshape(s, (1, actor.state_dim))))
            #        print("\033[94m {}\033[00m".format("        -------Rnd2 Net action:"), a)
            #    # ha semmifeltetel a fentiekbol nem teljesul, akkor meg a halo altal mondott lepest lepjuk
            #"""
            # Jani random lépés
            elif (rand_step is True):
                step_color = (1, 1, 0)
                a = int(rnd.uniform(-180, 180))
                print("\033[94m {}\033[00m".format("        -------full random action:"), a)
            else:
                a = int(actor.predict(np.reshape(s, (1, actor.state_dim))) + 0 * actor_noise()) + int(
                    np.random.randint(-3, 3, size=1))
                step_color = (1, 0, 0)
                print("Netwrk action:--------", a)

            a = max(min(a, 180), -180)
            gg_action = env.gg_action(a)  # action-höz tartozó vektor lekérése
            #általában ez a fenti két sor egymsor. csak nálunk most így van megírva a környezet, hogy így kell neki beadni az actiont
            #megnézzük mit mond a környezet az adott álapotban az adott action-ra:
            #s2, r, terminal, info = env.step(a)
            v_new, pos_new, reward, end, section_nr = env.step(gg_action, v, pos, draw, draw_text='little_reward',\
                                                               color=step_color)
            t_diff = env.get_time_diff(pos, pos_new, reward, end)
            #megintcsak a kétfelől összemásolgatott küdok miatt, feleltessünkk meg egymásnak változókat:
            s2 = [v_new[0], v_new[1], pos_new[0], pos_new[1]]
            r = t_diff
            terminal = end
            ep_reward += r

            #és akkor a megfeleltetett változókkal már lehet csinálni a replay memory-t:
            replay_buffer.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), r, terminal, np.reshape(s2, (actor.state_dim,)))

            # Keep adding experience to the memory until there are at least minibatch size samples, És amig a
            # tanulas elejen a random lepkedos fazisban vagyunk.
            if replay_buffer.size() > int(args['minibatch_size']): # and not rand_episode:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

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

            if terminal:
                #Ha egybol (J=0-nal vege)
                if j == 0:
                    j = 1
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                break
            # end of if terminal
        # end of steps
        print('| Reward: {:.3f} | Episode: {:d} | Qmax: {:.4f}'.format(ep_reward, i, (ep_ave_max_q / float(j))))

        # minden századik epizód után legyen mentés
        if i % args['save_graph_episodes'] == 0:
            saver.save(sess, args['network_dir'] + '/full_network_e' + str(i) + '.tfl')
        # minden x epizód után legyen kép mentés
        if i % args['save_image_episodes'] == 0:
            if draw_where['file']:
                env.draw_info(1300,800, 'reward'+ str(ep_reward))
                env.draw_save(name='e', count=str(i))


def main(args):
    with tf.Session() as sess:

        #env = gym.make(args['env'])

        trk_col = np.array([99, 99, 99])  # pálya színe (szürke), a kornyezet inicializalasahoz kell

        #sections = np.array([[350,  60, 350, 100],
        #                    [425, 105, 430, 95],
        #                    [500, 140, 530, 110],
        #                    [520, 160, 580, 150]])
        #                     [ 35, 200,  70, 200],
        #                     [250,  60, 250, 100]])
        # env = PaperRaceEnv('PALYA3.bmp', trk_col, 'GG1.bmp', start_line, random_init=False)

        # palya4 próbálkozások
        #sections = np.array([[273, 125, 273,  64],
        #                     [347, 125, 347,  65],

        #sections = np.array([[273, 125, 273, 64], # [333, 125, 333, 64],[394, 157, 440, 102],[370, 195, 440, 270],[331, 212, 331, 267]] [220, 300, 280, 300]])
        #                     [240, 400, 300, 380]])
        #                    #[190, 125, 190, 64]])

        # palya4 teljes
        # sections = np.array([[273, 125, 273, 64],  # [333, 125, 333, 64],[394, 157, 440, 102],[240, 400, 330, 380]
        #                     [100, 250, 180, 250]])
        # env = PaperRaceEnv('PALYA4.bmp', trk_col, 'GG1.bmp', sections, random_init=False)

        # palya5.bmp-hez:
        # sections = np.array([[670, 310, 670, 130],  # [333, 125, 333, 64],[394, 157, 440, 102],
        #                     [1250, 680, 1250, 550]])
        # env = PaperRaceEnv('PALYA5.bmp', trk_col, 'GG1.bmp', sections, random_init=False)

        # palya h1.bmp
        # csak az eleje meg a vége
        # Jani
        # sections = np.array([[150, 10, 150, 250], [150, 1240, 210, 1430]])
        # Gergo
        sections = np.array([[200, 220, 200, 50],  # [333, 125, 333, 64],[394, 157, 440, 102],
                             [200, 1250, 250, 1400]])
        env = PaperRaceEnv('h1.bmp', trk_col, 'GG1.bmp', sections, random_init=False, \
                           save_env_ref_buffer_dir=args['save_env_ref_buffer_dir'],\
                           save_env_ref_buffer_name=args['save_env_ref_buffer_name'],\
                           load_env_ref_buffer=args['load_env_ref_buffer'],\
                           load_all_env_ref_buffer_dir=args['load_all_env_ref_buffer_dir'])

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        #env.seed(int(args['random_seed']))

        state_dim = 4 #[vx, vy, posx, posy]
        action_dim = 1 #szam (fok) ami azt jelenti hogy a gg diagramon melikiranyba gyorsulunk
        action_bound = 180 #0: egyenesen -180,180: fék, -90: jobbra kanyar
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']))

        print("actor created")

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        print("critic created")

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        print("actor noise created")

        # Initialize replay memory
        replay_buffer = ReplayBuffer(buffer_size = int(args['buffer_size']), \
                                     random_seed = int(args['random_seed']))

        replay_buffer.load(load_file=args['load_experince_name'], \
                           load_all_dir=args['load_all_experince_dir'])

        train(sess, env, args, actor, critic, actor_noise, replay_buffer)

        replay_buffer.save(save_dir = args['save_experience_dir'], \
                           save_name = args['save_experience_name'])

        #cleaning
        replay_buffer.clear()
        env.clean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate',   default=0.00001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.998)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1500000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=128)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='pyperrace')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=12131)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=10)
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
    parser.add_argument('--save-graph-episodes', help='save graph in every x epides', default=1000)
    parser.add_argument('--save-image-episodes', help='save image in every x epides', default=5)


    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)

    #TODO: elejen az elsp lepesek tok random legyenek kicsit, es utana csak a ref lepessorhoz
    #TODO: csinalni valami alap ref lepessor csinalo algoritmust (pretrain, esetleg?)
    #TODO: reward vissza az egyszeru felere. -1 minden lepes, csak kieseskor lehet a hatralevo tav