# ===========================
#   Actor and Critic DNNs
# ===========================

import tensorflow as tf
import numpy as np
import tflearn

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, device, state_dim, action_dim, action_bound, learning_rate, tau):
        with tf.device(device):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.action_bound = action_bound
            self.learning_rate = learning_rate
            self.tau = tau

            self.sess = sess

            # Actor Network
            self.inputs, self.out, self.scaled_out = self.create_actor_network(scope='actor')

            self.network_params = tf.trainable_variables()

            # Target Network
            self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network(
                scope='actor_target')

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

            # initialise variables
            # init = tf.global_variables_initializer()
            # self.sess.run(init)

            # writer = tf.summary.FileWriter(args['summary_dir'], self.sess.graph)
            # writer.close()

    def create_actor_network(self, scope='actor'):
        with tf.name_scope(scope):
            inputs = tflearn.input_data(shape=[None, self.state_dim], name='actor_input')
            net1 = tflearn.fully_connected(inputs, 400, name='actor_fc1')
            net2 = tflearn.layers.normalization.batch_normalization(net1, name='actor_norm1')
            net3 = tflearn.activations.relu(net2)
            net4 = tflearn.fully_connected(net3, 100, name='actor_fc2')
            net5 = tflearn.layers.normalization.batch_normalization(net4, name='actor_norm2')
            net6 = tflearn.activations.relu(net5)
            net7 = tflearn.fully_connected(net6, 30, name='actor_fc3')
            net8 = tflearn.layers.normalization.batch_normalization(net7, name='actor_norm3')
            net9 = tflearn.activations.relu(net8)
            net10 = tflearn.fully_connected(net9, 10, name='actor_fc4')
            net11 = tflearn.layers.normalization.batch_normalization(net10, name='actor_norm4')
            net12 = tflearn.activations.relu(net11)
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
                net12, self.action_dim, activation='tanh', weights_init=w_init, name='actor_output')
            # Scale output to -action_bound to action_bound

            scaled_out = tf.multiply(out, self.action_bound)
            # scaled_out = np.sign(out)
            return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs, add_uncertainity = True):
        prediction =  self.sess.run(self.scaled_out, feed_dict={
                self.inputs: inputs})

        if add_uncertainity:
            return prediction + self.uncertanity()

        return prediction


    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    # for learning
    def uncertanity(self):
        return np.random.randint(-3, 3, size=1)

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, device, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        with tf.device(device):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.tau = tau
            self.gamma = gamma

            self.sess = sess

            # Create the critic network

            # self.sess = tf.Session(graph = self.graph)

            self.inputs, self.action, self.out = self.create_critic_network(scope='critic')

            self.network_params = tf.trainable_variables()[num_actor_vars:]

            # Target Network
            self.target_inputs, self.target_action, self.target_out = self.create_critic_network(scope='critic_target')

            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

            # Op for periodically updating target network with online network
            # weights with regularization
            self.update_target_network_params = \
                [self.target_network_params[i].assign(
                    tf.add(tf.multiply(self.network_params[i], self.tau),
                           tf.multiply(self.target_network_params[i], 1. - self.tau, name='mult_params_' + str(i)),
                           name='add_params_' + str(i)))
                    for i in range(len(self.target_network_params))]

            # Network target (y_i)
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name='critc_predicted_q')

            self.learning_rate = learning_rate
            self.tf_learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            # Define loss and optimization Op
            self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
            self.optimize = tf.train.AdamOptimizer(
                learning_rate=self.tf_learning_rate).minimize(self.loss)

            # Get the gradient of the net w.r.t. the action.
            # For each action in the minibatch (i.e., for each x in xs),
            # this will sum up the gradients of each critic output in the minibatch
            # w.r.t. that action. Each output is independent of all
            # actions except for one.
            self.action_grads = tf.gradients(self.out, self.action, name='critic_action_grads')

            # initialise variables
            # init = tf.global_variables_initializer()
            # self.sess.run(init)

            # writer = tf.summary.FileWriter(args['summary_dir'], self.sess.graph)
            # writer.close()

    def create_critic_network(self, scope='critic'):
        with tf.name_scope(scope):
            inputs = tflearn.input_data(shape=[None, self.state_dim], name='critic_input')
            net = tflearn.fully_connected(inputs, 300, name='critic_fc1')
            net = tflearn.layers.normalization.batch_normalization(net, name='critic_norm1')
            net = tflearn.activations.relu(net)
            t1 = tflearn.fully_connected(net, 200, name='critic_fc2')

            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            action = tflearn.input_data(shape=[None, self.action_dim], name='critic_action_input')
            t2 = tflearn.fully_connected(action, 200, name='critic_norm2')
            add_t2 = tf.add(tf.matmul(action, t2.W), t2.b, name='critic_t2_add')

            net = tflearn.activation(tf.add(tf.matmul(net, t1.W), add_t2), activation='relu', name='critic_relu')

            net = tflearn.fully_connected(net, 90)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

            net = tflearn.fully_connected(net, 40)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

            net = tflearn.fully_connected(net, 20)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

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
                self.tf_learning_rate: self.learning_rate,
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

class ActorCritic(object):
    def __init__(self, state_dim, action_dim, action_bound, device, actor_lr, critic_lr, tau, \
                 gamma, log_dir, network_dir, actor_noise = 'default'):
        self.sess = tf.Session()
        #with tf.Session() as self.sess:

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_dir = log_dir
        self.network_dir = network_dir

        if actor_noise == 'default':
            self.actor_noise = False
            # currently not using in network noise

        self.actor = ActorNetwork(self.sess, device, state_dim, action_dim, action_bound,
                                   actor_lr, tau)

        print("actor created")

        self.critic = CriticNetwork(self.sess, device, state_dim, action_dim,
                                     critic_lr, tau, gamma,
                                     self.actor.get_num_trainable_vars())

        print("critic created")

        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        print("actor noise created")

        self.build_summaries()

        print("summaries built")

        self.sess.run(tf.global_variables_initializer())

        # to save all 100th
        self.saver = tf.train.Saver(max_to_keep=2)

        # Initialize target network weights
        self.actor.update_target_network()
        print("target actor initialised")
        self.critic.update_target_network()
        print("target critic initialised")

    def learning_rate_provider(self):
        # TODO more sophiosticated learning rate changing method
        raise ('Not implemented: learning rate provider')

    def train(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        # Calculate targets
        target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

        y_i = []
        batch_size = np.size(t_batch)
        #terminal is a boolen 1d array
        for k in range(batch_size):
            if t_batch[k]:
                y_i.append(r_batch[k])
            else:
                y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

        # Update the critic given the targets
        predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, (batch_size, 1)))

        # Update the actor policy using the sampled gradient
        a_outs = self.actor.predict(s_batch)

        # gradienseket ezzel kiolvassa a tensorflow graph-ból és visszamásolja
        grads = self.critic.action_gradients(s_batch, a_outs)
        self.actor.train(s_batch, grads[0])

        # Update target networks
        self.actor.update_target_network()
        self.critic.update_target_network()
        return np.amax(predicted_q_value)

    def save(self, additional_info):
        path = self.network_dir + '/full_network_e' + additional_info + '.tfl'
        self.saver.save(self.sess, path)

    def set_random_seed(self, random_seed):
        tf.set_random_seed(random_seed)

    # ===========================
    #   Tensorflow Summary Ops
    # ===========================

    def build_summaries(self):
        episode_reward = tf.Variable(0., name='reward')
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0., name='qmax')
        tf.summary.scalar("Qmax Value", episode_ave_max_q)

        max_episode_reward = tf.Variable(0., name='reward_max')
        tf.summary.scalar("Max episode reward Value", max_episode_reward)

        self.summary_vars = [episode_reward, episode_ave_max_q, max_episode_reward]
        self.summary_ops = tf.summary.merge_all()

        self.log_writer = tf.summary.FileWriter(logdir=self.log_dir)

    def update_summaries(self, ep_reward, ep_ave_max_q, best_reward, step):
        summary_str = self.sess.run(self.summary_ops, feed_dict={
        self.summary_vars[0]: ep_reward,
        self.summary_vars[1]: ep_ave_max_q,
        self.summary_vars[2]: best_reward})

        self.log_writer.add_summary(summary_str, step)
        self.log_writer.flush()
