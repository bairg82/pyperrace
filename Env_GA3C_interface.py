import environment
from Paper_Config import Config

class Environment:
    def __init__(self):
        self.game = environment.PaperRaceEnv(track_name = Config.track_name, car_name = Config.car_name,\
                                             random_init = Config.random_init, ref_calc = Config.ref_calc, \
                                             save_env_ref_buffer_dir = Config.save_env_ref_buffer_dir, \
                                             save_env_ref_buffer_name = Config.save_env_ref_buffer_name, \
                                             load_env_ref_buffer = Config.load_env_ref_buffer, \
                                             load_all_env_ref_buffer_dir = Config.load_all_env_ref_buffer_dir)

        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.game.reset(Config.SHOW_WINDOW)

    def get_num_actions(self):
        return 1

    def reset(self):
        self.game.reset(Config.SHOW_WINDOW)

    def step(self, action):
        v_new, pos_new, step_reward, pos_reward = self.game.step(action, Config.SHOW_WINDOW, draw_text='little_reward')

        end, time, last_t_diff, game_pos_reward, game_ref_reward = self.game.getstate()

        # no image, only pos and speed is the observation
        observation = [v_new[0], v_new[1], pos_new[0], pos_new[1]]

        self.previous_state = self.current_state

        # TODO it is not markovian because reward depends on past states as well
        self.current_state = [v_new[0], v_new[1], pos_new[0], pos_new[1]]

        reward = step_reward
        done = end

        self.total_reward += reward

        return reward, done
