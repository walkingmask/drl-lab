from datetime import datetime, timedelta, timezone
import time

import numpy as np

from drl_lab.sim import Simulator
from drl_lab.utils import Saver


class Experiment():
    def __init__(self, name):
        JST = timezone(timedelta(hours=+9), 'JST')
        self.name = name+'_'+datetime.now(JST).strftime('%Y%m%d%H%M%S')

    def _convert_save_at(self, save_at, max_steps):
        if type(save_at) is int:
            # save_at as number of save times
            return [(max_steps // save_at)*i for i in range(save_at+1)]
        elif type(save_at) is list:
            # save_at as timing to save
            if type(save_at[0]) is int:
                return save_at
            elif type(save_at[0]) is float:
                # As ratio
                return [int(ratio*max_steps) for ratio in save_at]
            else:
                raise TypeError(
                    "{} expected {} as list or int, but {} given".format(
                        '_convert_save_at()', 'save_at[0]', type(save_at)))
        else:
            raise TypeError(
                "{} expected {} as list or int, but {} given".format(
                    '_convert_save_at()', 'save_at', type(save_at)))

    def run(self, env_hparams, run_hparams, nn_hparams):
        simulator = Simulator(env_hparams, nn_hparams)

        interval = run_hparams['interval']
        num_runs = run_hparams['num_runs']
        verbose = run_hparams['verbose']
        max_steps = run_hparams['max_steps']

        # save settings
        save_at = run_hparams['save_at']
        save = save_at not in [0, None, [], '']
        if save:
            save_at = self._convert_save_at(save_at, max_steps)
            self.saver = Saver(self.name)
            self.saver.init()
            self.saver.save_hparams(env_hparams, run_hparams, nn_hparams)
            self.saver.save_model(simulator.agent.nn.nn, 'init')

        for num_run in range(1, num_runs+1):
            self._run(simulator, interval, max_steps,
                      num_run, save, save_at, verbose)

        # fin
        if save:
            self.saver.save_plot_all_n_average_rewards(num_runs, interval)

    def _run(self, simulator, interval, max_steps,
             num_run, save, save_at, verbose):
        average_rewards = []
        best_score = -100000

        r = simulator.run(iterations=interval, update=False)
        r = np.mean(r)  # random agent baseline (random dqn weights)
        average_rewards.append(r)

        while True:
            steps = simulator.agent.step_counter
            t = time.time()
            r = simulator.run(iterations=interval, update=True)
            steps = steps - simulator.agent.step_counter  # steps done
            step_per_sec = steps / (t - time.time())
            mr = np.mean(r)

            if verbose:
                remaining_min = max_steps - simulator.agent.step_counter
                remaining_min /= step_per_sec
                remaining_min /= 60
                remaining_min = np.round((remaining_min, 2))
                print('steps: ', simulator.agent.step_counter,
                      'mean reward: ', mr,
                      'epsilon: ',
                      np.round(simulator.agent.explore_chance, 2),
                      'steps/sec: ', np.round(step_per_sec, 2),
                      'remaining min: ', remaining_min)

            average_rewards.append(mr)

            # save best
            if mr > best_score and save:
                if verbose:
                    print('new best score, saving model...')
                self.saver.save_model(simulator.agent.nn.nn, num_run, 'best')
                best_score = mr

            # save en route
            if save:
                if simulator.agent.step_counter in save_at:
                    self.saver.save_model(simulator.agent.nn.nn, num_run,
                                          simulator.agent.step_counter)

            if simulator.agent.step_counter >= max_steps:
                if save:
                    self.saver.save_run_rewards(average_rewards, num_run)
                break
