import os
from pprint import pprint

import matplotlib as mpl
mpl.use('SVG')  # NOQA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Saver:
    def __init__(self, name=None):
        self.name = str(name)
        if name is None:
            self.name = 'saver_'+str(os.getpid())

    def init(self, save_reward=True, save_model=True, save_image=True):
        here = os.path.dirname(os.path.realpath(__file__))
        results_root = here+'/results/'+self.name
        results_dirs = [['results_root', results_root]]

        if save_reward:
            results_dirs.append(['reward_results', results_root+'/rewards'])
        if save_model:
            results_dirs.append(['model_results', results_root+'/models'])
        if save_image:
            results_dirs.append(['image_results', results_root+'/images'])

        for results_dir in results_dirs:
            name, path = results_dir
            if not os.path.exists(path):
                os.makedirs(path)
            setattr(self, name, path)

    def save_hparams(self, env_hparams, run_hparams, nn_hparams):
        with open(self.results_root+'/hparams.py', 'w') as f:
            f.write('env_hparams = ')
            pprint(env_hparams, stream=f)
            f.write('run_hparams = ')
            pprint(run_hparams, stream=f)
            f.write('nn_hparams = ')
            pprint(nn_hparams, stream=f)

    def _save_rewards(self, rewards, path):
        np.save(path, np.array(rewards))

    def save_run_rewards(self, rewards, num_run):
        path = self.reward_results+'/rewards_'+str(num_run)+'.npy'
        self._save_rewards(rewards, path)

    def save_plot_all_n_average_rewards(self, num_runs, interval):
        all_rewards = []
        rewards_min_lenght = 999999999
        for i in range(1, num_runs+1):
            rewards = np.load(self.reward_results+'/rewards_'+str(i)+'.npy')
            all_rewards.append(rewards)
            if len(rewards) < rewards_min_lenght:
                rewards_min_lenght = len(rewards)

        all_rewards = np.array(all_rewards)
        average_rewards = np.zeros((rewards_min_lenght, num_runs))
        plt.plot()

        for i in range(1, num_runs+1):
            rewards = all_rewards[i-1]
            rewards_length = len(rewards)
            plt.plot(range(0, rewards_length * interval, interval),
                     rewards, alpha=0.225)
            average_rewards[0:, i-1] = rewards[0:rewards_min_lenght]

        plt.plot(range(0, rewards_min_lenght * interval, interval),
                 np.mean(average_rewards, axis=1))
        plt.savefig(self.reward_results+'/all_n_average.png')

    def save_model(self, model, num_run=None, step_num=None):
        path = self.model_results+'/model'
        if num_run is not None:
            path += '_'+str(num_run)
        if step_num is not None:
            path += '_'+str(step_num)
        model.save(path, include_optimizer=True)

    def save_arrays_as_images(self, arrays, name, prefix='image'):
        path = self.image_results+'/'+name
        if not os.path.exists(path):
            os.makedirs(path)
        save_arrays_as_images(arrays, path, prefix)


def deprocess(array, warning=True):
    if type(array) != np.ndarray:
        raise TypeError('deprocess: np.ndarray is required.')

    deprocessed_array = np.copy(array)
    deprocessed = '|'

    if deprocessed_array.min() < 0.0:
        deprocessed_array = deprocessed_array - deprocessed_array.min()
        deprocessed += '| -min |'
    if deprocessed_array.max() > 255:
        deprocessed_array = deprocessed_array / deprocessed_array.max()
        deprocessed += '| /max |'
    if deprocessed_array.max() <= 1.0:
        deprocessed_array = deprocessed_array * 255
        deprocessed += '| *255 |'
    if deprocessed_array.dtype != np.uint8:
        deprocessed_array = np.uint8(deprocessed_array)
        deprocessed += '| uint() ||'

    if warning and deprocessed is not '|':
        print("***** Waring *****: array deprocessed: "+deprocessed)

    return deprocessed_array


def bulk_deprocess(arrays, warning=True):
    deprocessed_arrays = []
    for array in arrays:
        deprocessed_arrays.append(deprocess(array, warning))
    return deprocessed_arrays


def array2image(array):
    return Image.fromarray(array)


def arrays2images(arrays):
    """
    Notes
    -----
    uint8 is recommended for array.
    """
    images = [array2image(array) for array in arrays]

    return images


def save_image(image, save_path):
    image.save(save_path)


def save_images(images, save_dir, prefix='image'):
    for i, image in enumerate(images):
        save_path = "{}/{}_{:04d}.png".format(save_dir, prefix, i)
        save_image(image, save_path)


def save_array_as_image(array, save_path):
    image = array2image(array)
    save_image(image, save_path)


def save_arrays_as_images(arrays, save_dir, prefix='image'):
    images = arrays2images(arrays)
    save_images(images, save_dir, prefix)


def save_gif(images, save_path):
    images[0].save(save_path, save_all=True, append_images=images[1:],
                   optimize=False, duration=50, loop=0)
