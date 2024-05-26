import numpy as np
import gym
import metaworld
import random

ENVS = {
    #["ML1", "MT1", "ML10", "MT10", "ML45", "MT50"]
    #'ML1-pick-place-v2': metaworld.ML1('pick-place-v2'),
    #'MT1-pick-place-v2': metaworld.MT1('pick-place-v2'),
    #'ML1-push-v2': metaworld.ML1('push-v2'),
    #'MT1-push-v2': metaworld.MT1('push-v2'),

    #'ML1-reach-v2': metaworld.ML1('reach-v2'),
    #'MT1-reach-v2': metaworld.MT1('reach-v2'),
    #'ML1-sweep-into-v2': metaworld.ML1('sweep-into-v2'),
    #'MT1-sweep-into-v2': metaworld.MT1('sweep-into-v2'),
    #'ML1-window-open-v2': metaworld.ML1('window-open-v2'),
    #'MT1-window-open-v2': metaworld.MT1('window-open-v2'),

    #'ML1-basketball-v2': metaworld.ML1('basketball-v2'),
    #'MT1-basketball-v2': metaworld.MT1('basketball-v2'),
    #'ML1-button-press-v2': metaworld.ML1('button-press-v2'),
    #'MT1-button-press-v2': metaworld.MT1('button-press-v2'),
    #'ML1-dial-turn-v2': metaworld.ML1('dial-turn-v2'),
    #'MT1-dial-turn-v2': metaworld.MT1('dial-turn-v2'),
    #'ML1-drawer-close-v2': metaworld.ML1('drawer-close-v2'),
    #'MT1-drawer-close-v2': metaworld.MT1('drawer-close-v2'),
    #'ML1-peg-insert-side-v2': metaworld.ML1('peg-insert-side-v2'),
    #'MT1-peg-insert-side-v2': metaworld.MT1('peg-insert-side-v2'),
    #####################################################################
    #'ML10': metaworld.ML10(),
    #'MT10': metaworld.MT10(),
    #'ML45': metaworld.ML45(),
    #'MT50': metaworld.MT50(),
}

def load_env(key, task: str = None, task_num: int = None, render_mode: str = 'rgb_array'):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """

    # Gym
    # Meta Gym
    '''
    Format: [MetaGym or Gym]-[Agent(MT1 or hopper)]
    optional(task) as an input
    '''
    if task is not None:
        key = '-'.join((key, task))
        task_name = '-'.join((task, 'v2'))

    if key == 'Gym-Ant':
        training_envs = [gym.make('Ant-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-HalfCheetah':
        training_envs = [gym.make('HalfCheetah-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-Hopper':
        training_envs = [gym.make('Hopper-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-Humanoid-Standup':
        training_envs = [gym.make('HumanoidStandup-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-Humanoid':
        training_envs = [gym.make('Humanoid-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-InvertedDoublePendulum':
        training_envs = [gym.make('InvertedDoublePendulum-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-InvertedPendulum':
        training_envs = [gym.make('InvertedPendulum-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-Reacher':
        training_envs = [gym.make('Reacher-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-Swimmer':
        training_envs = [gym.make('Swimmer-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]
    elif key == 'Gym-Walker':
        training_envs = [gym.make('Walker-v4', render_mode=render_mode)]
        eval_env_idx = 0
        testing_envs = training_envs[eval_env_idx]

    elif key == 'MetaGym-ML1-pick-place':
        assert task is not None
        assert task_num is not None
        ml = metaworld.ML1(task_name)
        training_envs = []
        for name, env_cls in ml.train_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            training_envs.append(env)
        testing_envs = []
        for name, env_cls in ml.test_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.test_tasks
                                    if task.env_name == name])
            env.set_task(task)
            testing_envs.append(env)
        eval_env_idx = random.choice(range(len(testing_envs)))
        testing_envs = testing_envs[eval_env_idx]
    elif key == 'MetaGym-MT1-pick-place':
        assert task is not None
        assert task_num is not None
        ml = metaworld.MT1(task_name)
        tasks = random.sample(ml.train_tasks, task_num)
        training_envs = []
        for task in tasks:
            env = ml.train_classes[task_name](render_mode=render_mode)  # Create an environment with task `pick_place`
            env.set_task(task)
            training_envs.append(env)
        eval_env_idx = random.choice(range(len(training_envs)))
        testing_envs = training_envs[eval_env_idx]
    elif key == 'MetaGym-ML1-door-open':
        assert task is not None
        assert task_num is not None
        ml = metaworld.ML1(task_name)
        training_envs = []
        for name, env_cls in ml.train_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            training_envs.append(env)
        testing_envs = []
        for name, env_cls in ml.test_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.test_tasks
                                    if task.env_name == name])
            env.set_task(task)
            testing_envs.append(env)
        eval_env_idx = random.choice(range(len(testing_envs)))
    elif key == 'MetaGym-MT1-door-open':
        assert task is not None
        assert task_num is not None
        ml = metaworld.MT1(task_name)
        tasks = random.sample(ml.train_tasks, task_num)
        training_envs = []
        for task in tasks:
            env = ml.train_classes[task_name](render_mode=render_mode)  # Create an environment with task `pick_place`
            env.set_task(task)
            training_envs.append(env)
        eval_env_idx = random.choice(range(len(training_envs)))
        testing_envs = training_envs[eval_env_idx]
    elif key == 'MetaGym-ML1-lever-pull':
        assert task is not None
        assert task_num is not None
        ml = metaworld.ML1(task_name)
        training_envs = []
        for name, env_cls in ml.train_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            training_envs.append(env)
        testing_envs = []
        for name, env_cls in ml.test_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.test_tasks
                                    if task.env_name == name])
            env.set_task(task)
            testing_envs.append(env)
        eval_env_idx = random.choice(range(len(testing_envs)))
    elif key == 'MetaGym-MT1-lever-pull':
        assert task is not None
        assert task_num is not None
        ml = metaworld.MT1(task_name)
        tasks = random.sample(ml.train_tasks, task_num)
        training_envs = []
        for task in tasks:
            env = ml.train_classes[task_name](render_mode=render_mode)  # Create an environment with task `pick_place`
            env.set_task(task)
            training_envs.append(env)
        eval_env_idx = random.choice(range(len(training_envs)))
        testing_envs = training_envs[eval_env_idx]
    elif key == 'MetaGym-ML10':
        if task_num is not None:
            print('Warning: task_num is given')
        ml = metaworld.ML10()
        for name, env_cls in ml.train_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            training_envs.append(env)
        testing_envs = []
        for name, env_cls in ml.test_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.test_tasks
                                    if task.env_name == name])
            env.set_task(task)
            testing_envs.append(env)
        eval_env_idx = random.choice(range(len(testing_envs)))
        testing_envs = testing_envs[eval_env_idx]
    elif key == 'MetaGym-MT10':
        if task_num is not None:
            print('Warning: task_num is given')
        ml = metaworld.MT10()
        training_envs = []
        for name, env_cls in ml.train_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            training_envs.append(env)
        eval_env_idx = random.choice(range(len(training_envs)))
        testing_envs = training_envs[eval_env_idx]

    return training_envs, testing_envs, eval_env_idx