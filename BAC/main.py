# Seizing Serendipity: Exploiting the Value of Past Success in Off-Policy Actor-Critic (2024-BAC)
# Expectile regression + entropy regularization in SAC
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from BAC import BAC_Agent, Buffer_Replay
import os, shutil
from datetime import datetime
from Env_Name import Name, Brief_Name
import argparse
import warnings
import time
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def str2bool(V):
    if isinstance(V, bool):
        return V
    elif V.lower in ('yes', 'true', 't', 'y'):
        return True
    elif V.lower in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def eval_func(env, model, e_turns, isrender, a_range):
    score = 0
    for _ in range(e_turns):
        s, _ = env.reset()
        done = False
        while not done:
            a = model.action_select(s, True)
            a = mapping(a, a_range, mark=True)
            s_, r, dw, tr, _ = env.step(a)
            score += r
            done = (dw or tr)
            s = s_
            if isrender:
                time.sleep(0.001)
    return score / e_turns


def mapping(a, a_range, mark):
    if mark:
        a = (a_range[1] - a_range[0]) / 2 * a + (a_range[0] + a_range[1]) / 2
    else:
        a = (2 * a - (a_range[0] + a_range[1])) / (a_range[1] - a_range[0])
    return a


parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=3, help='Index of environment')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=True, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=int(2e6), help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(2e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(100e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--tua', type=float, default=0.005, help='tua for updating target')
parser.add_argument('--quantile', type=float, default=0.7, help='the quantile regression for value function')
parser.add_argument('--lamda', type=float, default=0.5, help='value for lambda')
parser.add_argument('--start_steps', type=int, default=int(1e4), help='random steps')
parser.add_argument('--updates_per_step', type=int, default=int(1), help='updates per step')
opt = parser.parse_args()
opt.algo = 'BAC'
opt.Env_Name = Name[opt.EnvIdex]
opt.BName = Brief_Name[opt.EnvIdex]
print(opt)


def main():
    Env_Name = opt.Env_Name
    algo = opt.algo
    BName = opt.BName
    env_train = gym.make(Env_Name, render_mode=None)
    env_eval = gym.make(Env_Name, render_mode="human" if opt.render else None)
    opt.action_dim = env_train.action_space.shape[0]
    opt.a_range = [env_train.action_space.low[0], env_train.action_space.high[0]]
    opt.state_dim = env_train.observation_space.shape[0]
    opt.max_e_steps = env_train._max_episode_steps

    env_seed = opt.seed
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:', algo, '  Env:', BName, '  state_dim:', opt.state_dim, ' control range:', opt.a_range,
          '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(algo, BName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    model = BAC_Agent(**vars(opt))
    if opt.Loadmodel: model.load(BName, opt.ModelIdex)
    Replay = Buffer_Replay(opt.state_dim, opt.action_dim, max_size=int(1e6))

    if opt.render:
        while True:
            env_show = gym.make(Env_Name, render_mode="human")
            score = eval_func(env_show, model, 10, True, opt.a_range)
            print(f'Env:{BName}, seed:{opt.seed}, Episode Reward:{score}')
    else:
        total_steps = 0
        while total_steps <= opt.Max_train_steps:
            state, _ = env_train.reset(seed=env_seed)
            done = False
            env_seed += 1
            while not done:
                if total_steps <= 5 * opt.max_e_steps:
                    action = env_train.action_space.sample()
                    action_p = mapping(action, opt.a_range, mark=False)
                else:
                    action_p = model.action_select(state, False)
                    action = mapping(action_p, opt.a_range, mark=True)

                state_, reward, dw, tr, _ = env_train.step(action)
                if opt.EnvIdex == 1 or opt.EnvIdex == 2:
                    if reward <= -100: reward = -1
                elif opt.EnvIdex == 4:
                    reward = (reward + 8) / 8

                done = (dw or tr)
                Replay.add(state, action_p, reward, state_, done, dw)

                state = state_
                total_steps += 1

                if Replay.size > opt.batch_size:
                    for _ in range(opt.updates_per_step):
                        # Expectile regression + SAC
                        model.train(Replay)

                # if total_steps % opt.update_every == 0 and total_steps >= 2 * opt.max_e_steps:
                #     for _ in range(opt.update_every):
                #         model.train(Replay)

                if total_steps % opt.eval_interval == 0 or total_steps == 1:
                    score = eval_func(env_eval, model, 3, False, opt.a_range)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:', BName, 'seed:', opt.seed, 'steps:{}k'.format(int(total_steps / 1e3)), 'score:',
                          score)

                if total_steps % opt.save_interval == 0:
                    model.save(BName, total_steps)

    env_show = gym.make(Env_Name, render_mode="human")
    score = eval_func(env_show, model, 10, True, opt.a_range)
    print(f'Env:{BName}, seed:{opt.seed}, Episode Reward:{score}')
    env_show.close()
    env_train.close()
    env_eval.close()


if __name__ == '__main__':
    main()
