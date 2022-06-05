import multiprocessing as mp
import gym
import numpy as np
import torch
from torch.nn.functional import normalize
from torch import optim
import matplotlib.pyplot as plt



from AC import ActorCritic



def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    eplenlist = []

    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards, G = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards, G)
        eplenlist.append(eplen)
        counter.value = counter.value + 1
    plt.plot(eplenlist)
    plt.show()


def run_episode(worker_env, worker_model, N_steps=10):
    raw_state = np.array(worker_env.env.state)
    state = torch.from_numpy(raw_state).float()
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    G=torch.Tensor([0])

    while (j < N_steps and done == False):
        j+=1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
            G = value.detach()
        rewards.append(reward)
    return values, logprobs, rewards, G

def update_params(worker_opt,values,logprobs,rewards,G,clc=0.1,gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = G

    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1)
    Returns = normalize(Returns, dim=0)
    actor_loss = -1*logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns,2)
    loss = actor_loss.sum() + clc*critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)

if __name__ == '__main__':
    MasterNode = ActorCritic()
    MasterNode.share_memory()
    processes = []

    params = {
        'epochs': 5000,
        'n_workers': 4,
    }

    counter = mp.Value('i', 0)

    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start()
    processes.append(p)

    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    #Save Trained Modell
    torch.save(MasterNode, 'trained_modell.pth')





