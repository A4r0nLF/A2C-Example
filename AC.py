import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp

#PyTorch umhüllt die in Python eingebaute Multiprocessing-Bibliothek

class ActorCritic(nn.Module): #Definiert   kombiniertes Modell für den Actor/Critic
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))#Der Actor-Head gibt die
        y = F.relu(self.l2(y))#Log-Wahrscheinlichkeiten
        actor = F.log_softmax(self.actor_lin1(y),dim=0) #über die 2 Aktionen zurück.
        c = F.relu(self.l3(y.detach()))#Der Critic gibt eine einzige Zahl zurück,
        critic = torch.tanh(self.critic_lin1(c))# die durch –1 und +1 begrenzt ist.
        return actor, critic # Gibt die Actor- und Critic-Ergebnisse als Tupel zurück