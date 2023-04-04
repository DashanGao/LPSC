"""
Convey Byzantine Attack

Implemented Byzantine Attack:
1. Gaussian attack: Gaussian noise N(0, \epsilon)
2. Sign-flip attack: invert sign of logit
3. Same-value attck: Ones
4. No attack: Zeros  (Real no attack should set TrainerFedBoosting.num_byzantine_passive_parties=0 )
"""

import torch


class Attacker():
    def __init__(self, device, attack_type="gaussian", size=None):
        self.attack_type = attack_type
        self.size = size
        self.std = 10
        self.device = device
        self.step = 0

    def attack(self, size=None, grad=None, value=None):
        if not size:
            size = self.size
        if self.attack_type == "gaussian":
            return self.gaussian_attack(size)
        if self.attack_type == "sign_flip":
            return self.sign_flip_attack(grad)
        if self.attack_type == "same_value":
            return self.same_value_attack(size)
        if self.attack_type == "no":
            return torch.zeros(size=size).to(self.device).float()

    def gaussian_attack(self, size=None):
        noise = torch.normal(mean=0, std=10, size=size).to(self.device).float()
        return noise

    def sign_flip_attack(self, grad):
        if grad == None:
                raise ValueError("Sign flip attack needs grad as input")
        noise = -1 * grad
        return noise
    
    def same_value_attack(self, size=None):
        noise = torch.mul(torch.tensor(1.0), torch.ones(size=size)).to(self.device).float()
        return noise

    def set_step_num(self, step):
        """
        For tensorboard step use. 
        """
        self.step = step
