"""
Defend Byzantine Attack.
"""

import torch
import math
import numpy as np
import torch.nn.functional as F


class Defender():
    def __init__(self, num_benign_passive_parties, num_passive_parties, defence_type="no", tensorboard=None, device="cuda:0", loss='MSE'):
        self.defence_type = defence_type
        self.num_benign_passive_parties = num_benign_passive_parties
        self.num_passive_parties = num_passive_parties
        self.num_byzantine_passive_parties = num_passive_parties - num_benign_passive_parties
        self.num_filter_out_parties = self.num_byzantine_passive_parties
        self.cosin_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        self.overall_sim = None
        self.tensorboard = tensorboard
        self.length_sim_records = None
        self.device = device
        self.step = 0
        self.filter_out_Byzantine = False  # Do NOT change.   Filter out Byzantine in "weight" harms AUC. 
        if defence_type == "weight" or defence_type == "diverseFL_weight" :
            self.linear_model = torch.nn.Sequential(
                torch.nn.Linear(self.num_passive_parties, 1, bias=False),   # Tested:  bias must be False. 
                torch.nn.Flatten(0, 1)
            ).to(device)
            self.linear_model_pred = torch.nn.Sequential(
                torch.nn.Linear(self.num_passive_parties, 1, bias=False),
                torch.nn.Flatten(0, 1)
            ).to(device)
            self.optim = torch.optim.Adam(self.linear_model.parameters(), lr=1e-4, weight_decay=0.01)
            if loss=="MSE":
                self.loss = torch.nn.MSELoss(reduction='mean').to(device=device)
            elif loss=='BCEWithLogits':
                self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device=device)

    def set_loss_type(self, loss, device="cuda:0"):
        if loss=="MSE":
            self.loss = torch.nn.MSELoss(reduction='mean').to(device=device)
        elif loss=='BCEWithLogits':
            self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device=device)

    def set_step_num(self, step):
        self.step = step

    def defend(self, input, groundtruth=None):
        """_summary_

        Args:
            input (torch.Tensor): shape: (x * y):   x updates, each update has y dimensions

        Returns:
            torch.Tensor: y dimensional update.
        """

        if self.defence_type == "no":
            return torch.mean(input, dim=0, keepdim=False)
        
        if self.defence_type == "oracle":
            return torch.mean(input[:self.num_benign_passive_parties, :], dim=0, keepdim=False)

        if self.defence_type == "trimmed_mean":
            assert self.num_passive_parties > 3, "Trimmed mean must have at least 4 passive parties"
            sorted_input, _ = torch.sort(input, dim=0)
            ignore = math.ceil((self.num_passive_parties - self.num_benign_passive_parties) / 2.)
            assert self.num_passive_parties - 2 * ignore > 1, "Trimmed mean must have at least {} passive parties".format(2*ignore + 2)
            result = torch.mean(sorted_input[ignore: -1-ignore, :], dim=0, keepdim=False)
            return result
            
        if self.defence_type == "median":
            median_input, _ = torch.median(input, dim=0, keepdim=False)
            return median_input

        if self.defence_type == "noisy_median":
            input = input + torch.normal(mean=0, std=0.1, size=input.shape).to(input.get_device())
            noisy_median, _ = torch.median(input, dim=0, keepdim=False)
            return noisy_median

        if self.defence_type == "diverseFL":
            if groundtruth is not None:
                self.diverseFL_fit(input, groundtruth)  # compute Byzntine score for each party.
            return self.diverseFL_predict(input)    # defend the attack. 

        if self.defence_type == "weight":
            if groundtruth is not None:
                self.weight_fit(input, groundtruth)
            return self.weight_predict(input)
        
        if self.defence_type == "diverseFL_weight":
            pass

    def weight_fit(self, input, groundtruth):
        input_ = input.clone().detach().transpose(0, 1)
        self.optim.zero_grad()
        prediction = self.linear_model(input_)
        loss = self.loss(prediction, groundtruth)  # For VFXOR, this should be binary-crossentropy loss. F.binary_cross_entropy_with_logits(prediction, groundtruth)
        print("Defence Loss: ", loss.data)
        loss.backward()

        self.optim.step()
        for p in self.linear_model.parameters():
            p.data.clamp_(0)  # only clamp weights

        self.optim.zero_grad()
        self.tensorboard.add_scalar("defence/weight_loss", loss, self.step)
        self.tensorboard.add_scalar("defence/prediction_mean", torch.mean(prediction), self.step)
        self.tensorboard.add_scalar("defence/error_mean", torch.mean(prediction - groundtruth), self.step)
        self.tensorboard.add_scalar("defence/error_std", torch.std(prediction - groundtruth), self.step)
        
        baseline = torch.mean(input, dim=0, keepdim=False)
        self.tensorboard.add_scalar("defence/Averaging_mean", torch.mean(baseline), self.step)
        self.tensorboard.add_scalar("defence/avg_error_mean", torch.mean(baseline - groundtruth), self.step)
        self.tensorboard.add_scalar("defence/avg_error_std", torch.std(baseline - groundtruth), self.step)
        
    def weight_predict(self, input):
        if self.filter_out_Byzantine:
            params = self.linear_model.state_dict()
            min_value_indices = torch.argsort(params['0.weight'])[0, : self.num_byzantine_passive_parties]
            params_ = {}
            params_['0.weight'] = params['0.weight'].clone().detach()
            params_['0.weight'][:, min_value_indices] = 0  
            self.linear_model_pred.load_state_dict(params_)
            prediction = self.linear_model_pred(input.transpose(0, 1)) / torch.sum(params['0.weight'])
        else:
            params = self.linear_model.state_dict()
            prediction = self.linear_model(input.transpose(0, 1)) / torch.sum(params['0.weight'])
        return prediction

    def log_tensorboard(self):
        if self.defence_type == "diverseFL":
            self.length_sim_records = np.clip(self.length_sim_records, -3, 10)
            self.overall_sim_records = np.clip(self.overall_sim_records, -3, 5)
            for i in range(self.num_passive_parties):
                self.tensorboard.add_histogram('defence/Length_similarity', self.length_sim_records[:, i], i)
                self.tensorboard.add_histogram('defence/Cosine_similarity', self.cos_sim_records[:, i], i)
                self.tensorboard.add_histogram('defence/Overall_similarity', self.overall_sim_records[:, i], i)
                self.tensorboard.add_histogram('defence/Sorted_rank', self.indices_records[:, i], i)
        
        if self.defence_type == "weight":
            params = []
            for param in self.linear_model.parameters():
                params.append(param.data)
            if len(params) > 1:
                params = torch.cat([params[0].squeeze(), params[1]], dim=0)
            else:
                params = params[0].squeeze()
            for i in range(len(params)):
                self.tensorboard.add_histogram('defence/weight', params[i], i)
