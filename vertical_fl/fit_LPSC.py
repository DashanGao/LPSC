"""
The LPSC algorithm.
"""

import os
import logging
import copy
import json
from time import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch import index_select, tensor
from torch.utils.data import DataLoader
from deepctr_torch.inputs import SparseFeat, DenseFeat
from byzantine.attack import Attacker
from byzantine.defence import Defender
from utils.loss import KL_Loss, MSE_Loss, CE_Loss
from utils.utils import get_best_performane_marker, process_input, init_tensorboard, load_data, save_training_result
from utils.dp import add_laplace_noise, random_response

class TrainerLPSC(object):
    def __init__(self, config=None):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        self.config = config
        self.local_model = None
        self.boost_models = []
        self.schedulers = []
        self.indices = []
        self.num_benign_passive_parties = len(config['boost_features']) # Begign passive party number
        self.num_byzantine_passive_parties = config['num_byzantine']
        self.num_passive_parties = self.num_byzantine_passive_parties + self.num_benign_passive_parties
        self.boost_model_loss_type = self.config["boost_model_loss_type"]
        self.boost_data_batchsize = self.config["batch_size"]
        self.device = 'cuda:0' if (torch.cuda.is_available()) else 'cpu'
        self.train_local_flag, self.train_boosting_flag = self.config['train_local_flag'], True
        self.tensorboard = init_tensorboard(config)
        self.attacker = Attacker(device=self.device, attack_type=self.config['byzantine_attack'])
        self.defender = Defender(num_benign_passive_parties=self.num_benign_passive_parties, 
                        num_passive_parties=self.num_passive_parties, 
                        defence_type=self.config['defence_type'],
                        tensorboard=self.tensorboard,
                        device=self.device)
        torch.autograd.set_detect_anomaly(True)
        self.init_record_data()

    def fit_boosting_models(self):
        """Fit boosting models given pre-trained local model. 
        """
        input_data = []
        self.indices = []
        idx = 0
        for model in [self.local_model] + self.boost_models:
            input_data.append(process_input(self.data, model.feature_index))  
            self.indices.append(tensor(range(idx, idx + input_data[-1].shape[-1])))
            idx += input_data[-1].shape[-1]
        input_data = np.concatenate(input_data, axis=-1) 

        target = self.data['label'].values
        train_number = int(self.data.shape[0] * 0.8 )

        train_tensor_data = Data.TensorDataset( 
            torch.from_numpy(input_data[:train_number, :]),
            torch.from_numpy(target[:train_number])
        )

        testset = {'input_data': input_data[train_number:, :], 
                    'target': target[train_number:]
                }
        train_data_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=self.config["batch_size"])

        for boost_model in self.boost_models:
            boost_model.train()

        sample_num = len(train_tensor_data)
        steps_per_epoch = int(self.config['common_data_ratio'] * (sample_num - 1) // self.config["batch_size"]) + 1

        logging.info("Train on {0} samples,  {1} steps per epoch".format(
            len(train_tensor_data), steps_per_epoch))

        boost_mse_loss = nn.MSELoss().to(device=self.device)
        step = 0
        for epoch in range(self.config["initial_epoch"], self.config['epochs']):
            iter_t = time()
            try:
                with tqdm(enumerate(train_data_loader), disable=(self.config['verbose'] == 0)) as t:
                    for index, (x, y) in t:
                        if index >= steps_per_epoch:  
                            break
                        step += 1
                        self.defender.set_step_num(step)
                        self.attacker.set_step_num(step)
                        # 1. load data
                        x_local = index_select(x, 1, self.indices[0])  
                        x_local_ = x_local.to(self.device).float()
                        x_local_common_ = x_local_[: self.boost_data_batchsize, :] 
                        x_boost_ = x[: self.boost_data_batchsize, :].to(self.device).float()
                        y = y.to(self.device).float()
                        y_boost_ = y[: self.boost_data_batchsize]
                        for boost_model in self.boost_models:
                            boost_model.optim.zero_grad()

                        # ------ Update Boost model ---------
                        # *********  Fit the residual via Gradient boosting  *********
                        y_boost_logit = []
                        for idx, boost_model in enumerate(self.boost_models):
                            x_boost_input = index_select(x_boost_, 1, self.indices[idx + 1].to(self.device))
                            y_boost_logit.append(boost_model.logit(x_boost_input).squeeze())
                        for i in range(self.config['num_byzantine']):
                            noise = self.attacker.attack(size=y_boost_logit[0].shape, grad=y_boost_logit[i+2])
                            y_boost_logit.append(noise)
                        y_boost_logit = torch.stack(y_boost_logit)

                        with torch.autograd.no_grad():
                            y_local_common_pred = self.local_model(x_local_common_).squeeze()
                        residual = (y_boost_ - y_local_common_pred) / torch.mul(y_local_common_pred,
                                                                                (1 - y_local_common_pred))

                        y_boost_logit_aggregated = self.defender.defend(y_boost_logit, residual)
                        boost_model_loss = boost_mse_loss(y_boost_logit_aggregated, residual)

                        if self.config['beta_unlearn'] != 0:
                            unlearn_label_loss = []
                            for i in range(len(self.boost_models)):
                                unlearn_label_loss.append(-1. * self.config['beta_unlearn'] * F.binary_cross_entropy_with_logits(y_boost_logit[i, :].squeeze(), y_boost_, reduction="mean"))
                            for unlearn_loss in unlearn_label_loss:
                                unlearn_loss.backward(retain_graph=True)
                            
                        self.tensorboard.add_scalar("training/residual_mean", torch.mean(residual), step)
                        self.tensorboard.add_scalar("training/residual_std", torch.std(residual), step)

                        # *********!  Fit the residual via Gradient boosting  !*********
                        total_boost_model_loss = boost_model_loss.clone() + boost_model.reg_loss.clone() 

                        # 3. Compute gradient
                        total_boost_model_loss.backward(retain_graph=True)

                        # 4. Update parameters
                        for boost_model in self.boost_models:
                            boost_model.update_regularization_loss()
                            boost_model.reg_loss.backward(retain_graph=True)  
                            boost_model.optim.step()
                        
                        self.tensorboard.add_scalar("training/boost_loss", boost_model_loss, step)
                        # ------! Update Boost model !---------

                        # Evaluate on test set and save the best model.
                        if index % self.config['eval_interval'] == 0 and index > 0:
                            time_left = (time() - iter_t) / (index + 1) * (steps_per_epoch - index)
                            logging.info("Training\tEpoch {0} iter {1} - epoch_time_left: {2: .4f} s".format(epoch,
                                                            (str(index) + "/" + str(steps_per_epoch)), time_left))

                            test_results = self.evaluate_federated_model(testset=testset)
                            self.log_evaluation_resultes(test_results, step)

                            fed_test_result = test_results["fed_test_result"]
                            local_test_result = test_results["local_test_result"]
                            boost_test_result = test_results["mean_boost_test_result"]

            except KeyboardInterrupt as e:
                logging.error("Error happened in the epoch loop during training.  " + str(e))
                t.close()
                raise
            t.close()

            # Upate LR for boosting models.
            for scheduler in self.schedulers:
                scheduler.step()
            test_results = self.evaluate_federated_model(testset=testset)
            self.log_evaluation_resultes(test_results, epoch)
            self.defender.log_tensorboard()
        self.save_record_data()

    def fit_averaging_models(self):
        """Fit passive parties' models dirctly from labels and use averaging-based aggreegation. 
        """
        # Process and split dataset. 
        input_data = []
        self.indices = []
        idx = 0
        for model in [self.local_model] + self.boost_models:
            input_data.append(process_input(self.data, model.feature_index))  # Convert OrderedDict to numpy.array
            self.indices.append(tensor(range(idx, idx + input_data[-1].shape[-1])))
            idx += input_data[-1].shape[-1]
        input_data = np.concatenate(input_data, axis=-1) # Concate inputs by feature columns

        target = self.data['label'].values
        train_number = int(self.data.shape[0] * 0.8 )

        train_tensor_data = Data.TensorDataset(  # Boost dataset and local dataset have the save #rows
            torch.from_numpy(input_data[:train_number, :]),
            torch.from_numpy(target[:train_number])
        )

        testset = {'input_data': input_data[train_number:, :], 
                    'target': target[train_number:]
                }
        train_data_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=self.config["batch_size"])

        for boost_model in self.boost_models:
            boost_model.train()

        sample_num = len(train_tensor_data)
        steps_per_epoch = int(self.config['common_data_ratio'] * (sample_num - 1) // self.config["batch_size"]) + 1

        logging.info("Train on {0} samples,  {1} steps per epoch".format(
            len(train_tensor_data), steps_per_epoch))

        step = 0
        for epoch in range(self.config["initial_epoch"], self.config['epochs']):
            iter_t = time()
            
            try:
                with tqdm(enumerate(train_data_loader), disable=(self.config['verbose'] == 0)) as t:
                    for index, (x, y) in t:
                        if index >= steps_per_epoch:  # Modify boosting batchsize. 
                            break
                        step += 1
                        self.defender.set_step_num(step)
                        self.attacker.set_step_num(step)
                        # 1. load data
                        x_local = index_select(x, 1, self.indices[0])  # Select columns
                        x_local_ = x_local.to(self.device).float()
                        x_local_common_ = x_local_[: self.boost_data_batchsize, :]  # Select the features of x_local
                        x_boost_ = x[: self.boost_data_batchsize, :].to(self.device).float()
                        y = y.to(self.device).float()
                        y_boost_ = y[: self.boost_data_batchsize]
                        for boost_model in self.boost_models:
                            boost_model.optim.zero_grad()

                        # ------ Update Boost model ---------
                        # Baseline condition. Local model locally train. Boost model locally train over common data.
                        # *********  Fit the label  *********
                        y_pred = []
                        y_pred_logit = []
                        for idx, boost_model in enumerate(self.boost_models):
                            x_boost_input = index_select(x_boost_, -1, self.indices[idx + 1].to(self.device))
                            x_boost_logit_ = boost_model.logit(x_boost_input)
                            y_pred_ = boost_model.out(x_boost_logit_)
                            y_pred.append(y_pred_)
                            y_pred_logit.append(x_boost_logit_.squeeze().clone().detach())

                        for i in range(self.config['num_byzantine']):
                            noise = self.attacker.attack(size=y_pred[0].shape, grad=y_pred[i+2])
                            y_pred.append(noise)
                        y_pred = torch.stack(y_pred).squeeze()
                        y_pred_logit = torch.stack(y_pred_logit)

                        self.record_data(y_boost_, "label")
                        self.record_data(y_pred_logit, "residuals")
                        # self.record_data(self.defender.linear_model[0].weight, "lambdas")

                        y_pred = self.defender.defend(y_pred, y_boost_).unsqueeze(1)

                        # LDP protect label privacy: Random Response
                        if self.config['ldp_rr'] > 0:
                            y_boost_ = random_response(y_boost_, self.config['ldp_rr'])
                        if self.config['dp_Lap_epsilon'] is not None:
                            y_pred = add_laplace_noise(y_pred.squeeze(), epsilon=self.config['dp_Lap_epsilon'])
                        y_pred.clamp_(0, 1)
                        boost_model_loss = F.binary_cross_entropy(y_pred.squeeze(), y_boost_.squeeze(),
                                                                reduction="mean")
                        # ********!  Fit the label  !********
                        total_boost_model_loss = boost_model_loss.clone()  # + boost_model.reg_loss.clone()   # *

                        # 3. Compute gradient
                        total_boost_model_loss.backward(retain_graph=True) # Delete retain_graph=True   # *

                        # 4. Update parameters
                        for boost_model in self.boost_models:
                            boost_model.optim.step()
                        
                        # ------! Update Boost model !---------

                        # Evaluate on test set and save the best model.
                        if index % self.config['eval_interval'] == 0 and index > 0:
                            time_left = (time() - iter_t) / (index + 1) * (steps_per_epoch - index)
                            logging.info("Training\tEpoch {0} iter {1} - epoch_time_left: {2: .4f} s".format(epoch,
                                                            (str(index) + "/" + str(steps_per_epoch)), time_left))

                            test_results = self.evaluate_federated_model(testset=testset)
                            self.log_evaluation_resultes(test_results, step)

                            fed_test_result = test_results["fed_test_result"]
                            local_test_result = test_results["local_test_result"]
                            boost_test_result = test_results["mean_boost_test_result"]

            except KeyboardInterrupt as e:
                logging.error("Error happened in the epoch loop during training.  " + str(e))
                t.close()
                raise
            t.close()

            test_results = self.evaluate_federated_model(testset=testset)
            self.log_evaluation_resultes(test_results, step)
            self.defender.log_tensorboard()
        self.save_record_data()

    def fit_local_model(self):
        """Fit the local model on the labeled local data. 
        """
        # Process and split dataset. 
        input_data = process_input(self.data, self.local_model.feature_index)  # Convert OrderedDict to numpy.array
        self.indices = tensor(range(0, input_data.shape[-1]))
        target = self.data['label'].values
        train_number = int(self.data.shape[0] * 0.8 )

        train_tensor_data = Data.TensorDataset(  # Boost dataset and local dataset have the save #rows
            torch.from_numpy(input_data[:train_number, :]),
            torch.from_numpy(target[:train_number])
        )

        testset = {'input_data': input_data[train_number:, :], 
                    'target': target[train_number:],
                }
        train_data_loader = DataLoader(dataset=train_tensor_data, shuffle=False, batch_size=self.config["batch_size"])

        self.local_model.train()

        local_optim = self.local_model.optim
        local_scheduler = torch.optim.lr_scheduler.ExponentialLR(local_optim, gamma=self.config["lr_decay_gamma"])
        sample_num = len(train_tensor_data)
        steps_per_epoch = int( (sample_num - 1) // self.config["batch_size"]) + 1

        logging.info("Train local model on {0} samples,  {1} steps per epoch".format(
            len(train_tensor_data), steps_per_epoch))

        step = 0
        for epoch in range(self.config["initial_epoch"], self.config['epochs']):
            iter_t = time()
            loss_epoch, total_loss_epoch = 0, 0
            try:
                with tqdm(enumerate(train_data_loader), disable=(self.config['verbose'] == 0)) as t:
                    for index, (x, y) in t:
                        step += 1

                        # 1. load data
                        x_local = index_select(x, 1, self.indices)  # Select columns
                        x_local_ = x_local.to(self.device).float()
                        y = y.to(self.device).float()
                        local_optim.zero_grad()

                        # ------ Update Local model ---------
                        # Empirical loss
                        y_local_pred = self.local_model(x_local_)
                        bceloss = nn.BCELoss()
                        self.local_model.update_regularization_loss()
                        local_total_loss = bceloss(y_local_pred.squeeze(), y.squeeze()) + self.local_model.reg_loss.clone()
                        local_total_loss.backward()

                        self.tensorboard.add_scalar("training/local_model_loss", local_total_loss, step)

                        # ------! Update Local model !---------

                        # Evaluate on test set and save the best model.
                        if index % self.config['eval_interval'] == 0 and index > 0:
                            time_left = (time() - iter_t) / (index + 1) * (steps_per_epoch - index)
                            logging.info("Training\tEpoch {0} iter {1} - epoch_time_left: {2: .4f} s".format(epoch,
                                                            (str(index) + "/" + str(steps_per_epoch)), time_left))

                            local_test_result = self.evaluate_local_model(testset=testset)
                            logging.info('Local model: AUC:{0}, Loss:{1}'.format(local_test_result['auc'], local_test_result['binary_crossentropy']))

            except KeyboardInterrupt as e:
                logging.error("Error happened in the epoch loop during training.  " + str(e))
                t.close()
                raise
            t.close()

            local_optim.step()
            if (epoch + 1) % 10 == 0:
                local_test_result = self.evaluate_local_model(testset=testset)
                logging.info('Epoch:{0} AUC:{1}, Loss:{2}'.format(epoch, local_test_result['auc'], local_test_result['binary_crossentropy']))

        local_test_result = self.evaluate_local_model(testset=testset)
        logging.info('Local model: AUC:{0}, Loss:{1}'.format(local_test_result['auc'], local_test_result['binary_crossentropy']))

    def generate_input_features_for_all_partyes(self, data):
        input_data = []
        self.indices = []
        idx = 0
        for model in [self.local_model] + self.boost_models:
            input_data.append(process_input(data, model.feature_index))  # Convert OrderedDict to numpy.array
            self.indices.append(tensor(range(idx, idx + input_data[-1].shape[-1])))
            idx += input_data[-1].shape[-1]
        return np.concatenate(input_data, axis=-1) # Concate inputs by feature columns

    def evaluate_local_model(self, testset):
        """Evaluate the local model on test set. 

        Args:
            testset (_type_): test set. 

        Returns:
            Dict: eval_result_local = {'auc': ... , 'binary_crossentropy': ...}
        """
        x, y = testset["input_data"], testset["target"]
        self.local_model.eval()
        tensor_data = Data.TensorDataset(
            torch.from_numpy(x)
        )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=self.config["batch_size"]) 
        pred_local = []
        with torch.no_grad():
            for index, [x] in enumerate(test_loader):
                x_local = index_select(x, 1, self.indices).to(self.device).float()
                y_local_pred =  self.local_model(x_local).cpu().data.numpy()
                pred_local.append(y_local_pred)
        pred_local = np.concatenate(pred_local)

        eval_result_local = {}
        for name, metric_fun in self.local_model.metrics.items():
            eval_result_local[name] = metric_fun(y, pred_local)
        return eval_result_local

    def predict_local_model_and_save_to_csv(self):
        """
        Predict the local model on test set. 

        Args:
            path (_type_): path to save the csv.
        """
        assert self.config['local_model_prediction_file'] is not None, "Wrong, Must have local_model_prediction_file name."

        if not self.local_model:
            self.init_local_model()
        # Init self.records.
        name_list = ["label", "local_prediction", "gt_residual"]
        self.records = []
        for name in name_list:
            self.records.append(pd.Series(dtype='float64', name=name))

        # Evaluate on all data. 
        x = process_input(self.data, self.local_model.feature_index)  # Convert OrderedDict to numpy.array
        y = self.data['label'].values
        self.indices = tensor(range(0, x.shape[-1]))
        self.local_model.eval()
        tensor_data = Data.TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y)
        )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=self.config["batch_size"]) 
        pred_local = []
        residual_list = []
        with torch.no_grad():
            with tqdm(enumerate(test_loader)) as t:
                for index, [x, y] in t:
                    y = y.to(self.device)
                    x_local = index_select(x, 1, self.indices).to(self.device).float()
                    y_local_pred = self.local_model(x_local).squeeze()
                    residual = (y - y_local_pred) / torch.mul(y_local_pred,
                                                                (1 - y_local_pred))
                    # Record results. 
                    self.record_data(y, 'label')
                    self.record_data(y_local_pred, 'local_prediction')
                    self.record_data(residual, 'gt_residual')
                    
        # Dump to csv file.
        records = pd.concat(self.records, axis=1)
        records.reset_index(inplace=True, drop=True)
        column_list = ["label", "local_prediction", "gt_residual"]
        records.set_axis(column_list, axis=1)
        records.to_csv(self.config['local_model_prediction_file'], header=column_list)

    def evaluate_federated_model(self, testset):
        """
        evaluate federated model by: add boost model and local model prediction.
        """
        x, y = testset["input_data"], testset["target"]
        results = self.predict_federated_model(x)
        eval_result, eval_result_fed, eval_result_boost, eval_result_local, eval_result_baseline = {}, {}, {}, {}, {}
        for name, metric_fun in self.boost_models[0].metrics.items():
            eval_result_fed[name] = metric_fun(y, results['fed'])
            if self.num_passive_parties > 1:
                eval_result_boost[name] = [metric_fun(y, results['boost'][i, :]) for i in range(results['boost'].shape[0])]
            else:
                eval_result_boost[name] = [metric_fun(y, results['boost']) ]
            eval_result_local[name] = metric_fun(y, results['local'])
        eval_result = {'fed_test_result': eval_result_fed, 
                        'boost_test_result': eval_result_boost,
                        'local_test_result': eval_result_local,
                        'baseline_test_result': eval_result_baseline,
                        }
        eval_result["mean_boost_test_result"] = {
                        "auc": np.mean([eval_result_boost["auc"][i] for i in range(len(eval_result_boost["auc"]))]),
                        "binary_crossentropy": np.mean([eval_result_boost["binary_crossentropy"][i] for i in range(len(eval_result_boost["auc"]))]),
                        }
        logging.info("[" + ", ".join([str(eval_result_boost["auc"][i]) for i in range(len(eval_result_boost["auc"]))]) + "]")
        logging.info(np.mean([eval_result_boost["auc"][i] for i in range(len(eval_result_boost["auc"]))]))
        return eval_result

    def predict_federated_model(self, x):
        """
        predict federated model
        :param local_model:
        :param boost_model:
        :param x_boost:
        :param x_local:
        :param y:
        :return:
        """
        for boost_model in self.boost_models:
            boost_model.eval()
        self.local_model.eval()

        tensor_data = Data.TensorDataset(
            torch.from_numpy(x)
        )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=self.config["batch_size"]) 

        pred_fed = []
        pred_boost = []
        pred_local = []
        pred_baseline = []
        pred_each_boost = []

        with torch.no_grad():
            for index, [x] in enumerate(test_loader):
                y_boost_pred = []
                for idx, boost_model in enumerate(self.boost_models):
                    x_boost = index_select(x, 1, self.indices[idx + 1]).to(self.device).float()
                    y_boost_pred.append(boost_model(x_boost))

                if self.config['attack_in_test_phase']:
                    for i in range(self.config['num_byzantine']):
                        noise = self.attacker.attack(size=y_boost_pred[0].shape, grad=y_boost_pred[i+2])
                        y_boost_pred.append(noise)
                y_boost_pred = torch.stack(y_boost_pred).squeeze()
                if self.config['defend_in_test_phase']:  # e.g. Defend low-quality party
                    y_boost_pred_aggregated = self.defender.defend(y_boost_pred).unsqueeze(-1)
                else:
                    y_boost_pred_aggregated = torch.mean(y_boost_pred, dim=0, keepdim=False).unsqueeze(-1)

                y_boost_pred_aggregated = y_boost_pred_aggregated.cpu().data.numpy()
                y_boost_pred = y_boost_pred.cpu().data.numpy()
                x_local = index_select(x, 1, self.indices[0]).to(self.device).float()
                y_local_pred =  self.local_model(x_local).cpu().data.numpy()
                if self.baseline_model:
                    y_baseline_pred =  self.baseline_model(x_local).cpu().data.numpy()
                else:
                    y_baseline_pred = y_local_pred
                
                y_pred = self.config['alpha_local'] * y_local_pred + self.config['alpha_boosting'] * y_boost_pred_aggregated
                pred_fed.append(y_pred)
                pred_boost.append(y_boost_pred)
                pred_local.append(y_local_pred)
                pred_baseline.append(y_baseline_pred)
        pred_fed = np.concatenate(pred_fed)
        if self.num_passive_parties > 1:
            pred_boost = np.concatenate(pred_boost, axis=1)  # multiple boost models
        else:
            pred_boost = np.concatenate(pred_boost)
        pred_local = np.concatenate(pred_local)
        pred_baseline = np.concatenate(pred_baseline)

        results = {'fed': pred_fed, 'boost': pred_boost, 'local': pred_local, 'baseline': pred_baseline}
        return results

    def load_data(self):
        logging.info("Loading dataset.")
        t = time()
        self.data, _, _ = load_data(self.config['data_path'], self.config['local_model_prediction_file'])
        logging.info("Data loaded. time: {0: .4f}".format(time() - t))

    def init_all_models(self):
        if self.config['embedding_vocabulary'] is not None:
            with open(self.config['embedding_vocabulary'], 'r') as src:
                vocabulary_size_dict = json.load(src)
        else:
            vocabulary_size_dict = dict()
        local_features = self.config['local_features']
        boost_features = self.config['boost_features']

        self.local_model = self.init_model(local_features, vocabulary_size_dict, lr=self.config["local_model_lr"])
        if self.config["local_model"]:  # Load model from checkpoint
            self.load_pretrained_model(self.local_model, self.config['local_model'], device=self.device)

        for i in range(len(boost_features)):
            boost_model = self.init_model(boost_features[i], vocabulary_size_dict, lr=self.config["boost_model_lr"])
            self.boost_models.append(boost_model)
            self.schedulers.append(torch.optim.lr_scheduler.ExponentialLR(boost_model.optim, gamma=self.config["lr_decay_gamma"]))

        if self.config["baseline_model"]:
            self.baseline_model = self.init_model(local_features, vocabulary_size_dict, lr=self.config["baseline_model_lr"])
        else:
            self.baseline_model = None

    def init_local_model(self):
        if self.config['embedding_vocabulary'] is not None:
            with open(self.config['embedding_vocabulary'], 'r') as src:
                vocabulary_size_dict = json.load(src)
        else:
            vocabulary_size_dict = dict()
        local_features = self.config['local_features']
        self.local_model = self.init_model(local_features, vocabulary_size_dict, lr=self.config['local_model_lr'])
        if self.config['local_model'] is not None:
            self.load_pretrained_model(self.local_model, self.config['local_model'], device=self.device)

    def init_model(self, feature, vocabulary_size_dict, lr=None, embedding_dict=None):
        """Initialize a model with given type and feature.

        Args:
            sparse_feature (_type_): pre-defined sparse feature objects
            dense_feature (_type_): pre-defined dense feature objects
            vocabulary_size_dict (dict): vocabulary size of each categorical feature
            config (dict): 
            lr (_type_, optional): Learning rate. Defaults to None.
            embedding_dict (_type_, optional): pre-defined embedding dict. Defaults to None.

        Returns:
            BaseModel: an initialized CTR prediction model.
        """
        sparse_feature = [x for x in feature if x[0] == 'C' ]
        dense_feature = [x for x in feature if x[0] == 'I' ]

        fixlen_feature_columns = [SparseFeat(name=feat, vocabulary_size=vocabulary_size_dict[feat], embedding_dim=4)
                                    for feat in sparse_feature] + \
                                    [DenseFeat(name=feat, dimension=1, ) for feat in dense_feature]
        model = self.config["model"](linear_feature_columns=fixlen_feature_columns,
                            dnn_feature_columns=fixlen_feature_columns,
                            task=self.config['task'],
                            l2_reg_linear=self.config["l2_reg_linear"],
                            l2_reg_embedding=self.config["l2_reg_embedding"],
                            l2_reg_dnn=self.config["l2_reg_dnn"],
                            device=self.device,
                            embedding_dict=embedding_dict)

        model.compile(self.config["optimizer"], self.config["loss"],
                            metrics=self.config["metrics"], lr=lr)
        # Optimizer
        model.optim = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        return model

    def load_pretrained_model(self, model, path, device='cuda:0'):
        """ Load pretrained parameters to model object.
        Args:
            model (BaseModel): model object
            config (dict):
        """
        # Load model from checkpoint
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.train()

    def log_evaluation_resultes(self, test_results, step):
        fed_test_result = test_results["fed_test_result"]
        local_test_result = test_results["local_test_result"]
        boost_test_result = test_results["mean_boost_test_result"]
        if self.num_passive_parties > 1:
            boost_aucs = [test_results["boost_test_result"]['auc'][i] for i in range(len(test_results["boost_test_result"]))]
        else:
            boost_aucs = [test_results["boost_test_result"]['auc']]
        median_boost_auc = np.median(boost_aucs)
        self.tensorboard.add_scalar('AUC/Boost_median', median_boost_auc ,step)
        self.tensorboard.add_histogram('Boost/Boost_model_AUC', boost_aucs ,step)

        self.tensorboard.add_scalar('AUC/Fed_model', fed_test_result['auc'], step)
        self.tensorboard.add_scalar('AUC/Local_model', local_test_result['auc'], step)
        self.tensorboard.add_scalar('AUC/Boost_model', boost_test_result['auc'], step)
        self.tensorboard.add_scalar('AUC/Boosting_improve', fed_test_result['auc'] - local_test_result['auc'], step)

        self.tensorboard.add_scalar('Loss/Fed_model', fed_test_result['binary_crossentropy'], step)
        self.tensorboard.add_scalar('Loss/Local_model', local_test_result['binary_crossentropy'], step)
        self.tensorboard.add_scalar('Loss/Boost_model', boost_test_result['binary_crossentropy'], step)

        eval_str = "Step{}\tTestset evaluation: ".format(step)
        eval_str += " - fed : {0: .4f} ".format(fed_test_result["auc"])
        eval_str += " - boost : {0: .4f} ".format(boost_test_result["auc"])
        eval_str += " - local : {0: .4f} ".format(local_test_result["auc"])
        if self.baseline_model:
            eval_str += " - baseline : {0: .4f} ".format(test_results['baseline_test_result']["auc"])
        logging.info(eval_str)

        logging.info("Fed   - loss: {0: .5f} AUC: {1: .5f}".format(boost_test_result['binary_crossentropy'], 
                            fed_test_result['auc'], max(fed_test_result['auc'])))
        logging.info("Boost - loss: {0: .5f} AUC: {1: .5f} ".format(boost_test_result['binary_crossentropy'], 
                            boost_test_result['auc'], max(boost_test_result['auc'])))
        logging.info("Local - loss: {0: .5f} AUC: {1: .5f} ".format(local_test_result['binary_crossentropy'], 
                            local_test_result['auc'], max(local_test_result['auc'])))
        if self.baseline_model:
            baseline_test_result = test_results["baseline_test_result"]
            self.tensorboard.add_scalar('AUC/Baseline_model', baseline_test_result['auc'], step)
            self.tensorboard.add_scalar('Loss/Baseline_model', baseline_test_result['binary_crossentropy'], step)
        
    def init_record_data(self):
        name_list = ["label", "local_prediction", "gt_residual", "fed_residual"]
        self.records = []
        for name in name_list:
            self.records.append(pd.Series(dtype='float64', name=name))
        for name in ["residuals", "lambdas"]:
            self.records.append(pd.DataFrame(dtype='float64'))

    def record_data(self, data, name):
        """record the gradient of the residuals for each boosting model. 
        Args:
            data (torch.Tensor): one batch of data
            name (str): name of the field.
        """
        if not self.config['record_data_flag']:
            return
        name_list = {"label":0, "local_prediction":1, "gt_residual":2, "fed_residual":3, "residuals":4, "lambdas":5}
        if name=='residuals':
            data = torch.transpose(data, 0, 1)

        if name_list[name] < 4:
            data = pd.Series(data.cpu().data.numpy())
        else:
            data = pd.DataFrame(data.cpu().data.numpy())
        self.records[name_list[name]] = pd.concat([self.records[name_list[name]], data], axis=0)

    def save_record_data(self):
        if not self.config['record_data_flag']:
            return
            
        path = self.config['logger_file_path'].replace('.log', '.csv')
        if self.config['boost_model_loss_type'] == 'averaging': 
            path = path.replace('.csv', '_avg.csv')

        self.records[-1].reset_index(inplace=True, drop=True)
        index_list = []
        for i in range(self.records[-1].shape[0]):
            index_list += [i] * self.config['batch_size']
        self.records[-1] = self.records[-1].loc[index_list]
        self.records[-1].reset_index(inplace=True, drop=True)

        records_ = pd.concat(self.records[:4], axis=1)
        records_.reset_index(inplace=True, drop=True)
        self.records[-2].reset_index(inplace=True, drop=True)
        records = pd.concat([records_, self.records[-2], self.records[-1]], axis=1)

        residuals_list = ["residual_"+str(x) for x in range(self.num_passive_parties)]
        lambda_list = ["lambda_"+str(x) for x in range(self.num_passive_parties)]
        column_list = ["label", "local_prediction", "gt_residual", "fed_residual"] + residuals_list + lambda_list
        records.set_axis(column_list, axis=1)
        records.to_csv(path, header=column_list)
