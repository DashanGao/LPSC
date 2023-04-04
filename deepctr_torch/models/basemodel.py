# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""
from __future__ import print_function

import time
import logging
from string import Template
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(  # A True / False list
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(   # A True / False list
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1)).to(
                device)
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        varlen_embedding_list = get_varlen_pooling_list(self.embedding_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list = sparse_embedding_list + varlen_embedding_list

        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            linear_sparse_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
            linear_dense_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit = linear_sparse_logit + linear_dense_logit
        elif len(sparse_embedding_list) > 0:
            linear_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
        elif len(dense_value_list) > 0:
            linear_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
        else:
            linear_logit = torch.zeros([X.shape[0], 1])
        return linear_logit


class BaseModel(nn.Module):
    def __init__(self,
                 linear_feature_columns,
                 dnn_feature_columns,
                 dnn_hidden_units=(128, 128),
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5,
                 l2_reg_dnn=0,
                 init_std=0.0001,
                 seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu',
                 embedding_dict=None,
                 task='binary', device='cpu'):

        super(BaseModel, self).__init__()

        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device  # device

        self.feature_index = build_input_features(  # OrderedDict():  { feat_name: {start_index, end_index} }
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns
        if embedding_dict:
            self.embedding_dict = embedding_dict
        else:
            self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_linear = l2_reg_linear
        self.add_regularization_loss(
            self.embedding_dict.parameters(), l2_reg_embedding)
        self.add_regularization_loss(
            self.linear_model.parameters(), l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

    def fit(self, x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            initial_epoch=0,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            use_double=False,
            testset=dict(),
            save_model=False,
            des_path="checkpoints/tmp.pt",
            show_progress=False):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param use_double: Boolean. Whether to use double precision in metric calculation.

        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        # Generate validation data
        if validation_data:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        # logging.info(self.device, end="\n")
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        logging.info("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        # Set best_performance_maker
        best_performance_marker = -1
        try:
            if "auc" in self.metrics.keys() or "accuracy" in self.metrics.keys() or "acc" in self.metrics.keys():
                best_performance_marker = 0
            elif "mse" in self.metrics.keys() or "binary_crossentropy" in self.metrics.keys() or "logloss" in self.metrics.keys():
                best_performance_marker = 1e7
            if best_performance_marker == -1:
                raise NameError("The metric is not in {auc, accuracy, acc, nse, binary_crossentropy, logloss}.")
        except Exception as e:
            print("Error: ", e)
            exit()

        # Set overfitting marker:
        overfitting_marker = 0
        loss_compute_interval = 10
        loss_update_interval = 10
        for epoch in range(initial_epoch, epochs):
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            iter_t = time.time()
            print("Epoch " + str(epoch))
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for index, (x_train, y_train) in t:
                        # 1. load data
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x)
                        optim.zero_grad()

                        # 2. Compute loss
                        # Empirical loss
                        loss = loss_func(y_pred.squeeze(), y.squeeze(), reduction='sum')

                        # Compute regularization loss
                        if index % loss_compute_interval == 0:  # Update reg_loss
                            self.add_regularization_loss(
                                self.embedding_dict.parameters(), self.l2_reg_embedding)
                            self.add_regularization_loss(
                                self.linear_model.parameters(), self.l2_reg_linear)
                            total_loss = loss.clone() + self.reg_loss.clone()  # + self.aux_loss.clone()
                        elif index % loss_update_interval == 0:
                            total_loss = loss.clone() + self.reg_loss.clone()  # + self.aux_loss.clone()
                        else:
                            total_loss = loss.clone()

                        # 3. Compute gradient
                        total_loss.backward(retain_graph=True)

                        # 4. Update parameters
                        optim.step()

                        # works after model training
                        # record loss
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()

                        # Log training performance during training.
                        if verbose > 0:
                            templ = Template("Epoch ${epoch} iter $iter ")
                            training_performance = templ.substitute({"epoch": epoch, "iter": index})
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                if use_double:
                                    train_result[name].append(metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                                else:
                                    train_result[name].append(metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy()))
                            #     training_performance += "- {0}: {1: .4f} ".format(name, train_result[name][-1])
                            # logging.info(training_performance)

                        # Log the progress of each epoch
                        if show_progress and index % 50 == 0:
                            # -------- Log training info
                            training_performance = templ.substitute({"epoch": epoch, "iter": index})
                            for name in self.metrics.keys():
                                training_performance += "- {0}: {1: .4f} ".format(name, np.sum(train_result[name][-50:]) / 50)
                            logging.info(training_performance)
                            # -------- *
                            time_left = (time.time() - iter_t) / (index + 1) * (steps_per_epoch - index)
                            logging.info("Training\tEpoch {0} iter {1} - epoch_time_left: {2: .4f} s".format(epoch,
                                                                                                             (str(
                                                                                                                 index) + "/" + str(
                                                                                                                 steps_per_epoch)),
                                                                                                             time_left))
                        # Evaluate on test set and save the best model.
                        if index % 50 == 0:
                            eval_str = "\tTestset evaluation: "
                            if len(val_x) and len(val_y):
                                eval_result = self.evaluate(val_x, val_y, batch_size)
                                for name, result in eval_result.items():
                                    eval_str += " - val_" + name + \
                                                ": {0: .4f}".format(result)

                            test_result = self.evaluate(testset["input"], testset["target"])
                            for name, metric_fun in self.metrics.items():
                                eval_str += " - test_{0}: {1: .4f} ".format(name, test_result[name])
                            logging.info(eval_str)
                            logging.info("Current test performance: " + str(test_result) + " Best performance: " + str(best_performance_marker))
                            if self.check_best_model(test_result, best_performance_marker):
                                overfitting_marker = 0
                                # Relax loss_update_interval
                                loss_update_interval = 10
                                if not save_model:
                                    best_performance_marker = test_result[self.best_performance_metric]
                            else:
                                overfitting_marker += 1
                                # Decrease loss update interval when overfit
                                loss_update_interval = int(loss_update_interval * 0.6)

                                if overfitting_marker >= 4:
                                    logging.info("Overfitting: Model performance has not increased for 3 periods. Stop training")
                                    t.close()
                                    return round(best_performance_marker, 4)
                                    # exit()
                                    # raise KeyboardInterrupt
                            if save_model and self.check_best_model(test_result, best_performance_marker):
                                # print("Save best model")
                                # add check best model here
                                self.save(des_path, type="Best_model_checkpoint_during_training", **test_result)
                                best_performance_marker = test_result[self.best_performance_metric]
                                logging.info("Best model saved to {0}".format(des_path))

            except KeyboardInterrupt as e:
                logging.error("Error happened in the epoch loop during training.  " + str(e))
                t.close()
                raise
            t.close()

            epoch_time = int(time.time() - start_time)
            if verbose > 0:
                logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, total_loss_epoch / sample_num)

                # Train MSE
                for name, result in train_result.items():
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(np.sum(result) / steps_per_epoch)

                # Validation MSE
                if len(val_x) and len(val_y):
                    eval_result = self.evaluate(val_x, val_y, batch_size)
                    for name, result in eval_result.items():
                        eval_str += " - val_" + name + \
                                    ": {0: .4f}".format(result)

                # Test set MSE
                if testset:
                    test_result = self.evaluate(testset["input"], testset["target"])
                    for name, metric_fun in self.metrics.items():
                        eval_str += " - test_{0}: {1: .4f} ".format(name, test_result[name])

                logging.info(eval_str)

                # if save_model and self.check_best_model(test_result, best_performance_marker):
                #     self.save(des_path, type="Best_model_checkpoint_during_training", **test_result)
                #     best_performance_marker = test_result[self.best_performance_metric]
                #     logging.info("Best model saved to {0}".format(des_path))
        return round(best_performance_marker, 4)

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size:
        :return: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        """
        self.eval()
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256, use_double=False):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        if isinstance(x, list):
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)
            x = np.concatenate(x, axis=-1)
        assert isinstance(x, np.ndarray), "X is not of correct type, " + str(type(x))

        tensor_data = Data.TensorDataset(
            torch.from_numpy(x))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                # y = y_test.to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        if use_double:
            return np.concatenate(pred_ans).astype("float64")
        else:
            return np.concatenate(pred_ans)

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        varlen_sparse_embedding_list = get_varlen_pooling_list(self.embedding_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_loss(self, weight_list, weight_decay, p=2):
        """Suggested by GPT-4"""
        reg_loss = []
        for w in weight_list:
            if isinstance(w, tuple):
                reg_loss.append(torch.linalg.norm(w[1].clone(), ord=p, ))
            else:
                reg_loss.append(torch.linalg.norm(w.clone(), ord=p,))
        if len(reg_loss) == 0:
            self.reg_loss = torch.tensor(0)
        else:
            self.reg_loss = weight_decay * torch.stack(reg_loss).sum()

    def add_auxiliary_loss(self, aux_loss, alpha):
        # Not used in this func
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                lr=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        :param lr: learning rate of the optimizer
        """
        self.optim = self._get_optim(optimizer, lr)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics, set_eps=True)
        self.best_performance_metric = self._find_best_performance_marker()

    def _get_optim(self, optimizer, lr=None):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps=eps,
                        normalize=normalize,
                        sample_weight=sample_weight,
                        labels=labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
        return metrics_

    def check_best_model(self, eval_results, min_error):
        # check best parameters according to matrix in {auc, mse, acc}
        if self.best_performance_metric in ["auc", "accuracy", "acc"]:
            is_best = eval_results[self.best_performance_metric] > min_error
        elif self.best_performance_metric in ["logloss", "binary_crossentropy", "mse"]:
            is_best = eval_results[self.best_performance_metric] < min_error
        else:
            raise NotImplementedError
        return is_best

    def _find_best_performance_marker(self):
        """
        find a best performance marker according to matrix in {auc, mse, acc}
        Author: Dashan Gao
        :return:
        """
        if "auc" in self.metrics.keys():
            marker = "auc"
        elif "accuracy" in self.metrics.keys():
            marker = "accuracy"
        elif "acc" in self.metrics.keys():
            marker = "acc"
        elif "mse" in self.metrics.keys():
            marker = "mse"
        elif "binary_crossentropy" in self.metrics.keys():
            marker = "binary_crossentropy"
        elif "logloss" in self.metrics.keys():
            marker = "logloss"
        else:
            raise NotImplementedError
        return marker

    def save(self, des_path, **kwargs):
        """
        save the model
        Author: Dashan Gao
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            **kwargs
        }, des_path)
        # metadata_to_save = {
        #     'model_state_dict': "Model parameters, ignored here.",
        #     **kwargs
        # }
        # print("Saved model. ", metadata_to_save)

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]

    def update_regularization_loss(self):
        pass
