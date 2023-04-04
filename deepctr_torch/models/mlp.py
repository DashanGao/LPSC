import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import FM, DNN


class MLPLogit(BaseModel):
    """Instantiates the DeepFM Network architecture.
    forward() with probability and logit as output.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.
    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(128, 64, 32),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 embedding_dict=None,
                 dnn_activation='relu',
                 dnn_use_bn=False, task='binary',
                 device='cpu'):

        super(MLPLogit, self).__init__(linear_feature_columns, dnn_feature_columns,
                                     dnn_hidden_units=dnn_hidden_units,
                                     l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, 
                                     l2_reg_dnn=l2_reg_dnn, 
                                     init_std=init_std,
                                     seed=seed,
                                     dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                     embedding_dict=embedding_dict,
                                     task=task, device=device)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(len(dnn_feature_columns), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear_model = self.fc1
        self.to(device)

    def forward(self, X):
        logit = self.logit(X)
        y_pred = self.sigmoid(logit)
        return y_pred

    def logit(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logit = self.fc3(x)
        return logit

    def update_regularization_loss(self):
        l1_lambda = 0.01
        l2_lambda = 0.01
        
        l1_reg = torch.norm(self.fc1.weight, p=1) + torch.norm(self.fc2.weight, p=1)
        l2_reg = torch.norm(self.fc1.weight, p=2) + torch.norm(self.fc2.weight, p=2)
        
        self.reg_loss = l1_lambda * l1_reg + l2_lambda * l2_reg
