
import torch
import torch.nn as nn
import torch.nn.functional as F


class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss


class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch      -> B X num_classes
        # teacher_outputs   -> B X num_classes

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)

        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)

        return loss


class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()

    def forward(self, output_batch, teacher_outputs):
        # output_batch      -> B X num_classes
        # teacher_outputs   -> B X num_classes

        batch_size = output_batch.size(0)
        output_batch = F.softmax(output_batch, dim=1)
        teacher_outputs = F.softmax(teacher_outputs, dim=1)
        # Same result MSE-loss implementation torch.sum -> sum of all element
        loss = torch.sum((output_batch - teacher_outputs) ** 2) / batch_size

        return loss


def update_regularization_loss(model):
    """
    Update the l2-regularization of the model
    """
    model.add_regularization_loss(
        model.embedding_dict.parameters(), model.l2_reg_embedding)
    model.add_regularization_loss(
        model.linear_model.parameters(), model.l2_reg_linear)