import torch
import numpy as np
from torch.distributions.laplace import Laplace


def add_laplace_noise(tensor, epsilon=0.1, delta=1e-5, sensitivity=1.0):
    # Calculate the scale parameter for the Laplace distribution
    scale = sensitivity * delta / epsilon

    # Create a Laplace distribution with the calculated scale
    laplace = Laplace(torch.tensor([0.0]), torch.tensor([scale]))
    
    # Sample noise from the Laplace distribution with the same shape as the input tensor
    noise = laplace.sample(tensor.shape).to(tensor.device)
    # Add the noise to the input tensor
    noisy_tensor = tensor.squeeze() + noise.squeeze()

    # Clamp the noisy tensor to the range [0, 1]
    clamped_noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

    return clamped_noisy_tensor


def random_response(tensor, p=0.25):
    # Create a mask with the same shape as the input tensor
    mask = torch.rand(tensor.shape, device=tensor.device) < p

    # Flip the values in the tensor according to the mask and probability p
    noisy_tensor = torch.where(mask, 1 - tensor, tensor)

    return noisy_tensor
