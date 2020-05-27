import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSum(nn.Module):

    def __init__(self):
        pass

    def forward(self, extracted_interests):
        return torch.sum(extracted_interests)

class SimpleMean(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return torch.mean(x)

class LastHidden(nn.Module):

    def __init__(self):
        pass

    def forward(self, extracted_interests):
        return extracted_interests[-1]