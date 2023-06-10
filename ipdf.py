import torch

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from scipy.spatial.transform import Rotation as R


class IPDF(nn.Module):
    def __init__(self):
        super().__init__()
        # See: https://github.com/google-research/google-research/tree/master/implicit_pdf#reproducing-symsol-results
        # and Section S8.
        self.config = {'name': 'implicit_s_o3', 'trainable': True, 'dtype': 'float32', 'len_visual_description': 2048, 'number_fourier_components': 3, 'so3_sampling_mode': 'healpix', 'number_train_queries': 4096, 'number_eval_queries': 16384}
        self.number_eval_queries = self.config['number_eval_queries']
        self.grids = {}
        self.len_rotation = 9
        self.number_fourier_components = self.config['number_fourier_components']
        self.frequencies = torch.arange(self.number_fourier_components, dtype=torch.float32)
        self.frequencies = torch.pow(2., self.frequencies)
        self.so3_sampling_mode = self.config["so3_sampling_mode"]
        self.number_train_queries = self.config["number_train_queries"]
        self.number_eval_queries = self.config["number_eval_queries"]
        if self.number_fourier_components == 0:
            self.len_query = self.len_rotation
        else:
            self.len_query = self.len_rotation * self.number_fourier_components * 2
        

        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        visual_embedding_size = self.cnn.layer4[2].bn3.num_features
        self.L = 3
        R_feats = 2 * self.L * 9
        n_hidden_nodes = 256
        self.img_linear = nn.Linear(visual_embedding_size, n_hidden_nodes)
        self.R_linear = nn.Linear(R_feats, n_hidden_nodes)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, 1),
        )

    def output_pdf(self, imgs, query_rotations):
        """Returns a normalized distribution over pose, given a vision description.
        Args:
        vision_description: A batch of feature vectors, representing the images on
            which to estimate the pose.
        num_queries: The number of queries to evaluate the probability for.
        query_rotations: If supplied, these rotations will be used to evaluate the
            distribution and normalize it, instead using the kwarg num_queries.
        Returns:
        Both the rotations and their corresponding probabilities.
        """

        log_probs = self.get_scores(imgs, query_rotations)

        probabilities = torch.softmax(log_probs, dim=1)
        
        return query_rotations, probabilities
    
    def get_scores(self, imgs, Rs):
        x = self.cnn.conv1(imgs)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = self.cnn.avgpool(x)
        x = torch.flatten(x, 1)

        Rs_encoded = []
        for l_pos in range(self.L):
            Rs_encoded.append(torch.sin(2**l_pos * torch.pi * Rs))
            Rs_encoded.append(torch.cos(2**l_pos * torch.pi * Rs))

        Rs_encoded = torch.cat(Rs_encoded, dim=-1)

        # See Equation (9) in Section S8 and:
        # https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L120-L126.
        x = self.img_linear(x).unsqueeze(1) + self.R_linear(Rs_encoded)
        x = self.mlp(x).squeeze(2)

        return x

    def forward(self, imgs, Rs_fake_Rs):
        # See: https://github.com/google-research/google-research/blob/207f63767d55f8e1c2bdeb5907723e5412a231e1/implicit_pdf/models.py#L188
        # and Equation (2) in the paper.
        V = torch.pi**2 / Rs_fake_Rs.shape[1]
        probs = 1 / V * torch.softmax(self.get_scores(imgs, Rs_fake_Rs), 1)[:, 0]
        return probs
