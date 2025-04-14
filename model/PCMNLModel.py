import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------
# 1. ParetoLayer
# ---------------------------
class ParetoLayer(nn.Module):
    def __init__(self, num_features, init_F_u=0.85, init_threshold=1.0, init_alpha=1.0, eps=1e-6):
        super(ParetoLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.raw_F_u = nn.Parameter(torch.tensor(init_F_u))
        self.raw_threshold = nn.Parameter(torch.ones(num_features) * init_threshold)
        self.raw_alpha = nn.Parameter(torch.ones(num_features) * init_alpha)

    def forward(self, x):
        F_u = torch.sigmoid(self.raw_F_u)
        # x: shape (batch, num_features)
        u = F.softplus(self.raw_threshold) + self.eps  # shape (num_features,)
        alpha = F.softplus(self.raw_alpha) + self.eps  # shape (num_features,)
        u = u.unsqueeze(0)  # (1, num_features)
        alpha = alpha.unsqueeze(0)  # (1, num_features)
        x_clamped = torch.clamp(x, min=self.eps)

        # x <= u f_body = (x/u)*F_u
        f_body = (x_clamped / u) * F_u
        # x > u
        # f_tail = F_u + (1 - F_u) * (1 - (u/x)^alpha)
        f_tail = F_u + (1 - F_u) * (1 - (u / x_clamped) ** alpha)
        out = torch.where(x_clamped <= u, f_body, f_tail)
        out = torch.clamp(out, self.eps, 1 - self.eps)

        return out, alpha


# ---------------------------
# 2. GaussianCopulaLayer
# ---------------------------
class GaussianCopulaLayer(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(GaussianCopulaLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.raw_L = nn.Parameter(torch.randn(num_features, num_features))

    def forward(self, u):
        # u: shape (batch, num_features)
        u = torch.clamp(u, self.eps, 1 - self.eps)
        # inverse_normal(u) = sqrt(2)*erfinv(2u - 1)
        device = u.device
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=device))
        z = sqrt2 * torch.erfinv(2 * u - 1)  # z ~ N(0,1)

        L = torch.tril(self.raw_L)
        diag_indices = torch.arange(self.num_features, device=device)
        L_diag = F.softplus(L[diag_indices, diag_indices])
        L = L.clone()
        L[diag_indices, diag_indices] = L_diag
        # z_corr = z @ L^T
        z_corr = torch.matmul(z, L.t())
        return z_corr, L


# ---------------------------
# 3. MixedLogitLayer
# ---------------------------
class MixedLogitLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MixedLogitLayer, self).__init__()
        self.mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma = nn.Parameter(torch.ones(out_features, in_features)*0.1)
        # self.sigma = F.softplus(self.sigma)

    def forward(self, x):
        epsilon = torch.randn_like(self.sigma)
        weight = self.mu + self.sigma * epsilon
        return F.linear(x, weight)


# ---------------------------
# 4. PCMNLModel
# ---------------------------
class PCMNLModel(nn.Module):
    def __init__(self, config):
        super(PCMNLModel, self).__init__()
        self.continuous_size = config['num_continuous_features']
        self.discrete_size = config['num_discrete_features']
        self.emb_size = config['total_unique_categorical']
        self.emb_dim = config['embedding_dim']
        self.hidden_size = config['hidden_size']
        self.output_dim = config['num_choices']

        self.pareto_layer = ParetoLayer(self.continuous_size)
        self.copula_layer = GaussianCopulaLayer(self.continuous_size)

        self.embeddings = nn.Embedding(self.emb_size, self.emb_dim)

        self.batch_norm1 = nn.BatchNorm1d(self.continuous_size)
        self.batch_norm2 = nn.BatchNorm1d(self.discrete_size * self.emb_dim)
        # self.fc1 = nn.Linear(self.continuous_size, self.output_dim)
        # self.fc2 = nn.Linear(self.discrete_size * self.emb_dim, self.output_dim)
        self.fc1 = MixedLogitLayer(self.continuous_size, self.output_dim)
        self.fc2 = MixedLogitLayer(self.discrete_size * self.emb_dim, self.output_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.continuous_size + self.discrete_size * self.emb_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_dim),
        )

    def forward(self, x):

        continuous = x[:, :self.continuous_size]
        u, alpha = self.pareto_layer(continuous)
        z_corr, L = self.copula_layer(u)

        discrete = x[:, self.continuous_size:].long()
        emb_discrete = self.embeddings(discrete)
        emb_discrete = emb_discrete.view(emb_discrete.size(0), -1)

        z_corr = self.batch_norm1(z_corr)
        emb_discrete = self.batch_norm2(emb_discrete)
        continuous_logits = self.fc1(z_corr)
        discrete_logits = self.fc2(emb_discrete)

        fusion_input = torch.cat([z_corr, emb_discrete], dim=1)  # (batch, continuous_size + discrete_size)
        gate_raw = self.gate_mlp(fusion_input)
        gate = torch.sigmoid(gate_raw)

        # fusion_weight * cont_out + (1 - fusion_weight) * disc_out
        output = gate * continuous_logits + (1 - gate) * discrete_logits
        return output


