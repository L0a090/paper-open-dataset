import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------
# Module 1: Bayesian Linear Layer
# -------------------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-3))

    def forward(self, x):

        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * weight_eps
        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * bias_eps
        return F.linear(x, weight, bias)


# ---------------------------------------------
# Module 2: Geological Side - Gated MLP (gMLP)
# ---------------------------------------------
class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedMLP, self).__init__()
        self.channel_proj = nn.Linear(input_dim, hidden_dim)
        self.sgu_gate = nn.Linear(hidden_dim, hidden_dim)
        self.sgu_feat = nn.Linear(hidden_dim, hidden_dim)
        self.residual = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_proj = self.activation(self.channel_proj(x))
        gate = torch.sigmoid(self.sgu_gate(x_proj))
        feat = self.sgu_feat(x_proj)
        x_fused = gate * feat
        return self.activation(self.residual(x_fused) + x_proj)


# ---------------------------------------------
# Module 3: Engineering Side - LSTM Sequence
# ---------------------------------------------
class EngineeringLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(EngineeringLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output[:, -1, :]  # return final timestep hidden state


# --------------------------
# Module 4: Full G-BFNet
# --------------------------
class GBFNet(nn.Module):
    def __init__(self, geo_input_dim, eng_input_dim, hidden_dim=64):
        super(GBFNet, self).__init__()
        self.geo_branch = GatedMLP(geo_input_dim, hidden_dim)
        self.eng_branch = EngineeringLSTM(eng_input_dim, hidden_dim, 2)
        self.fusion_fc1 = BayesianLinear(2 * hidden_dim, 64)
        self.fusion_fc2 = BayesianLinear(64, 1)

    def forward(self, geo_input, eng_input):
        geo_feat = self.geo_branch(geo_input)
        eng_feat = self.eng_branch(eng_input)
        combined = torch.cat((geo_feat, eng_feat), dim=1)
        x = F.relu(self.fusion_fc1(combined))
        return self.fusion_fc2(x)


# ------------------------------------------------
# Module 5: Variational ELBO Loss with KL Penalty
# ------------------------------------------------
def elbo_loss(pred_mean, target, model, beta=1.0):
    mse = F.mse_loss(pred_mean, target, reduction='mean')
    kl = 0
    for module in model.modules():
        if isinstance(module, BayesianLinear):
            weight_sigma = torch.exp(module.weight_log_sigma)
            bias_sigma = torch.exp(module.bias_log_sigma)
            kl += 0.5 * torch.sum(
                weight_sigma.pow(2) + module.weight_mu.pow(2) - 1 - 2 * module.weight_log_sigma
            )
            kl += 0.5 * torch.sum(
                bias_sigma.pow(2) + module.bias_mu.pow(2) - 1 - 2 * module.bias_log_sigma
            )
    return mse + beta * kl


# ---------------------------------------------------
# Module 6: Monte Carlo Sampling for Uncertainty Est
# ---------------------------------------------------
def monte_carlo_predict(model, geo_input, eng_input, samples=50):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            pred = model(geo_input, eng_input)
            preds.append(pred)
    preds = torch.stack(preds, dim=0)
    mean = preds.mean(0)
    std = preds.std(0)
    return mean, std


# -----------------------------------------
# Module 7: Training Loop with Dummy Data
# -----------------------------------------
def train_gbfnet(model, train_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for geo_input, eng_input, target in train_loader:
            optimizer.zero_grad()
            output = model(geo_input, eng_input)
            loss = elbo_loss(output, target, model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f\"[Epoch {epoch+1}] Avg Loss: {total_loss / len(train_loader):.4f}\")


# -----------------------------------------
# Module 8: Example Usage with Fake Inputs
# -----------------------------------------
if __name__ == '__main__':

    # geo_input  : geological features input
    # eng_input  : drilling params input
    # rop_target : ROP ground truth

    dataset = TensorDataset(geo_input, eng_input, rop_target)
    train_loader = DataLoader(dataset, batch_size= drilling_params_sizes, shuffle=True) # drilling_params_sizes

    model = GBFNet(geo_input_dim=G_f_sizes, eng_input_dim=drilling_params_sizes, hidden_dim=64)
    train_gbfnet(model, train_loader)

    # Monte Carlo Inference

    mean, std = monte_carlo_predict(model, geo_sample, eng_sample)
    print(\"Predicted Mean ROP:\", mean.squeeze())
    print(\"Predicted Std (Uncertainty):\", std.squeeze())
