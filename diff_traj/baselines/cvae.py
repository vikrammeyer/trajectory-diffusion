"""
AutoEncoder
VAE
Conditional VAE
"""

import torch
import torch.nn as nn

class VariationalEncoder(nn.Module):
  def __init__(self, in_dim, latent_dim, n_layers, layer_size):
    super(VariationalEncoder, self).__init__()
    self.latent_dim = latent_dim
    enc = nn.ModuleList()

    for i in range(n_layers+1):
      if i == 0:
        enc.append(nn.Linear(in_dim, layer_size))
        enc.append(nn.ReLU())
      elif i == n_layers:
        enc.append(nn.Linear(layer_size, 2*latent_dim)) # mu and log(sigma^2)
      else:
        enc.append(nn.Linear(layer_size, layer_size))
        enc.append(nn.ReLU())

    self.enc = nn.Sequential(*enc)

    self.kl_loss = 0.0

  def kl_div_N01(self, mu, log_sigma_sq):
      term_1 = log_sigma_sq.sum(axis=1)       # tr(log Sigma_q)
      term_2 = log_sigma_sq.exp().sum(axis=1) # tr(Sigma_q)
      term_3 = (mu*mu).sum(axis=1)            # mu_q squared and summed
      return 0.5*(-term_1 + term_2 + term_3 - mu.shape[1]).mean()

  def forward(self, x):
    z = self.enc(x)

    mean = z[...,:self.latent_dim]
    log_sigma_sq = z[...,self.latent_dim:]

    sigma = log_sigma_sq.exp().sqrt()
    # TODO: ensure sampling can run on GPU (set device)
    z = mean + torch.randn_like(mean)*sigma # eps ~ N(0,1)

    self.kl_loss = self.kl_div_N01(mean, log_sigma_sq)

    return z


class ConditionalDecoder(nn.Module):
  def __init__(self, latent_dim, cond_dim, out_dim, n_layers, layer_size):
    super(ConditionalDecoder, self).__init__()

    dec = nn.ModuleList()

    for i in range(n_layers + 1):
      if i == 0: # first layer
        dec.append(nn.Linear(latent_dim + cond_dim, layer_size))
        dec.append(nn.ReLU())
      elif i == n_layers: # last layer
        dec.append(nn.Linear(layer_size, out_dim))
      else: # hidden layers
        dec.append(nn.Linear(layer_size, layer_size))
        dec.append(nn.ReLU())

    self.dec = nn.Sequential(*dec)

  def forward(self, x, c):
    inpt = torch.cat((x,c), dim=1)
    return self.dec(inpt)

class CVAE(nn.Module):
  def __init__(self, in_dim, latent_dim, cond_dim, n_layers, layer_size):
    super(CVAE, self).__init__()

    self.encoder = VariationalEncoder(in_dim, latent_dim, n_layers, layer_size)
    self.decoder = ConditionalDecoder(latent_dim, cond_dim, in_dim, n_layers, layer_size)

  def forward(self, x, c):
    z = self.encoder(x)
    return self.decoder(z, c)
