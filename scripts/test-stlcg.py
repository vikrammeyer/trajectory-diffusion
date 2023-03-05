from trajdiff.stlcg import *

import numpy as np
import torch
import matplotlib.pyplot as plt

# plt.figure(figsize=(8,5))
# plt.plot(x,y)
# plt.scatter(x,y)
# plt.grid()
# plt.xlabel("Time steps")
# plt.ylabel("s")
# plt.title("Bump data")
# plt.tight_layout()

# class NN(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(NN, self).__init__()
#         self.fc1 = torch.nn.Linear(input_size, hidden_size)
#         self.fc2 = torch.nn.Linear(hidden_size, output_size)
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# model = NN(1, 10, 1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

# stl_model = NN(1, 10, 1)
# stl_optimizer = torch.optim.Adam(stl_model.parameters(), lr=0.01, weight_decay=0.01)

# # Data
x = np.arange(-2,2,0.1)
y = np.zeros_like(x)
y[:10] = -0.5
y[10:30] = 0.5
y[30:] = -0.5
y += np.random.randn(len(x)) * 0.1

x_t = torch.tensor(x).reshape(-1,1) # [B, 1]
y_t = torch.tensor(y).reshape(-1,1) # [B, 1]

y_upper_lim = torch.tensor(0.52)
y_lower_lim = torch.tensor(0.48)


# STL Formula

y_exp = Expression("y", y_t)

sub1 = y_exp <= y_upper_lim
sub2 = y_exp >= y_lower_lim

formula = Always(subformula=And(sub1, sub2))
print(formula)

demo = torch.ones_like(y_t) * 0.6
print(demo.shape)
demo = demo[None, :]
print(demo.shape)
print(formula.robustness((demo, demo)).item())

# # %%
# # Train NN

# for _ in range(1000):
#     optimizer.zero_grad()
#     y_pred = model(x_t)
#     loss = torch.mean((y_pred - y_t)**2)
#     loss.backward()
#     optimizer.step()




