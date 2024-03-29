import json
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from trajdiff.stlcg import Expression, Always, And, Or, Not, Eventually, Until
from trajdiff.utils import write_obj


def train_baseline(model, train_loader, val_loader, epochs, learning_rate):
    """
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use for optimization.

    Returns:
        model (torch.nn.Module): The trained model.
        losses (dict): Lists of training & val losses at the end of each epoch.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("using %s", dev)

    model = model.to(dev)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()

        for inputs, targets in train_loader:
            inputs = inputs.to(dev)
            targets = targets.to(dev)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            train_losses.append(loss.item())
            loss.backward()

            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_loader:
                inputs = inputs.to(dev)
                targets = targets.to(dev)

                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            val_losses.append(val_loss / len(val_loader))

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}"
        )

    model = model.cpu()
    losses = {"train losses": train_losses, "val losses": val_losses}
    return model, losses


def train_baseline_stl(model, train_loader, val_loader, epochs, learning_rate, cfg, gamma=1):
    """
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use for optimization.
        gamma (float): The weight of the STL loss.

    Returns:
        model (torch.nn.Module): The trained model.
        losses (dict): Has lists of training & val losses at the end of each epoch.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("using %s", dev)

    model = model.to(dev)

    mse = nn.MSELoss()

    # s_exp = Expression("x", )
    # params = Expression("p")
    # obst_radi = Expression("r")

    # # y is N dimensional tensor of all the y coordinates of the trajectory
    # ys = torch.tensor([])
    # y_exp = Expression("y", ys)

    # xs = torch.tensor([])
    # x_exp = Expression("x", xs)

    # thetas = torch.tensor([])
    # theta_exp = Expression("theta", thetas)

    # vs = torch.tensor([])
    # vel_exp = Expression("vel", vs)


    # upper_lane = y_exp <= (cfg.lane_width / 2) * torch.ones(cfg.n_intervals)
    # lower_lane = y_exp >= (-cfg.lane_width / 2) * torch.ones(cfg.n_intervals)
    # in_lane = And(upper_lane, lower_lane)

    # traj_in_lane = Always(in_lane)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()

        for inputs, targets in train_loader:
            inputs = inputs.to(dev)
            targets = targets.to(dev)

            optimizer.zero_grad()

            outputs = model(inputs)

            mse_loss = mse(outputs, targets)

            # flip input signal along the trajectory axis to get the reverse trajectory while
            # preserving batch size
            # robustness = traj_in_lane.robustness(outputs.flip((-1)))


            loss = mse_loss # + gamma * torch.relu(-robustness)

            train_losses.append(
                {
                    "total loss": loss.item(),
                    "mse loss": mse_loss.item(),
                    #"stl loss": stl_loss.item(),
                }
            )
            loss.backward()

            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_loader:
                inputs = inputs.to(dev)
                targets = targets.to(dev)

                outputs = model(inputs)
                val_loss += (
                    mse(outputs, targets).item() #+ gamma * stl(inputs, outputs).item()
                )
            val_losses.append(val_loss / len(val_loader))

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Train Total Loss: {train_losses[-1]['total loss']}, Train MSE Loss: {train_losses[-1]['mse loss']}, Train STL Loss: {train_losses[-1]['stl loss']} Val Loss: {val_losses[-1]}"
        )

    model = model.cpu()
    losses = {"train losses": train_losses, "val losses": val_losses}
    return model, losses


def save_results(model, losses, save_folder, args):
    torch.save(model.state_dict(), save_folder / "model.pt")

    cfg_file = save_folder / "cfg.json"
    cfg_file.write_text(json.dumps(vars(args)))

    write_obj(losses, save_folder / "losses.pkl")
    logging.info("model, losses, and cfg saved.")
