import torch
import logging
from einops import rearrange
from trajdiff.diffusion.diffusion_utils import *

def train(
    diffusion_model,
    dataset,
    results_folder,
    *,
    batch_size=16,
    lr=1e-4,
    num_train_steps=100000,
    adam_betas=(0.9, 0.99),
    save_and_sample_every=1000):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("using %s", dev)
    model = diffusion_model.to(dev)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    dataloader = cycle(dataloader)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=adam_betas)

    n_agents = dataset.n_agents

    step = 0
    while step < num_train_steps:
        history, future = next(dataloader)
        history, future = history.to(dev), future.to(dev)

        # randomly select an agent in each situation to generate a trajectory prediction for
        batch = history.shape[0] # don't use batch_size cause we last batch not necessarily that size
        select_agents = torch.randint(low=0, high=n_agents, size=(batch,))

        # [B, traj_length, statedim (channels)]
        future_traj = future[torch.arange(batch), select_agents, :, :]
        # model expects trajectories of shape: [B, channels, seq_length]
        future_traj = rearrange(future_traj, "batch seq_len channels -> batch channels seq_len")
        # [B, statedim (channels- assumed to be 2 in the Unet1D agent of interest encoder)]
        init_states = history[torch.arange(batch), select_agents, -1, :]

        # set transformer encoder expects cond_vec of shape [B, n_agents, seq_length, channel]
        # (channels will be flattened in the set transformer encoder)
        loss = model(future_traj, init_states=init_states, cond_vecs=history)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        step += 1

        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            logging.info("milestone %d loss: %f", milestone, loss.item())
            save(model, opt, results_folder, step, milestone)

    save(model, opt, results_folder, step, 'final')

    logging.info('finished training.')


def save(model, optimizer, results_folder, step: int, milestone: str):
    data = {
        "step": step,
        "model": model.state_dict(),
        "opt": optimizer.state_dict()
    }

    milestone_path = str(results_folder / f"checkpoint-{milestone}.pt")
    torch.save(data, milestone_path)

    logging.info("milestone %s model saved", str(milestone))
