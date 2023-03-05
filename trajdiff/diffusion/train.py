import torch
import logging

from trajdiff.diffusion.diffusion_utils import *

def train(
    diffusion_model,
    dataset,
    results_folder,
    *,
    train_batch_size=16,
    train_lr=1e-4,
    train_num_steps=100000,
    adam_betas=(0.9, 0.99),
    save_and_sample_every=1000):

    step = 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = diffusion_model.to(dev)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True
    )

    dataloader = cycle(dataloader)

    opt = torch.optim.Adam(model.parameters(), lr=train_lr, betas=adam_betas)

    while step < train_num_steps:
        params, trajs = next(dataloader)
        trajs, params = trajs.to(dev), params.to(dev)

        loss = model(trajs, cond_vecs=params)
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


@torch.inference_mode()
def sample(diffusion_model, test_dataset):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=None, pin_memory=True
    )

    for normalized_params, normalized_gt_trajs in dataloader:
        normalized_sample = diffusion_model.sample(cond_vecs=params.to(dev)).cpu()

        sampled_traj, _ = test_dataset.unnormalize(normalized_sample, normalized_params)
        gt_traj, params = test_dataset.unnormalize(normalized_gt_trajs, normalized_params)

        yield (params, gt_traj, sampled_traj)

    logging.info('finished sampling from model')

def save(model, optimizer, results_folder, step: int, milestone: str):
    data = {
        "step": step,
        "model": model.state_dict(),
        "opt": optimizer.state_dict()
    }

    milestone_path = str(results_folder / f"checkpoint-{milestone}.pt")
    torch.save(data, milestone_path)

    logging.info("milestone %s model saved", str(milestone))

def load(self, checkpoint_path, dev):
    data = torch.load(checkpoint_path, map_location=dev)

    self.model.load_state_dict(data["model"])
    self.step = data["step"]
    self.opt.load_state_dict(data["opt"])

    logging.info('loaded model and optimizer from %s', checkpoint_path)
