import logging

import torch
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from trajdiff.diffusion.diffusion_utils import *


class Trainer1D:
    def __init__(
        self,
        diffusion_model,
        dataset,
        results_folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
    ):
        super().__init__()

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = diffusion_model.to(self.dev)

        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.seq_length = diffusion_model.seq_length

        self.ds = dataset
        dl = DataLoader(
            self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True
        )

        self.dl = cycle(dl)

        self.opt = torch.optim.Adam(
            diffusion_model.parameters(), lr=train_lr, betas=adam_betas
        )

        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.results_folder = results_folder

        self.step = 0

    def save(self, milestone: str):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
        }

        milestone_path = str(self.results_folder / f"model-{milestone}.pt")
        torch.save(data, milestone_path)
        logging.info("milestone %s model saved", milestone)

    def load(self, checkpoint_path):
        data = torch.load(checkpoint_path, map_location=self.dev)

        self.model.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

    def train(self):
        while self.step < self.train_num_steps:

            total_loss = 0.0

            for _ in range(self.gradient_accumulate_every):
                params, trajs = next(self.dl)
                trajs, params = trajs.to(self.dev), params.to(self.dev)

                loss = self.model(trajs, cond_vecs=params)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.opt.step()
            self.opt.zero_grad()

            self.step += 1
            self.ema.to(self.dev)
            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema.ema_model.eval()
                milestone = self.step // self.save_and_sample_every
                logging.info("milestone %d loss: %f", milestone, total_loss)
                self.save(str(milestone))

        self.save("finished")

    @torch.inference_mode()
    def sample(self, test_dataset):
        self.ema.ema_model.eval()
        test_dl = DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True)
        for params, gt_trajs in test_dl:
            yield (
                gt_trajs,
                params,
                self.ema.ema_model.sample(cond_vecs=params.to(self.dev)).cpu(),
            )
