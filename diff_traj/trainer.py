from pathlib import Path
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

from ema_pytorch import EMA
from tqdm.auto import tqdm
from multiprocessing import cpu_count
from .dataset import StateDataset
from .utils import *
from .viz import Visualizations


class Trainer1D:
    def __init__(
        self,
        diffusion_model,
        dataset_path,
        cfg,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.seq_length = diffusion_model.seq_length

        # dataset and dataloader

        self.ds = StateDataset(dataset_path)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True) # num_workers = cpu_count()

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = torch.optim.Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.viz = Visualizations(cfg)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        un_norm_trajs = self.ds.un_normalize

        # plot ground truth trajectories for the test scenarios
        test_trajs, test_params = next(self.dl)
        test_trajs_np, test_params_np = test_trajs.squeeze().numpy(), test_params.numpy()
        for i in range(test_trajs.shape[0]):
            self.viz.save_trajectory(un_norm_trajs(test_trajs_np[i]), test_params_np[i][:4], test_params_np[i][4:], self.results_folder/f'gt-{i}.png')

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    trajs, params = next(self.dl)

                    with self.accelerator.autocast():
                        loss = self.model(trajs, cond_vecs = params)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every

                        with torch.no_grad():
                            sampled_trajs = self.ema.ema_model.sample(cond_vecs = test_params).detach().squeeze().numpy()

                            for i in range(sampled_trajs.shape[0]):
                                self.viz.save_trajectory(un_norm_trajs(sampled_trajs[i]), test_params_np[i][:4], test_params_np[i][4:], self.results_folder/f'{milestone}-{i}.png')

                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')