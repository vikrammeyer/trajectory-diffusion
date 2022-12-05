import torch
from torch.utils.data import DataLoader

from ema_pytorch import EMA
from tqdm.auto import tqdm
from diff_traj.diffusion.diffusion_utils import *
from diff_traj.viz import Visualizations


class Trainer1D:
    def __init__(
        self,
        diffusion_model,
        dataset,
        cfg,
        results_folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25
    ):
        super().__init__()

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = diffusion_model.to(self.dev)

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.seq_length = diffusion_model.seq_length

        self.ds = dataset
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True)

        self.dl = cycle(dl)

        self.opt = torch.optim.Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = results_folder

        self.step = 0

        self.viz = Visualizations(cfg)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict()
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.dev)

        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

    def train(self):
        un_norm = self.ds.un_normalize

        # plot ground truth trajectories for the test scenarios
        test_trajs, test_params = next(self.dl)
        test_trajs_np, test_params_np = test_trajs.squeeze().cpu().numpy(), test_params.cpu().numpy()
        for i in range(test_trajs.shape[0]):
            traj, param = un_norm(test_trajs_np[i], test_params_np[i])
            self.viz.save_trajectory(traj, param, self.results_folder/f'gt-{i}.png')

        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    trajs, params = next(self.dl)
                    trajs, params = trajs.to(self.dev), params.to(self.dev)

                    loss = self.model(trajs, cond_vecs = params)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                self.ema.to(self.dev)
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()
                    milestone = self.step // self.save_and_sample_every

                    with torch.no_grad():
                        sampled_trajs = self.ema.ema_model.sample(cond_vecs = test_params).detach().squeeze().cpu().numpy()

                        for i in range(sampled_trajs.shape[0]):
                            traj, param = un_norm(sampled_trajs[i], test_params_np[i])
                            self.viz.save_trajectory(traj, param, self.results_folder/f'{milestone}-{i}.png')

                    self.save(milestone)

                pbar.update(1)
