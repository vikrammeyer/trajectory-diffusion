import torch
import wandb

from ema_pytorch import EMA
from tqdm.auto import tqdm
from diff_traj.viz import Visualizations
from diff_traj.diffusion.diffusion_utils import cycle
from diff_traj.utils.eval import n_collision_states, dynamics_violations

class Trainer:
    def __init__(
        self,
        model,
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
        num_samples = 25,
        debug_mode=False
    ):
        super().__init__()

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.dev)

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.dl = cycle(torch.utils.data.DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True))

        self.opt = torch.optim.Adam(self.model.parameters(), lr = train_lr, betas = adam_betas)

        self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = results_folder

        self.step = 0

        self.viz = Visualizations(cfg)

        self.debug_mode = debug_mode
        if not self.debug_mode:
            self.run = wandb.init(project="trajectory-diffusion", entity="vikram-meyer", job_type="train_fcnet")

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict()
        }

        milestone_path = str(self.results_folder / f'model-{milestone}.pt')
        torch.save(data, milestone_path)

        if not self.debug_mode:
            artifact = wandb.Artifact(f'model-{milestone}', type='checkpoint')
            artifact.add_file(milestone_path)

            self.run.log_artifact(artifact)

    def load(self, checkpoint_path):
        data = torch.load(checkpoint_path, map_location=self.dev)

        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

    def train(self):
        un_norm = self.ds.un_normalize

        # Test scenarios
        test_trajs, test_params = next(self.dl)
        test_trajs, test_params = test_trajs.to(self.dev), test_params.to(self.dev)
        test_trajs_np, test_params_np = test_trajs.squeeze().cpu().numpy(), test_params.cpu().numpy()

        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    trajs, params = next(self.dl)
                    trajs, params = trajs.squeeze().to(self.dev), params.to(self.dev)

                    preds = self.model(params)

                    loss = torch.nn.functional.mse_loss(trajs, preds)

                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                self.run.log({"loss": total_loss})

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                self.ema.to(self.dev)
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()
                    milestone = self.step // self.save_and_sample_every

                    with torch.no_grad():
                        predicted_trajs = self.ema.ema_model(test_params).squeeze().cpu().numpy()

                        for i in range(predicted_trajs.shape[0]):
                            traj, param = un_norm(predicted_trajs[i], test_params_np[i])
                            fname = str(self.results_folder/f'{milestone}-{i}.png')
                            self.viz.save_trajectory(traj, param, fname)

                    self.save(milestone)

                pbar.update(1)

        for i in range(test_trajs.shape[0]):
            traj, param = un_norm(test_trajs_np[i], test_params_np[i])
            fname = str(self.results_folder/f'gt-{i}.png')
            self.viz.save_trajectory(traj, param, fname)

        self.save('finished')

    @torch.inference_mode()
    def evaluate(self, test_dataset):
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        loss_fn = torch.nn.functional.mse_loss
        metrics = {'n_trajs': 0, 'n_collision_states': [], 'dynamics_violations': []}
        self.ema.ema_model.eval()

        for gt_trajs, params in test_dl:
            sampled_trajs = self.ema.ema_model(params.to(self.dev)).squeeze().cpu()
            gt_trajs = gt_trajs.squeeze()

            for i in range(sampled_trajs.shape[0]):
                metrics["n_trajs"] += 1

                traj, param = test_dataset.un_normalize(sampled_trajs[i], params[i])

                # Collision Free States Metric
                metrics['n_collision_states'].append(n_collision_states(traj, param))

                # Dynamics Violation Metric
                metrics['dynamics_violations'].append(dynamics_violations(traj))

                # MSE w Ground Truth Metric
                metrics['mse_gt'].append(loss_fn(sampled_trajs[i],gt_trajs[i]).item())

                self.viz.save_trajectory(traj, param, self.results_folder/f'{metrics["n_trajs"]}.png')

        return metrics
