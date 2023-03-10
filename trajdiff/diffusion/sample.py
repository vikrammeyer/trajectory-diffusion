import torch
import logging

from trajdiff.utils import write_obj, default, get_device

@torch.inference_mode()
def sample(diffusion_model, test_dataset, checkpoint_path, output_folder, n=None, custom_seq_len=None):
    dev = get_device()
    load_model(diffusion_model, checkpoint_path, dev)

    diffusion_model = diffusion_model.to(dev)
    logging.info('model on %s', dev)

    # not batched
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=None, pin_memory=True
    )

    samples = {'history': [], 'future_gt': [], 'future_pred': []}

    n = default(n, len(test_dataset))

    i = 0
    chunk = 0
    for history, future in dataloader:
        history, future = history.to(dev), future.to(dev)

        all_agent_future_preds = diffusion_model.sample(cond_vec=history, custom_seq_len=custom_seq_len)

        samples['history'].append(history)
        samples['future_gt'].append(future)
        samples['future_pred'].append(all_agent_future_preds)

        i += 1
        if i % 1000 == 0 or i == n:
            write_obj(samples, output_folder/f'samples{chunk}.pkl')
            logging.info('saved chunk %d', chunk)
            chunk += 1
            samples = {'history': [], 'future_gt': [], 'future_pred': []}
            if i == n: break

    logging.info('finished sampling from model')

def load_model(model, checkpoint_path, dev):
    data = torch.load(checkpoint_path, map_location=dev)

    model.load_state_dict(data["model"])

    logging.info('loaded trained model from %s', checkpoint_path)
