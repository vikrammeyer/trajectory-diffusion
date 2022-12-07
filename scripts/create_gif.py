from PIL import Image
from pathlib import Path

def create_gif_from_pngs(img_paths, output_path):
    frames = [Image.open(i) for i in img_paths]

    im = Image.new(frames[0].mode, frames[0].size)
    im.save(output_path, format='GIF', append_images=frames, save_all=True, duration=500, loop=1)

if __name__ == '__main__':
    i_traj = int(input('enter i-th traj: ').strip())
    output_path = Path('results/sample-diff-gifs')
    if not output_path.exists(): output_path.mkdir()
    pics_folder = Path('results/sample-diff-pics')

    img_paths = [pics_folder/f"checkpt-{checkpoint}/{i_traj}.png" for checkpoint in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]]
    create_gif_from_pngs(img_paths, output_path/f"{i_traj}.gif")