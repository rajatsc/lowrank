import sys, os
import cv2

# add ../src to system path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['font.size'] = '8'
import numpy as np

parent_path = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(parent_path,'src'))

# import DMD from ../src
from dmd import DMD
from utils import VideoIO

def find_model(vid, model, start):
    # find the DMD modes for the first 30 models
    x = vid.stack_frames(start, num_snapshots, gray=True)
    model.solve(x, 1 / (vid.fps - 1))
    # create mask and evolve
    mask = model.create_mask(threshold=0.1)
    x_approx_lowrank = model.evolve(mask=mask)

    return model, x_approx_lowrank

if __name__ == "__main__":

    videofile = 'a.mp4'
    video_path = os.path.join(parent_path, 'data', videofile)
    start = 351
    num_snapshots = 20
    projected = True

    vid = VideoIO(video_path)
    model = DMD(projected = projected)

    start_list = []
    start_copy = start + num_snapshots
    while start_copy + num_snapshots <= vid.frame_count:
        start_list.append(start_copy)
        start_copy = start_copy + num_snapshots
    start_set = set(start_list)

    model, x_approx_lowrank = find_model(vid, model, start)

    # initialize figure
    fig, axs = plt.subplots(nrows = 1, ncols= 3)
    fig.tight_layout()
    axs[0].set_axis_off()
    axs[0].set_title("Original")
    axs[1].set_axis_off()
    axs[1].set_title("DMD low rank (background)")
    axs[2].set_axis_off()
    axs[2].set_title("DMD sparse (foreground)")

    img = model.snapshots[:, 0]
    img_lowrank = np.real(x_approx_lowrank[:, 0])
    img_sparse = img - img_lowrank

    # taking care of negative residuals
    img_sparse_copy = np.copy(img_sparse)
    img_sparse_copy[img_sparse_copy > 0] = 0

    img_sparse = img_sparse + np.abs(img_sparse_copy)
    img_sparse = img_sparse * 10
    img_lowrank = img_lowrank + img_sparse_copy

    im1 = axs[0].imshow(img.reshape(model.dim), animated=True, cmap='gray')
    im2 = axs[1].imshow(img_lowrank.reshape(model.dim), animated=True, cmap='gray')
    im3 = axs[2].imshow(img_sparse.reshape(model.dim), animated=True, cmap='gray')

    current_start = start
    def animate(i):

        global model, current_start, x_approx_lowrank

        if i in start_set:
            # find the new model
            model, x_approx_lowrank = find_model(vid, model, i)
            current_start = i

        idx = i-current_start
        img = model.snapshots[:, idx]
        img_lowrank = np.real(x_approx_lowrank[:, idx])
        img_sparse = img - img_lowrank

        # taking care of negative residuals
        img_sparse_copy = np.copy(img_sparse)
        img_sparse_copy[img_sparse_copy > 0] = 0

        img_sparse = img_sparse + np.abs(img_sparse_copy)
        img_sparse = img_sparse * 10
        img_lowrank = img_lowrank + img_sparse_copy

        im1.set_array(img.reshape(model.dim))
        im2.set_array(img_lowrank.reshape(model.dim))
        im3.set_array(img_sparse.reshape(model.dim))
        return [im1, im2, im3]


    anim = FuncAnimation(fig,
                         animate,
                         frames=np.arange(start=start, stop=start_list[-1]+num_snapshots),
                         interval=5,
                         blit=True)

    plt.show()



