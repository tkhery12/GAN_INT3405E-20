import argparse
import datetime
import glob
import itertools
import logging
import math
import os
import random
import re
import shutil
from glob import glob
from pathlib import Path

import PIL
import cv2
import imageio
import matplotlib.pyplot as plt
import natsort
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import make_grid
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "4"



class NetG_MNIST(nn.Module):
    def __init__(self, latent_dim, image_shape, feature_size=64):
        super(NetG_MNIST, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.feature_size = feature_size

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.feature_size * 4,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.feature_size * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_size * 4, self.feature_size * 2,
                               kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 2),
            nn.ReLU(True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_size * 2, self.feature_size,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_size, self.image_shape[0],
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out


class NetD_MNIST(nn.Module):
    def __init__(self, image_shape, feature_size, loss_function="mse"):
        super(NetD_MNIST, self).__init__()
        self.image_shape = image_shape
        self.feature_size = feature_size
        self.loss_function = loss_function

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.image_shape[0], self.feature_size,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.feature_size, self.feature_size * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.feature_size * 2, self.feature_size * 4,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.feature_size * 4, 1,
                      kernel_size=4, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        if self.loss_function == "bce":
            out = self.sigmoid(out)
        return out.view(-1, 1)


def test_MNIST():
    z = torch.randn(128, 100)
    G = NetG_MNIST(latent_dim=100, image_shape=(1, 28, 28), feature_size=64)
    # img = torch.randn(128, 1, 28, 28)
    D = NetD_MNIST(image_shape=(1, 28, 28), feature_size=64, loss_function="bce")
    print(G(z).shape)
    print(D(G(z)).shape)


# from utils.general import weights_init
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def generate_images(epoch, path, fixed_noise, num_test_samples, netG, device, use_fixed=False):
    z = torch.randn(num_test_samples, 100, 1, 1, device=device)
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None

    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(z)
        path += 'variable_noise/'
        title = 'Variable Noise'

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k // 4
        j = k % 4
        ax[i, j].cla()
        ax[i, j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28, 28), cmap='Greys')
    label = 'Epoch_{}'.format(epoch + 1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path + label + '.png')


def save_gif(path, fps, fixed_noise=False):
    if fixed_noise:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path + 'animated.gif', gif, fps=fps)


def make_gif(folder, pattern="*.png", file_path='./out.gif'):
    images = []
    for filename in sorted(glob.glob(os.path.join(folder, pattern)),
                           key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]):
        # print(filename)
        images.append(imageio.imread(filename))
    imageio.mimsave(file_path, images)


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def create_dif(folder, pattern, output_path="out.gif"):
    with imageio.get_writer(output_path, mode="I") as writer:
        filenames = glob.glob(folder + pattern)
        filenames = sorted(filenames, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def getFrame(video_capture, sec, count, save_dir):
    video_capture.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = video_capture.read()
    if hasFrames:
        cv2.imwrite(f"{save_dir}/image_{count:04d}.jpg", image)  # save frame as JPG file
    return hasFrames


def video_to_frames(video_path, save_dir="./", fps=24):
    video_capture = cv2.VideoCapture(video_path)

    sec = 0
    frameRate = 1 / fps  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(video_capture, sec, count, save_dir)

    while success:
        print(count)
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(video_capture, sec, count, save_dir)


def frames_to_video(frames, video_path, fps=24):
    height, width = frames[0].shape[:2]
    size = (width, height)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frames)):
        # writing to an image array
        out.write(frames[i].astype('uint8'))
    out.release()


def frame_files_to_video(frame_files=None, video_path=None, fps=24):
    frame_array = []
    for filename in frame_files:
        # reading each files
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def load_image(image_path, color_space="rgb"):
    image = cv2.imread(image_path)

    assert image is not None

    if color_space == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError

    return image


# import preprocess_data
cuda = True if torch.cuda.is_available() else False


def generate_dataloader(name_dataset, img_size, batch_size):
    # Configure data loader

    # MNIST
    # Image size: (1, 28, 28)
    if name_dataset == 'mnist':
        os.makedirs('../datasets/mnist', exist_ok=True)

        dataloader_mnist = torch.utils.data.DataLoader(
            datasets.MNIST(
                '../datasets/mnist',
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
        )
        return dataloader_mnist


# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_noise(n_samples, z_dim):
    return torch.randn(n_samples, z_dim, device=device)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    if one_channel:
        plt.imshow(np_img, cmap="Greys")
    else:
        plt.imshow(np.transpose(np_img, (1, 2, 0)))


def show_tensor_images(image_tensor, writer, type_image, step, num_images=25):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflatt = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflatt[:num_images], nrow=5, normalize=True)
    # show images
    # matplotlib_imshow(image_grid, one_channel=True)
    # add tensorboard
    writer.add_image(type_image, image_grid, global_step=step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="size of the batches")

    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--loss_function", type=str, default="mse",
                        help="Loss Function", choices=["mse", "bce"])

    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--feature_size", type=int, default=64,
                        help="dimensionality of the feature")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "celeba", "cifar10"])
    parser.add_argument("--display_step", type=int, default=1000,
                        help="interval between image samples")
    parser.add_argument("--save_checkpoint_step", type=int, default=50000,
                        help="Saving checkpoint after step")

    parser.add_argument("--gpu", type=str, default='0', help='Specify GPU ')
    parser.add_argument('--log_dir', type=str, default="gan",
                        help='experiment root', choices=["gan", "dcgan"])
    return parser.parse_args("")


def main():
    global img_size, channels, adversarial_loss, generator, discriminator

    def log_string(string):
        logger.info(string)
        print(string)

    args = parse_args()
    log_model = args.log_dir + "_n_epochs_" + str(args.n_epochs)
    log_model = log_model + "_batch_size_" + str(args.batch_size)
    log_model = log_model + "_loss_" + str(args.loss_function)
    log_model = log_model + "_display_step_" + str(args.display_step)
    log_model = log_model + "_" + args.dataset

    img_size = 28
    channels = 1

    '''CREATE DIR'''
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(time_str)
    else:
        experiment_dir = experiment_dir.joinpath(log_model)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, "log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string(args)
    log_string(log_model)

    '''TENSORBROAD'''
    log_string('Creating Tensorboard ...')
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensor_dir = experiment_dir.joinpath('tensorboard/')
    if tensor_dir.exists():
        shutil.rmtree(tensor_dir)
    tensor_dir.mkdir(exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(tensor_dir))

    # GPU Indicator
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Save generated images
    saved_path = experiment_dir.joinpath('images/')
    os.makedirs(saved_path, exist_ok=True)

    # Configure data loader
    dataloader = generate_dataloader(
        name_dataset=args.dataset,
        img_size=img_size,
        batch_size=args.batch_size
    )

    # Loss functions
    if args.loss_function == "bce":
        adversarial_loss = torch.nn.BCELoss()
    elif args.loss_function == "mse":
        adversarial_loss = torch.nn.MSELoss()

        # Initialize generator and discriminator

        generator = NetG_MNIST(
            latent_dim=args.latent_dim,
            image_shape=(channels, img_size, img_size),
            feature_size=args.feature_size
        )
        discriminator = NetD_MNIST(
            image_shape=(channels, img_size, img_size),
            feature_size=args.feature_size,
            loss_function=args.loss_function
        )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Assign device for model, criterion
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # ----------
    #  Training
    # ----------
    log_string("Starting Training Loop...")

    G_losses = []
    D_losses = []

    fixed_noise = torch.randn(args.batch_size, args.latent_dim, device=device)

    wrapper = torch.nn.Sequential(generator, discriminator)

    summary_writer.add_graph(wrapper, input_to_model=torch.randn(1, args.latent_dim).to(device))

    for epoch in range(args.n_epochs):
        for i, (images, _) in enumerate(dataloader):

            # Adversarial ground truths
            real_label = 1.
            fake_label = 0.

            # Configure input
            label = torch.full((images.size(0), 1), real_label, dtype=torch.float, device=device)
            real_images = images.to(device)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with all-real batch

            discriminator.zero_grad()
            output = discriminator(real_images)

            errD_real = adversarial_loss(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(images.size(0), args.latent_dim, device=device)

            gen_images = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(gen_images.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake = adversarial_loss(output, label)
            errD_fake.backward()
            # Add the gradients from the all-real and all-fake batches
            errD = (errD_real + errD_fake) / 2
            D_G_z1_tensor = output
            D_G_z1 = output.mean().item()
            # Update D
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            generator.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            label.fill_(real_label)
            output = discriminator(gen_images)
            # Calculate G's loss based on this output
            errG = adversarial_loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2_tensor = output
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_G.step()

            if i % 22 == 0:
                log_string(
                    "[Epoch %d/%d] [Batch %d/%d]"
                    "\t[Loss_D: %.4f]\t[Loss_G: %.4f]\t[D(x): %.4f]\t[D(G(z)): %.4f / %.4f]"
                    % (epoch, args.n_epochs, i, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
                )

            D_losses.append(errD.item())
            G_losses.append(errG.item())

            steps = epoch * len(dataloader) + i
            summary_writer.add_scalars(
                'Loss',
                {
                    'D': errD.item(),
                    'G': errG.item()
                },
                steps
            )
            summary_writer.add_scalar('D(x)', D_x, steps)
            summary_writer.add_scalar('D(G(z1))', D_G_z1, steps)
            summary_writer.add_scalar('D(G(z2))', D_G_z2, steps)

            if steps % args.display_step == 0:
                with torch.no_grad():
                    fake = generator(fixed_noise)

                    # save_image(real_images.data[:25], saved_path.joinpath("real_%d.png" % steps),
                    #            nrow=5, normalize=True)
                    save_image(fake.data[:25], saved_path.joinpath("%d.png" % steps),
                               nrow=5, normalize=True)

                    show_tensor_images(fake, summary_writer, "Fake Image", steps)
                    show_tensor_images(real_images, summary_writer, "Real Image", steps)

            # do checkpointing
            if steps % args.save_checkpoint_step == 0:
                torch.save(generator.state_dict(),
                           checkpoints_dir.joinpath(f"{args.log_dir}_G_iter_{steps}.pth"))
                torch.save(discriminator.state_dict(),
                           checkpoints_dir.joinpath(f"{args.log_dir}_D_iter_{steps}.pth"))

    # Save final checkpoint
    log_string('Saving checkpoint .......... ')
    torch.save(generator.state_dict(),
               checkpoints_dir.joinpath(f"{args.log_dir}_G_final.pth"))
    torch.save(discriminator.state_dict(),
               checkpoints_dir.joinpath(f"{args.log_dir}_D_final.pth"))

    # Plot lossy graph
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(experiment_dir.joinpath('graph.png'))
    summary_writer.add_figure("Graph Loss", plt.gcf())

    # Compress many images to gif file
    log_string('Saving gif file .......... ')
    make_gif(saved_path, file_path=experiment_dir.joinpath('out.gif'))
    log_string('Finishing training phase ...............')
    summary_writer.close()


if __name__ == '__main__':
    main()
