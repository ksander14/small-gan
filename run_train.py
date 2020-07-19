import argparse
import os
import numpy as np
import math
import sys
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from scipy.spatial.distance import cdist

from nets import Discriminator, Generator

class GaussianMixture():
    def __init__(self, grid_side_size, gaus_std=0.05, cell_size=2):
        '''

        :param grid_side_size: number of grid points by side
        :param gaus_std: gaussians standard deviation, default=0.05 from papers experiments
        :param cell_size: , default=2 from papers experiments
        '''
        self.grid_side_size = grid_side_size
        self.gaus_std = gaus_std
        self.cell_size = cell_size

        self.gaus_number = grid_side_size ** 2
        self.grid_side = np.linspace(-(grid_side_size - 1) * cell_size / 2,
                                     (grid_side_size - 1) * cell_size / 2,
                                     grid_side_size)
        self.gaus_means = [(i, j) for i in self.grid_side for j in self.grid_side]
        self.gaus_covar = np.eye(2) * gaus_std

    def sample(self, size):
        '''
        Sampling from gaussian distributions mixture
        '''

        # Sampling distributions and count every type
        samples_count_by_gaus = np.bincount(np.random.randint(self.gaus_number, size))

        # Sampling from distributions and appending for full array
        gen_samples = np.ones(0)
        for j, samples_count in enumerate(samples_count_by_gaus):
            gen_samples = np.append(gen_samples,
                                    np.random.multivariate_normal(self.gaus_means[j], self.gaus_covar, samples_count))

        # Shuffle elements
        np.random.shuffle(gen_samples)
        return gen_samples

    def evaluate(self, generated_samples, hq_std_number):
        '''
        :param np.array generated_samples: generated samples
        :param int hq_std_number: number of std for high quality samples distance
        :return: tuple (recovered_modes_percent, high_quality_percent)
        '''
        # L1 pairwise matrix
        distance_matrix = cdist(generated_samples, self.gaus_means)

        # Mins and argmins
        mins = distance_matrix.min(axis=1)
        argmins = distance_matrix.argmin(axis=1)

        # Calculate metrics
        high_quality = (mins < hq_std_number * self.gaus_std).mean()
        recovered_modes = np.unique(argmins) / self.gaus_number

        return recovered_modes, high_quality



# def greedy_core_set(x, k, return_index=False):
#     '''
#     Implementation of GreedyCoreSet algorithm from paper
#
#     :param x: full minibatch
#     :param k: size of subset
#     :param return_index: return samples or indexes for image embedding case
#     :return:
#     '''
#     t1 = time.time()
#     # Return full batch if k >= |x|
#     if x.shape[0] < (k + 1):
#         return list(np.arange(x.shape[0])) if return_index else x
#
#     # For saving calculated distances
#     distances = lil_matrix((x.shape[0], x.shape[0]))
#     other_indexes = list(np.arange(x.shape[0]))
#
#     # Randomly select the first element
#     first_point = np.random.randint(x.shape[0])
#     subset_indexes = [first_point]
#     other_indexes.remove(first_point)
#     distances[first_point] = ((x - x[first_point]) ** 2).sum(axis=1).sqrt()
#
#     # Adding other elements
#     while len(subset_indexes) < k:
#         new_point = distances[subset_indexes, :][:, other_indexes].toarray().min(axis=0).argmax()
#         new_point = other_indexes[new_point]
#         subset_indexes.append(new_point)
#         other_indexes.remove(new_point)
#         distances[new_point] = ((x - x[new_point]) ** 2).sum(axis=1).sqrt()
#
#     t2 = time.time()
#     print(t2-t1)
#     return subset_indexes if return_index else x[subset_indexes]


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def greedy_core_set(x, k, return_index=False):
    '''
    Implementation of GreedyCoreSet algorithm from paper

    :param x: full minibatch
    :param k: size of subset
    :param return_index: return samples or indexes for image embedding case
    :return:
    '''
    # t1 = time.time()
    # Return full batch if k >= |x|
    if x.shape[0] < (k + 1):
        return list(np.arange(x.shape[0])) if return_index else x

    # Fobjghr saving calculated distances
    distances = pairwise_distances(x, x)
    other_indexes = list(np.arange(x.shape[0]))

    # Randomly select the first element
    first_point = np.random.randint(x.shape[0])
    subset_indexes = [first_point]
    other_indexes.remove(first_point)

    # Adding other elements
    while len(subset_indexes) < k:
        new_point = distances[subset_indexes, :][:, other_indexes].min(axis=0)[0].argmax()
        new_point = other_indexes[new_point]
        subset_indexes.append(new_point)
        other_indexes.remove(new_point)

    # t2 = time.time()
    # print(t2-t1)
    return subset_indexes if return_index else x[subset_indexes]

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--learning_type", type=str, default="gan", help="Type of learning - smallgan or gan")
parser.add_argument("--test_type", type=str, default="gaus", help="Learning type - image (for testing on MNIST) "
                                                                  "or gaussian (for gaussian mixture experiments)")
parser.add_argument("--n_epochs", type=int, default=200, help="number of standard epochs of training, for smallgan it "
                                                              "will be n_epochs*target_factor")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--prior_factor", type=int, default=4, help="Over-sampling factor for prior distribution p(z)")
parser.add_argument("--target_factor", type=int, default=8, help="Over-sampling factor for target distribution p(x)")
parser.add_argument("--random_seed", type=int, default=0, help="Random seed value for reproducibility")


parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=2, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")


opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

#cuda = True if torch.cuda.is_available() else False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(opt.latent_dim, np.prod(img_shape)).to(device)
discriminator = Discriminator(np.prod(img_shape)).to(device)
#
# if cuda:
#     generator.cuda()
#     discriminator.cuda()

# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size * opt.target_factor if opt.learning_type == 'smallgan' else opt.batch_size,
    shuffle=True,
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0

if opt.learning_type == 'smallgan':
    opt.n_epochs *= opt.target_factor


for epoch in range(opt.n_epochs):
    for i, (batch, _) in enumerate(dataloader):
        # Configure input and apply GreedyCoreset
        real_data = batch.type(Tensor).view(batch.shape[0], -1)
        if opt.learning_type == 'smallgan':
            real_data = Variable(greedy_core_set(real_data, opt.batch_size))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input and apply GreedyCoreset
        z = Tensor(np.random.normal(0, 1, (batch.shape[0], opt.latent_dim)))
        if opt.learning_type == 'smallgan':
            z = Variable(greedy_core_set(z, opt.batch_size))
        else:
            z = Variable(z)

        # Generate a batch of images
        fake_data = generator(z)

        # Real images
        real_validity = discriminator(real_data)
        # Fake images
        fake_validity = discriminator(fake_data)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_data = generator(z)
            #print(fake_data[0])
            #print(fake_data[1])
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_data)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                save_image(fake_data.view(fake_data.shape[0], *img_shape).data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic


