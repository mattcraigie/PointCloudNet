import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time


def gaussian_random_field(spectral_index, resolution):
    kspace = torch.arange(-resolution // 2, resolution // 2, dtype=torch.float32)
    kx, ky = torch.meshgrid(kspace, kspace)
    k_mags = torch.sqrt(kx ** 2 + ky ** 2)
    k_mags = k_mags ** -spectral_index
    k_mags[-resolution // 2, resolution // 2] = 0

    # make a grf in k-space
    grf = torch.sqrt(k_mags) * torch.randn(resolution, resolution) * torch.exp(2j * np.pi * torch.rand(resolution, resolution))

    # make a grf in real space
    grf = torch.fft.ifft2(torch.fft.fftshift(grf)).real

    return grf


def sample(density, num_points):

    density = torch.relu(density)

    # Normalize density values to probabilities
    density_probs = density / torch.sum(density)
    flat_probs = density_probs.flatten()

    # Sample from the flattened probability distribution
    sample_indices = torch.multinomial(flat_probs, num_points, replacement=True)
    positions = torch.from_numpy(np.array(np.unravel_index(sample_indices, density.shape)).T)
    positions = positions.float()
    positions += (torch.rand_like(positions) - 0.5)
    positions /= resolution
    return positions


# Import the L-picola power spectrum as a starting point for the GRF
data = pd.read_table("input_spectrum.dat", sep='\s+').drop(columns=['P']).rename(columns={"#": "k", "k/h": "P"})

tstart = time.time()

num_mocks = 1000
num_points = 1000

box_size = 1
resolution = 128
spectral_index_values = torch.linspace(0, 3, num_mocks)


all_mocks = []
for i, n in enumerate(spectral_index_values):
    mocks_points = []

    density = gaussian_random_field(n, resolution)

    sample_pts = sample(density, num_points)

    torch.save(sample_pts, 'data/mock_{:04d}.pt'.format(i))

tend = time.time()

print('for {} it takes t={:.3f}'.format(num_mocks, tend - tstart))

torch.save(spectral_index_values.unsqueeze(1), 'data/k_values.pt')
