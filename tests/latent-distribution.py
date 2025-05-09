import sys
import os

# Move up one directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
os.chdir("..")

import numpy as np
from matplotlib import pyplot as plt
from config import config
import zlib
import zlib

num_images = 32
experiment_dir = config.results_dir

def load_latent_vector(path, dtype=np.float32):
    fn = os.path.join(experiment_dir, path)
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Missing file: {fn}")
    with open(fn, 'rb') as f:
        compressed = f.read()
    decompressed = zlib.decompress(compressed)
    return np.frombuffer(decompressed, dtype=dtype)

latent_list = [load_latent_vector(f'latent_{i}.zz') for i in range(num_images)]
latent_stack = np.vstack(latent_list)                   # (32, D)
abs_stack = np.abs(latent_stack)                        # (32, D)
sorted_stack = np.sort(abs_stack, axis=1)[:, ::-1]      # (32, D)
mean_per_rank = np.mean(sorted_stack, axis=0)           # (D,)
ranks = np.arange(1, mean_per_rank.size + 1)

plt.scatter(ranks, mean_per_rank)
plt.xlabel('Rank')
plt.ylabel('Average Magnitude of Activation')
plt.title('Latent Vector Distribution')
plt.tight_layout()
plt.savefig('tests/latent-distribution.png')
plt.show()