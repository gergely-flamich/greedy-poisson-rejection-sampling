import jax.numpy as jnp
from jax import random

from tensorflow_probability.substrates.jax.distributions import Triangular, Uniform

from .util import trunc_gumbel


def kld_triangular_uniform(q: Triangular, p: Uniform):
  """Computes the KL-divergence between a triangular and a uniform distribution."""
  return jnp.where((p.low <= q.low) & (q.high <= p.high),
                    jnp.log(2.) - jnp.log(q.high - q.low) + jnp.log(p.high - p.low) - 0.5,
                    jnp.inf)


def infd_triangular_uniform(q: Triangular, p: Uniform):
  """Computes the infinity-divergence between a triangular and a uniform distribution."""
  return jnp.where((p.low <= q.low) & (q.high <= p.high),
                    jnp.log(2) + jnp.log(p.high - p.low) - jnp.log(q.high - q.low),
                    jnp.inf)


def log_inv_stretch(t, low, high):
  """Computes the logarithm of the inverse stretch function, assuming a standard
  uniform proposal and a triangular with support in [0, 1]."""

  return jnp.log(2.) + jnp.log(t) - jnp.log(2. + (high - low) * t)


def triangular_uniform_encode_example(seed: int, 
                                      proposal: Uniform,
                                      target: Triangular,
                                      max_iter: int = 10_000):

  proposal_width = proposal.high - proposal.low

  # Rescale the target so that the proposal is a standard uniform
  target = Triangular(low=(target.low - proposal.low) / proposal_width,
                      high=(target.high - proposal.low) / proposal_width,
                      peak=(target.peak - proposal.low) / proposal_width)

  key = random.PRNGKey(seed)

  log_time = -jnp.inf

  for i in range(max_iter):
    base_key = random.fold_in(key, i)
    time_key, x_key = random.split(base_key)

    log_time = -trunc_gumbel(time_key, (), loc=0., bound=-log_time)
    x = random.uniform(x_key)
    log_ratio = target.log_prob(x)

    if log_ratio > log_inv_stretch(jnp.exp(log_time), target.low, target.high):
      x = x * proposal_width + proposal.low

      return x, i

  raise ValueError("Sampler did not terminate!")


def triangular_uniform_decode(seed: int, index: int, proposal: Uniform):
  key = random.PRNGKey(seed)
  base_key = random.fold_in(key, index)
  _, x_key = random.split(base_key)

  x = random.uniform(x_key)

  x = x * (proposal.high - proposal.low) + proposal.low

  return x