import jax.numpy as jnp
from jax import random


def log1mexp(x):
    x = jnp.abs(x)
    return jnp.where(x < jnp.log(2.), jnp.log(-jnp.expm1(-x)), jnp.log1p(-jnp.exp(-x)))


def logsubexp(x, y, return_sign = False):
    larger = jnp.maximum(x, y)
    smaller = jnp.minimum(x, y)

    result = larger + log1mexp(jnp.maximum(larger - smaller, 0.))
    result = jnp.where(larger == -jnp.inf, -jnp.inf, result)

    if return_sign:
        return result, jnp.where(x < y, -1., 1.)

    return result


def trunc_gumbel(key, shape, loc, bound):
    """
    Samples a Gumbel variate truncated below the given bound with location loc
    """
    u = random.uniform(key, shape)
    g = -jnp.log(u) + jnp.exp(-bound + loc)
    g = loc - jnp.log(g)

    return g
