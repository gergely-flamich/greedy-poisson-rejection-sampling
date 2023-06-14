import jax.numpy as jnp


def laplace_kl(q_loc, q_scale, p_loc, p_scale):
    delta = jnp.abs(q_loc - p_loc)
    log_scale_ratio = jnp.log(p_scale) - jnp.log(q_scale)

    kl = log_scale_ratio = jnp.exp(-(log_scale_ratio + delta / q_scale))
    kl = kl + delta / p_scale - 1.

    return kl