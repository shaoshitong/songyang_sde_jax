import jax
import jax.numpy as jnp


def weight_fn(i, sum):
    N = 1000
    now_t = i / N
    diff_step = 1 / N
    beta_0 = 0.1
    beta_1 = 20
    pred_t = now_t - jnp.asarray(diff_step)
    pred_t = 0 if pred_t.item() < 0 else pred_t
    log_alphas = -0.25 * (pred_t) ** 2 * (
            beta_1 - beta_0) \
                 - 0.5 * (pred_t) * beta_0
    log_sigmas = 0.5 * jnp.log(1. - jnp.exp(2. * log_alphas) + 1e-5)
    lambdas = log_alphas - log_sigmas
    next_log_alphas = -0.25 * (now_t) ** 2 * (beta_1 - beta_0) \
                      - 0.5 * (now_t) * beta_0
    next_log_sigmas = 0.5 * jnp.log(1. - jnp.exp(2. * next_log_alphas) + 1e-5)
    next_lambdas = next_log_alphas - next_log_sigmas

    return sum.append((jnp.exp(-next_lambdas) - jnp.exp(-lambdas)))


devices = jax.local_devices(backend="gpu")[0]
sum = []
for i in range(1, 1000):
    weight_fn(i, sum)
result = 999 * jnp.asarray(sum) / jnp.sum(jnp.asarray(sum))
print(result)
