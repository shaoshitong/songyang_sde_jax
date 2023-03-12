import jax
import jax.numpy as jnp


def weight_fn(i, sum):
    N = 1000
    now_t = i / N
    diff_step = 1 / N
    beta_0 = 0.1
    beta_1 = 20
    log_alphas = -0.25 * (now_t - jnp.asarray(diff_step)) ** 2 * (
            beta_1 - beta_0) \
                 - 0.5 * (now_t - jnp.asarray(diff_step)) * beta_0
    log_sigmas = 0.5 * jnp.log(1. - jnp.exp(2. * log_alphas) + 1e-5)
    lambdas = log_alphas - log_sigmas
    next_log_alphas = -0.25 * (now_t) ** 2 * (beta_1 - beta_0) \
                      - 0.5 * (now_t) * beta_0
    next_log_sigmas = 0.5 * jnp.log(1. - jnp.exp(2. * next_log_alphas) + 1e-5)
    next_lambdas = next_log_alphas - next_log_sigmas
    return jnp.concatenate(((jnp.exp(-next_lambdas)-jnp.exp(-lambdas))[None,...],sum))
devices = jax.local_devices(backend="gpu")[0]
sum = jax.device_put(jnp.array([0.]),devices)
for i in range(1,1000):
    sum = weight_fn(i,sum)
result = (sum).tolist() / jnp.sum(sum)
print(result)
