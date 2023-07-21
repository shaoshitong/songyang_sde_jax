# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from utils import batch_mul
import optax
from flax.training import train_state


def schedule_fn(lr, step, config):
    warmup = config.optim.warmup
    if warmup > 0:
        lr = lr * jnp.minimum(step / warmup, 1.0)
    return lr


def get_optimizer(config):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optax.chain(
            optax.scale_by_schedule(lambda step: schedule_fn(lr=config.optim.lr, config=config, step=step)),
            optax.adam(learning_rate=config.optim.lr, b1=config.optim.beta1, eps=config.optim.eps))
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(state,
                    grad,
                    new_model_state,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        lr = state.lr
        if warmup > 0:
            lr = lr * jnp.minimum(state.step / warmup, 1.0)
        if grad_clip >= 0:
            # Compute global gradient norm
            grad_norm = jnp.sqrt(
                sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
            # Clip gradient
            clipped_grad = jax.tree_map(
                lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
        else:  # disabling gradient clipping if grad_clip < 0
            clipped_grad = grad
        return state.optimizer.apply_gradients(grads=clipped_grad)

    return optimize_fn


def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        """Compute the loss function.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """

        score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous,
                                       return_state=True)
        data = batch['image']

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, data.shape)
        mean, std = sde.marginal_prob(data, t)
        perturbed_data = mean + batch_mul(std, z)
        rng, step_rng = random.split(rng)
        score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

        if not likelihood_weighting:
            losses = jnp.square(batch_mul(score, std) + z)
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        else:
            g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
            losses = jnp.square(score + batch_mul(z, 1. / std))
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_kd_sde_loss_fn(sde, model, teacher_model, error_kd, train, reduce_mean=True, continuous=True,
                       likelihood_weighting=True,
                       eps=1e-5, **kwargs):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      error_kd: if use error to analysis KD.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, teacher_params, states, batch):
        """Compute the loss function.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """

        teacher_score_fn = mutils.get_score_fn(sde, teacher_model, teacher_params, states, train=train,
                                               continuous=continuous,
                                               return_state=True, is_teacher=True)
        student_score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous,
                                               return_state=True, is_teacher=False)

        data = batch['image']
        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, data.shape)
        mean, std = sde.marginal_prob(data, t)
        perturbed_data = mean + batch_mul(std, z)
        rng, step_rng = random.split(rng)
        student_score, new_model_state = student_score_fn(perturbed_data, t, rng=step_rng)

        rng, step_rng = random.split(rng)
        teacher_score, _ = teacher_score_fn(perturbed_data, t, rng=step_rng)
        if error_kd:
            diff_step = 1 / kwargs["diff_step"] if ("diff_step" in kwargs.keys()) else 1 / sde.N

            def weight_fn(i, sum):
                now_t = i / kwargs["diff_step"] if ("diff_step" in kwargs.keys()) else i / sde.N
                log_alphas = -0.25 * (now_t - jnp.asarray(diff_step)) ** 2 * (
                        sde.beta_1 - sde.beta_0) \
                             - 0.5 * (now_t - jnp.asarray(diff_step)) * sde.beta_0
                log_sigmas = 0.5 * jnp.log(1. - jnp.exp(2. * log_alphas) + 1e-5)
                lambdas = log_alphas - log_sigmas
                next_log_alphas = -0.25 * (now_t) ** 2 * (sde.beta_1 - sde.beta_0) \
                                  - 0.5 * (now_t) * sde.beta_0
                next_log_sigmas = 0.5 * jnp.log(1. - jnp.exp(2. * next_log_alphas) + 1e-5)
                next_lambdas = next_log_alphas - next_log_sigmas
                return sum + jnp.exp(-next_lambdas) - jnp.exp(-lambdas)

            pred_t = t - jnp.asarray(diff_step).broadcast((t.shape))
            pred_t = jnp.where(pred_t < 0, 0, pred_t)
            log_alphas = -0.25 * pred_t ** 2 * (sde.beta_1 - sde.beta_0) \
                         - 0.5 * pred_t * sde.beta_0
            log_sigmas = 0.5 * jnp.log(1. - jnp.exp(2. * log_alphas))
            lambdas = log_alphas - log_sigmas
            next_log_alphas = -0.25 * (t) ** 2 * (sde.beta_1 - sde.beta_0) \
                              - 0.5 * (t) * sde.beta_0
            next_log_sigmas = 0.5 * jnp.log(1. - jnp.exp(2. * next_log_alphas) + 1e-5)
            next_lambdas = next_log_alphas - next_log_sigmas
            final_weight = (jnp.exp(-next_lambdas) - jnp.exp(-lambdas))
            sum_weight = jax.lax.fori_loop(1, kwargs["diff_step"] if ("diff_step" in kwargs.keys()) else sde.N,
                                           weight_fn, jnp.asarray(0.))
            final_weight = final_weight * (
                kwargs["diff_step"] - 1 if ("diff_step" in kwargs.keys()) else sde.N - 1) / sum_weight
            final_weight = jnp.where(jnp.logical_or(jnp.isinf(final_weight), jnp.isnan(final_weight)), 1, final_weight)
        else:
            final_weight = jnp.ones_like(t)

        if not likelihood_weighting:
            origin_losses = jnp.square(batch_mul(student_score, std) + z)
            origin_losses = reduce_op(origin_losses.reshape((origin_losses.shape[0], -1)), axis=-1)
            kd_losses = batch_mul(jnp.square(batch_mul(student_score, std) - batch_mul(teacher_score, std)),
                                  final_weight)
            kd_losses = reduce_op(kd_losses.reshape((kd_losses.shape[0], -1)), axis=-1)
        else:
            g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
            origin_losses = jnp.square(student_score + batch_mul(z, 1. / std))
            origin_losses = reduce_op(origin_losses.reshape((origin_losses.shape[0], -1)), axis=-1) * g2
            kd_losses = batch_mul(jnp.square(student_score - teacher_score), final_weight)
            kd_losses = reduce_op(kd_losses.reshape((kd_losses.shape[0], -1)), axis=-1)

        kd_losses = batch_mul(kd_losses, jnp.asarray([kwargs["kd_weight"]]).broadcast((t.shape)))
        origin_losses = batch_mul(origin_losses, jnp.asarray([kwargs["ce_weight"]]).broadcast((t.shape)))
        losses = kd_losses + origin_losses

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_smld_loss_fn(vesde, model, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch['image']
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
        sigmas = smld_sigma_array[labels]
        rng, step_rng = random.split(rng)
        noise = batch_mul(random.normal(step_rng, data.shape), sigmas)
        perturbed_data = noise + data
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        target = -batch_mul(noise, 1. / (sigmas ** 2))
        losses = jnp.square(score - target)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * sigmas ** 2
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_kd_smld_loss_fn(vesde, model, teacher_model, error_kd, train, reduce_mean=False, **kwargs):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    smld_alpha_array = jnp.ones_like(smld_sigma_array)
    smld_lambda_array = jnp.log(smld_alpha_array / jnp.sqrt(smld_sigma_array + 1e-8))
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, teacher_params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train, is_teacher=False)
        teacher_model_fn = mutils.get_model_fn(teacher_model, teacher_params, states, train=train, is_teacher=True)
        data = batch['image']
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
        sigmas = smld_sigma_array[labels]
        rng, step_rng = random.split(rng)
        noise = batch_mul(random.normal(step_rng, data.shape), sigmas)
        perturbed_data = noise + data
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        rng, step_rng = random.split(rng)
        teacher_score, _ = teacher_model_fn(perturbed_data, labels, rng=step_rng)
        target = -batch_mul(noise, 1. / (sigmas ** 2))
        origin_losses = jnp.square(score - target)
        origin_losses = reduce_op(origin_losses.reshape((origin_losses.shape[0], -1)), axis=-1) * sigmas ** 2 * (
            kwargs["ce_weight"] if hasattr(kwargs, "ce_weight") else 1)

        if error_kd:
            lambdas = smld_lambda_array
            next_lambdas = jnp.concatenate((lambdas, lambdas[-1][None, ...]))
            now_lambdas = jnp.concatenate((lambdas[0][None, ...], lambdas))
            last_lambdas = jnp.exp(-next_lambdas) - jnp.exp(-now_lambdas)
            final_weight = last_lambdas.shape[0] * last_lambdas[labels] / jnp.sum(last_lambdas)

        else:
            final_weight = jnp.ones_like(labels)

        kd_losses = batch_mul(jnp.square(score - teacher_score), final_weight)
        kd_losses = reduce_op(kd_losses.reshape((kd_losses.shape[0], -1)), axis=-1) * sigmas ** 2 * (
            kwargs["kd_weight"] if hasattr(kwargs, "kd_weight") else 1)

        loss = jnp.mean(origin_losses + kd_losses)

        return loss, new_model_state

    return loss_fn


def get_ddpm_loss_fn(vpsde, model, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train)
        data = batch['image']
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, data.shape)
        perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + \
                         batch_mul(sqrt_1m_alphas_cumprod[labels], noise)
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        losses = jnp.square(score - noise)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_kd_ddpm_loss_fn(vpsde, model, train, teacher_model, error_kd, reduce_mean=True, **kwargs):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def loss_fn(rng, params, teacher_params, states, batch):
        model_fn = mutils.get_model_fn(model, params, states, train=train, is_teacher=False)
        teacher_model_fn = mutils.get_model_fn(teacher_model, teacher_params, states, train=train, is_teacher=True)

        data = batch['image']
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, data.shape)
        perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + \
                         batch_mul(sqrt_1m_alphas_cumprod[labels], noise)
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        rng, step_rng = random.split(rng)
        teacher_score, _ = teacher_model_fn(perturbed_data, labels, rng=step_rng)
        origin_losses = jnp.square(score - noise)
        origin_losses = reduce_op(origin_losses.reshape((origin_losses.shape[0], -1)), axis=-1) * (
            kwargs["ce_weight"] if hasattr(kwargs, "ce_weight") else 1)

        if error_kd:
            lambdas = vpsde.discrete_lambda
            next_lambdas = jnp.concatenate((lambdas, lambdas[-1][None, ...]))
            now_lambdas = jnp.concatenate((lambdas[0][None, ...], lambdas))
            last_lambdas = jnp.exp(-next_lambdas) - jnp.exp(-now_lambdas)
            final_weight = last_lambdas.shape[0] * last_lambdas[labels] / jnp.sum(last_lambdas)
        else:
            final_weight = jnp.ones_like(labels)

        kd_losses = batch_mul(jnp.square(score - teacher_score), final_weight)
        kd_losses = reduce_op(kd_losses.reshape((kd_losses.shape[0], -1)), axis=-1) * (
            kwargs["kd_weight"] if hasattr(kwargs, "kd_weight") else 1)

        loss = jnp.mean(origin_losses + kd_losses)
        return loss, new_model_state

    return loss_fn


def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False,
                teacher_score_model=None, config=None):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                                  continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(carry_state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        if train:
            params = state.optimizer.params
            states = state.model_state
            (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch)
            grad = jax.lax.pmean(grad, axis_name='batch')
            new_optimizer = optimize_fn(state, grad, new_model_state)
            new_params_ema = jax.tree_map(
                lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
                state.params_ema, new_optimizer.params
            )
            step = state.step + 1
            new_state = state.replace(
                step=step,
                optimizer=new_optimizer,
                model_state=new_model_state,
                params_ema=new_params_ema
            )
        else:
            loss, _ = loss_fn(step_rng, state.params_ema, state.model_state, batch)
            new_state = state

        loss = jax.lax.pmean(loss, axis_name='batch')
        new_carry_state = (rng, new_state)
        return new_carry_state, loss

    return step_fn


def get_eval_detail_step_fn(sde, model, train, reduce_mean=True, continuous=True):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """

    def loss_fn(rng, params, states, batch, t):
        """Compute the loss function.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """
        eps = 1e-5
        reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
        score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous,
                                       return_state=True)
        data = batch['image']
        mean, std = sde.marginal_prob(data, t)
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, data.shape)
        perturbed_data = mean + batch_mul(std, z)
        rng, step_rng = random.split(rng)
        score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)
        losses = jnp.square(batch_mul(score, std) + z)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = jnp.mean(losses)
        return loss, new_model_state

    def step_fn(carry_state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """
        (rng, state, t) = carry_state
        rng, step_rng = jax.random.split(rng)
        loss, _ = loss_fn(step_rng, state.params_ema, state.model_state, batch, t)
        new_state = state
        loss = jax.lax.pmean(loss, axis_name='batch')
        new_carry_state = (rng, new_state, t)
        return new_carry_state, loss

    return step_fn


def get_kd_step_fn(sde, model, teacher_model, train, optimize_fn=None, reduce_mean=False, continuous=True,
                   likelihood_weighting=False, config=None):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      teacher_model: A `flax.linen.Module` object that represents the teacher model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_kd_sde_loss_fn(sde, model, teacher_model, config.training.mode == "error_kd", train,
                                     reduce_mean=reduce_mean,
                                     continuous=True, likelihood_weighting=likelihood_weighting,
                                     diff_step=config.training.diff_step,
                                     kd_weight=config.training.kd_weight,
                                     ce_weight=config.training.ce_weight)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_kd_smld_loss_fn(sde, model, teacher_model, config.training.mode == "error_kd", train,
                                          reduce_mean=reduce_mean,
                                          diff_step=config.training.diff_step,
                                          kd_weight=config.training.kd_weight,
                                          ce_weight=config.training.ce_weight)
        elif isinstance(sde, VPSDE):
            loss_fn = get_kd_ddpm_loss_fn(sde, model, teacher_model, config.training.mode == "error_kd", train,
                                          reduce_mean=reduce_mean,
                                          diff_step=config.training.diff_step,
                                          kd_weight=config.training.kd_weight,
                                          ce_weight=config.training.ce_weight)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(carry_state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, teacher_params, state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        if train:
            params = state.optimizer.params
            states = state.model_state
            (loss, new_model_state), grad = grad_fn(step_rng, params, teacher_params, states, batch)
            grad = jax.lax.pmean(grad, axis_name='batch')
            new_optimizer = optimize_fn(state, grad, new_model_state)
            new_params_ema = jax.tree_map(
                lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
                state.params_ema, new_optimizer.params
            )
            step = state.step + 1
            new_state = state.replace(
                step=step,
                optimizer=new_optimizer,
                model_state=new_model_state,
                params_ema=new_params_ema
            )
        else:

            loss, _ = loss_fn(step_rng, state.params_ema, teacher_params, state.model_state, batch)
            new_state = state

        loss = jax.lax.pmean(loss, axis_name='batch')
        new_carry_state = (rng, teacher_params, new_state)
        return new_carry_state, loss

    return step_fn
