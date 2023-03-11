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

# Lint as: python3
"""Training NCSNv3 on CIFAR-10 with continuous sigmas."""
import copy

import ml_collections
from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.mode = "vanilla_kd"
  training.kd_weight = 1
  training.ce_weight = 1
  training.diff_step = 1000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True

  # eval
  eval = config.eval
  eval.begin_ckpt = 26
  eval.end_ckpt = 26
  eval.batch_size = 512
  eval.enable_sampling = True
  eval.num_samples = 10000
  eval.enable_loss = True
  eval.enable_bpd = False
  eval.bpd_dataset = 'test'

  #  inception_score: 9.617269e+00, FID: 5.967415e+00, KID: 1.595003e-03
  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  config.teacher_model = teacher_model = copy.deepcopy(config.model)
  # student model
  teacher_model.name = 'ncsnpp'
  teacher_model.scale_by_sigma = False
  teacher_model.ema_rate = 0.9999
  teacher_model.normalization = 'GroupNorm'
  teacher_model.nonlinearity = 'swish'
  teacher_model.nf = 64
  teacher_model.ch_mult = (1, 2, 2, 2)
  teacher_model.num_res_blocks = 2
  teacher_model.attn_resolutions = (16,)
  teacher_model.resamp_with_conv = True
  teacher_model.conditional = True
  teacher_model.fir = False
  teacher_model.fir_kernel = [1, 3, 3, 1]
  teacher_model.skip_rescale = True
  teacher_model.resblock_type = 'biggan'
  teacher_model.progressive = 'none'
  teacher_model.progressive_input = 'none'
  teacher_model.progressive_combine = 'sum'
  teacher_model.attention_type = 'ddpm'
  teacher_model.init_scale = 0.
  teacher_model.embedding_type = 'positional'
  teacher_model.fourier_scale = 16
  teacher_model.conv_size = 3

  return config

"""
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.6/ python main.py \
--config ./configs/vp/cifar10_ddpmpp_kd_continuous.py --eval_folder ./eval \
--mode train --workdir /home/Bigdata/mtt_distillation_ckpt/song_sde/cifar10_ddpmpp_kd_contiguous/
"""