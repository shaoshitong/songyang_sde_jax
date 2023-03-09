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

  config.student_model = student_model = ml_collections.ConfigDict()
  # student model
  student_model.name = 'ncsnpp'
  student_model.scale_by_sigma = False
  student_model.ema_rate = 0.9999
  student_model.normalization = 'GroupNorm'
  student_model.nonlinearity = 'swish'
  student_model.nf = 64
  student_model.ch_mult = (1, 2, 2, 2)
  student_model.num_res_blocks = 2
  student_model.attn_resolutions = (16,)
  student_model.resamp_with_conv = True
  student_model.conditional = True
  student_model.fir = False
  student_model.fir_kernel = [1, 3, 3, 1]
  student_model.skip_rescale = True
  student_model.resblock_type = 'biggan'
  student_model.progressive = 'none'
  student_model.progressive_input = 'none'
  student_model.progressive_combine = 'sum'
  student_model.attention_type = 'ddpm'
  student_model.init_scale = 0.
  student_model.embedding_type = 'positional'
  student_model.fourier_scale = 16
  student_model.conv_size = 3

  return config
