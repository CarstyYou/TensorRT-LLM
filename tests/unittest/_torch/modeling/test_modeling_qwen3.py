# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3Model
from tensorrt_llm._torch.modules.embedding import Embedding, LMHead
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping


def _make_mapping() -> Mapping:
    return Mapping(world_size=2, tp_size=2, rank=0)


def _make_fake_linear_init(
    captured_strategies: list[AllReduceStrategy],
) -> Callable[..., None]:
    def _fake_linear_init(
        self: Linear,
        *args: object,
        **kwargs: object,
    ) -> None:
        mapping = kwargs.get("mapping")
        if not isinstance(mapping, Mapping):
            mapping = Mapping()

        nn.Module.__init__(self)
        strategy = kwargs.get("allreduce_strategy")
        if not isinstance(strategy, AllReduceStrategy):
            raise AssertionError("Expected Linear to receive allreduce_strategy")
        captured_strategies.append(strategy)
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_mode = kwargs.get("tensor_parallel_mode")
        self.gather_output = bool(kwargs.get("gather_output", False))
        self.padding_size = 0
        self.dtype = kwargs.get("dtype")
        self.in_features = int(args[0])
        self.out_features = int(args[1])

    return _fake_linear_init


class _DummyModule(nn.Module):

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        super().__init__()


@pytest.mark.parametrize("module_cls", [LMHead, Embedding],
                         ids=["lm_head", "embedding"])
def test_embedding_family_forwards_allreduce_strategy(
    module_cls: type[nn.Module],
) -> None:
    captured_strategies: list[AllReduceStrategy] = []

    with patch("tensorrt_llm._torch.modules.linear.Linear.__init__",
               new=_make_fake_linear_init(captured_strategies)):
        module_cls(
            num_embeddings=128,
            embedding_dim=64,
            dtype=torch.bfloat16,
            mapping=_make_mapping(),
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            allreduce_strategy=AllReduceStrategy.NCCL_SYMMETRIC,
        )

    assert captured_strategies == [AllReduceStrategy.NCCL_SYMMETRIC]


def test_qwen3_model_forwards_allreduce_strategy_to_embed_tokens() -> None:
    captured_strategies: list[AllReduceStrategy] = []
    pretrained_config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        torch_dtype=torch.bfloat16,
    )
    model_config = ModelConfig(pretrained_config=pretrained_config,
                               mapping=_make_mapping(),
                               allreduce_strategy=AllReduceStrategy.NCCL)

    with patch("tensorrt_llm._torch.modules.linear.Linear.__init__",
               new=_make_fake_linear_init(captured_strategies)), \
         patch("tensorrt_llm._torch.models.modeling_qwen3.Qwen3DecoderLayer",
               _DummyModule), \
         patch("tensorrt_llm._torch.models.modeling_qwen3.RMSNorm",
               _DummyModule):
        Qwen3Model(model_config)

    assert captured_strategies == [AllReduceStrategy.NCCL]
