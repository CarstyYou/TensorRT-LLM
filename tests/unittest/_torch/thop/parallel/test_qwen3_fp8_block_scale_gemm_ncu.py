# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from _torch.helpers import calc_diff, per_block_cast_to_fp8_e8m0
from utils.util import getSMVersion

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils


QWEN3_TP2_SHAPES = {
    "qkv_proj": (5120, 5120),
    "o_proj": (5120, 4096),
    "gate_up_proj": (25600, 5120),
    "down_proj": (5120, 12800),
}

DEFAULT_M_CASES = [1, 2, 3, 4, 5, 6, 32, 1800, 3660]
DEFAULT_METRICS = ",".join([
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
])


def _parse_csv_env(name: str, default_values, cast_fn=str):
    raw = os.environ.get(name)
    if not raw:
        return list(default_values)
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast_fn(item))
    return values


def _prepare_fp8_inputs(m: int, k: int, n: int, dtype: torch.dtype):
    torch.random.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / k
    b = torch.randn((n, k), device="cuda", dtype=dtype) / k

    if getSMVersion() == 120:
        act_a_fp8, act_a_sf = fp8_utils.per_token_quant_and_transform(a)
        act_b_fp8, act_b_sf = per_block_cast_to_fp8_e8m0(b)
        act_b_sf = fp8_utils.transform_sf_into_required_layout(
            act_b_sf,
            mn=act_b_fp8.shape[0],
            k=act_b_fp8.shape[1],
            recipe=(1, 128, 128),
            is_sfa=False,
        )
    else:
        act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
        act_b_fp8, act_b_sf = per_block_cast_to_fp8_e8m0(b)

    return a, b, act_a_fp8, act_b_fp8, act_a_sf, act_b_sf


def _run_single_case(shape_name: str,
                     m: int,
                     dtype: torch.dtype = torch.bfloat16,
                     warmup_iters: int = 5,
                     measure_iters: int = 1,
                     check_correctness: bool = False):
    n, k = QWEN3_TP2_SHAPES[shape_name]
    a, b, act_a_fp8, act_b_fp8, act_a_sf, act_b_sf = _prepare_fp8_inputs(
        m=m, k=k, n=n, dtype=dtype)

    output_expected = None
    if check_correctness:
        output_expected = a @ b.t()

    for _ in range(warmup_iters):
        torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8, act_b_fp8,
                                                act_a_sf, act_b_sf)
    torch.cuda.synchronize()

    output = None
    for _ in range(measure_iters):
        output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8, act_b_fp8,
                                                         act_a_sf, act_b_sf)
    torch.cuda.synchronize()

    if check_correctness:
        diff = calc_diff(output, output_expected)
        assert diff < 1e-3
        torch.testing.assert_close(output,
                                   output_expected,
                                   atol=1e-3,
                                   rtol=1e-3)

    return {
        "shape_name": shape_name,
        "m": m,
        "n": n,
        "k": k,
        "dtype": str(dtype),
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
    }


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, choices=sorted(QWEN3_TP2_SHAPES))
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=1)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--metadata-out", default="")
    args = parser.parse_args()

    metadata = _run_single_case(shape_name=args.shape,
                                m=args.m,
                                warmup_iters=args.warmup_iters,
                                measure_iters=args.measure_iters,
                                check_correctness=args.check_correctness)

    if args.metadata_out:
        with open(args.metadata_out, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)


def _run_ncu_case(shape_name: str, m: int, outdir: Path):
    if shutil.which("ncu") is None:
        pytest.skip("ncu not found in PATH")

    outdir.mkdir(parents=True, exist_ok=True)
    case_name = f"{shape_name}_m{m}"
    rep_base = outdir / case_name
    metadata_out = outdir / f"{case_name}.json"

    metrics = os.environ.get("TRTLLM_QWEN3_NCU_METRICS", DEFAULT_METRICS)
    kernel_name = os.environ.get("TRTLLM_QWEN3_NCU_KERNEL_NAME",
                                 "regex:device_kernel")
    warmup_iters = int(os.environ.get("TRTLLM_QWEN3_NCU_WARMUP_ITERS", "5"))
    measure_iters = int(os.environ.get("TRTLLM_QWEN3_NCU_MEASURE_ITERS", "1"))
    check_correctness = os.environ.get("TRTLLM_QWEN3_NCU_CHECK_CORRECTNESS",
                                       "0") == "1"

    cmd = [
        "ncu",
        "--metrics",
        metrics,
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--kernel-name",
        kernel_name,
        "-f",
        "-o",
        str(rep_base),
        sys.executable,
        __file__,
        "--shape",
        shape_name,
        "--m",
        str(m),
        "--warmup-iters",
        str(warmup_iters),
        "--measure-iters",
        str(measure_iters),
        "--metadata-out",
        str(metadata_out),
    ]
    if check_correctness:
        cmd.append("--check-correctness")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, (
        f"NCU profile failed for shape={shape_name}, m={m}, "
        f"exit_code={result.returncode}")


@pytest.mark.skipif(
    getSMVersion() != 100 and getSMVersion() != 89 and getSMVersion() != 120,
    reason="The test is for Blackwell and Ada only. Current SM is %d." %
    getSMVersion(),
)
def test_qwen3_fp8_block_scale_gemm_ncu_harness():
    if os.environ.get("TRTLLM_RUN_NCU", "0") != "1":
        pytest.skip("Set TRTLLM_RUN_NCU=1 to enable manual NCU harness")

    shapes = _parse_csv_env("TRTLLM_QWEN3_NCU_SHAPES",
                            QWEN3_TP2_SHAPES.keys(), str)
    ms = _parse_csv_env("TRTLLM_QWEN3_NCU_MS", DEFAULT_M_CASES, int)

    outdir = Path(
        os.environ.get(
            "TRTLLM_QWEN3_NCU_OUTDIR",
            "/raid/xiy/TensorRT-LLM/.cursor/memory/exp34_cc6_mnt6144_decode_ab",
        ))

    for shape_name in shapes:
        assert shape_name in QWEN3_TP2_SHAPES, (
            f"Unknown Qwen3 TP=2 shape: {shape_name}")
        for m in ms:
            _run_ncu_case(shape_name=shape_name, m=m, outdir=outdir)


if __name__ == "__main__":
    _worker_main()
