from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch


SUPPORTED_POSITION_ENCODINGS = ("rope",)
SUPPORTED_MODES = ("nonthinking", "thinking")
ALL_MODEL_VARIANTS = tuple(
    f"{position}/{mode}"
    for position in SUPPORTED_POSITION_ENCODINGS
    for mode in SUPPORTED_MODES
)
REFERENCE_MODEL_VARIANTS = ("rope/nonthinking", "rope/thinking")
SUPPORTED_COUNT_TOKENIZATIONS = ("atomic", "digitwise")
SUPPORTED_TRACE_FORMATS = ("indexed", "separator")
VERSION_SPECS = {
    "v20": {"count_tokenization": "atomic", "trace_format": "indexed"},
    "v21": {"count_tokenization": "digitwise", "trace_format": "indexed"},
    # v22 is the matched v20 no-index control.  It keeps atomic final answers,
    # but every ordinal trace token is replaced one-for-one by a fixed <Sep>.
    "v22": {"count_tokenization": "atomic", "trace_format": "separator"},
    # v23 preserves the v22 grammar and reruns both modes with an 8x weight on
    # the final count target.  This isolates loss dilution from de-indexing.
    "v23": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "final_count_loss_weight": 8.0,
    },
    # v24 keeps the v22 objective and grammar but reduces the supported count
    # range to 1..10.  Both modes are retrained so all geometry comparisons are
    # made within the same data distribution.
    "v24": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        # Keep the expected target-set count within the accepted range.  The
        # v22 cap (0.12) admits triples with ~31 expected hits per 256-char
        # window, some of which have no valid <=10 window in a corpus split.
        "needle_pool_frequency_threshold": 10.0 / 256.0,
    },
    # v24.2 is the single-variable count-balance control for v24.  It keeps
    # the model, separator grammar, loss weights, count support, pool, seed,
    # and schedule fixed while drawing semantic counts uniformly from 1..10.
    "v24.2": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "uniform",
    },
    # v24.3 is the loss-only control for v24.2.  Sampling, data, model, seed,
    # and schedule stay fixed.  Only the task-output phase changes from a
    # token-weighted mean to separately normalized count/trace/structure
    # regions, preventing a longer Thinking trace from diluting count loss.
    "v24.3": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "uniform",
        "task_output_loss_reduction": "component_normalized",
    },
    # v24.4 keeps the v24.3 loss and changes only the training sampler.  A
    # maximum-entropy distribution over feasible (needle set, count) cells
    # matches both the uniform set marginal and the uniform count marginal.
    # Structural-zero cells remain at probability zero.
    "v24.4": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
    },
    # v24.5 keeps the v24.4 sampler, loss, count support, model, seed, and
    # schedule fixed, but reduces the needle pool from 100 sets to 20.  This
    # gives each semantic marker five times as much supervision and makes the
    # set x count support much closer to a complete Cartesian product.
    "v24.5": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_size": 20,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
    },
    # v24.6 is the readout-bridge control for v24.5.  The input embedding and
    # output projection start numerically identical, but their parameters are
    # no longer tied after initialization.  All data, objective, model width,
    # sampler, seed, and schedule settings remain fixed.
    "v24.6": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_size": 20,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "tie_word_embeddings": False,
    },
    # v24.7 keeps the complete v24.6 data/model/readout setting and adds one
    # training-only representation objective after the language-model phase.
    # At the <Ans> query, same-count examples are pulled together and
    # different-count examples are separated.  Inference is unchanged: the
    # ordinary LM head must still emit the raw atomic answer token.
    "v24.7": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_size": 20,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "tie_word_embeddings": False,
        "answer_query_contrastive_weight": 0.1,
        "answer_query_contrastive_temperature": 0.1,
    },
    # v25 is the retrieval-pressure setting.  It preserves the complete
    # v24.7 objective and separator trace, but its public wrapper supplies a
    # 1,024-character context (and the corresponding position budget and
    # frequency cap).  Count support stays at 1..10, so the experimental
    # pressure comes from searching a four-times-longer prompt rather than
    # from adding new answer classes or lengthening the trace.
    "v25": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "needle_pool_size": 20,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "tie_word_embeddings": False,
        "answer_query_contrastive_weight": 0.1,
        "answer_query_contrastive_temperature": 0.1,
    },
    # v26 returns to the 256-character v24.3 setting where Thinking already
    # has a held-out accuracy advantage.  Its sole training change is to
    # untie the native LM head at initialization, so input embeddings are no
    # longer moved by the highly asymmetric atomic-number output gradients.
    # No contrastive/probe objective is added: count compression remains an
    # emergent property to be measured after training.
    "v26": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "uniform",
        "task_output_loss_reduction": "component_normalized",
        "tie_word_embeddings": False,
    },
    # v28 is the minimal from-scratch readout-decoupling control for v24.3.
    # The full transformer and the original separator trace are trained
    # end-to-end with the unchanged v24.3 objective.  Only the ten atomic count
    # tokens use output vectors that are independent of their input embeddings;
    # every other vocabulary row remains tied.  This preserves trace copying
    # while removing the diagnosed count-input/count-output optimization
    # conflict, without a post-hoc calibration phase or auxiliary objective.
    "v28": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "uniform",
        "task_output_loss_reduction": "component_normalized",
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
    },
    # v29 is the single-scalar readout correction for v28.  The data, model,
    # partial count-only untying, separator trace, and schedule are unchanged.
    # Only the already component-normalized final-count region receives a 4x
    # coefficient during the task-output phase; the trace region keeps its
    # independent coefficient of 1 rather than being diluted token-wise.
    "v29": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "uniform",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 4.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
    },
    # v30 is the depth-only capacity control for v29.  It keeps the complete
    # data, trace, objective, width, and four-head setting fixed, and adds two
    # transformer blocks so the targeted-retrieval -> trace -> answer path has
    # more compositional depth without adding a new loss or inference rule.
    "v30": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "uniform",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 4.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 6,
    },
    # v31 returns to the independent four-layer v29 pair and changes one
    # scalar only: the already component-normalized final-count region receives
    # coefficient 8 instead of 4.  Data, trace, sampler, architecture, mode
    # separation, optimizer, and inference are unchanged.
    "v31": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "uniform",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
    },
    # v32 is the sampler-only control for v31.  The two modes remain separate
    # models and keep the same no-index trace, architecture, objective, seed,
    # and schedule.  Only the accepted training distribution changes to the
    # audited maximum-entropy distribution over feasible (set, count) cells,
    # sharply reducing the set-identity shortcut while retaining all 100 sets.
    "v32": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
    },
    # v33 keeps v32's low-shortcut max-entropy data, separator trace, partial
    # count readout, and two independently initialized model runs.  Its one
    # new optimization mechanism addresses the observed teacher-forced /
    # free-running trace gap: during the task-output phase, gold continuation
    # inputs are progressively replaced by the model's own previous-token
    # predictions, up to probability 0.5.  Targets and inference are unchanged.
    # The fixed 6,000-step budget is chosen from the v32 control curve, before
    # observing v33, so that the comparison measures sample-efficient learning
    # rather than the saturated 10,000-step Non-thinking endpoint.
    "v33": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "task_output_scheduled_sampling_max_probability": 0.5,
        "train_steps": 6_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v34 is the loss-only alternative to v33's failed token-level roll-in.
    # It keeps gold-prefix teacher forcing and raises only the already
    # component-normalized trace-region coefficient from 1 to 8, matching the
    # final-count coefficient.  The low-shortcut v32 data and fixed 6,000-step
    # efficiency budget remain unchanged; no input token is corrupted.
    "v34": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 6_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v35 corrects the component imbalance diagnosed in v34.  Count, trace,
    # and structure are all component-normalized, so assigning coefficient 8
    # to each gives them equal aggregate objective share.  This specifically
    # restores supervision for the continue-vs-close decision without changing
    # any trace token, prefix, architecture, sampler, or inference rule.
    "v35": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 6_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v36 is the schedule-only control for v35.  It still performs exactly
    # 6,000 optimizer updates, but evaluates the cosine schedule against the
    # original v32 10,000-step horizon.  Thus the learning rate is not forced
    # to zero at the screening endpoint.  Data, trace, loss, architecture,
    # initialization, teacher forcing, and the two independent model runs are
    # unchanged.
    "v36": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 6_000,
        "lr_decay_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v37 follows the v35 curve to a conservative minimum learning rate, then
    # adds a low-rate consolidation tail.  The first 6,000 updates use cosine
    # decay toward 1e-5; updates 6,001--8,000 remain at 1e-5.  This preserves
    # the stable v35 optimization regime while testing whether trace-length
    # decisions need additional refinement.  All task, trace, model, sampler,
    # objective, and independent-mode settings remain unchanged.
    "v37": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 8_000,
        "lr_decay_steps": 6_000,
        "min_lr": 1e-5,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
        ),
    },
    # v38 returns to the successful v35 objective and fixed 6,000-step
    # efficiency budget.  Its sole change is a mild exposure-bias control for
    # Thinking: during the task-output phase, the probability of replacing a
    # gold continuation input with the model's own previous-token prediction
    # rises linearly from 0 to 0.1.  Gold targets, the separator trace, data,
    # architecture, loss coefficients, inference, and the independent model
    # runs are unchanged.  Non-thinking has no eligible intermediate trace
    # positions, so this scalar is operationally inert for that baseline.
    "v38": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "task_output_scheduled_sampling_max_probability": 0.1,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 6_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v39 returns to v35's pure gold-prefix teacher forcing and changes only
    # the loss-schedule boundary.  Task-output-only supervision starts at the
    # first optimizer update instead of after 1,500 all-sequence language-
    # modeling updates.  Both independently trained modes receive the same
    # 6,000 task updates; the data, trace, targets, model, sampler, component
    # coefficients, learning-rate schedule, and inference rule are unchanged.
    "v39": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 6_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v40 returns to the successful v35 optimization path and changes only
    # the supported semantic count range from 1..10 to 1..5.  The 256-token
    # retrieval prompt, exact 100 marker sets, three-character target sets,
    # separator trace grammar, objective, model, optimizer, sampler family,
    # and independent mode-specific training remain fixed.  This cleanly tests
    # whether long no-index separator traces are the dominant behavioral
    # bottleneck while preserving broad-versus-targeted retrieval pressure.
    "v40": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 5,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 6_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v41 is the model-width screen on top of v40.  It leaves the four-layer
    # serial depth, 256-character prompt, count support, data, trace, loss,
    # optimizer, and independent mode-specific training fixed.  The residual
    # width grows 256 -> 384 while preserving a 64-dimensional attention head
    # (4 -> 6 heads) and the 4x MLP ratio (1024 -> 1536).  This adds parallel
    # retrieval capacity without giving Non-thinking extra serial depth.
    "v41": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 5,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 6,
        "n_embd": 384,
        "n_inner": 1536,
        "train_steps": 6_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v42 changes only the optimization horizon on top of v41.  The wider
    # model's behavioral transition occurred late in v41, so both independently
    # trained modes are restarted from random initialization and trained for
    # 8,000 updates with the same cosine rule (hence an 8,000-step horizon).
    # Data, trace, targets, architecture, loss coefficients, warmup, peak LR,
    # clipping, and inference remain unchanged.
    "v42": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 5,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 6,
        "n_embd": 384,
        "n_inner": 1536,
        "train_steps": 8_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
        ),
    },
    # v43 removes the only train/evaluation support mismatch discovered in the
    # v42 failure audit.  The maximum-entropy set x count cell probabilities
    # are unchanged, but sampling within each feasible cell now ranges over
    # every legal corpus window instead of a deterministic cap of 8,192
    # evenly-spaced starts.  Everything else is byte-for-byte/canonically v42.
    "v43": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 5,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 6,
        "n_embd": 384,
        "n_inner": 1536,
        "train_steps": 8_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
        ),
    },
    # v44 is the count-support restoration screen on top of v43.  It keeps
    # exact full within-cell sampling, the maximum-entropy set x count
    # distribution, separator/no-index trace, equal component-normalized
    # objective, wider four-layer model, optimizer, and 8,000-step endpoint
    # fixed.  The sole configured change is count support 1..5 -> 1..10.
    # This combines v35's scientifically useful long-count task with the
    # clean support and capacity controls established by v41--v43.
    "v44": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 6,
        "n_embd": 384,
        "n_inner": 1536,
        "train_steps": 8_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
        ),
    },
    # v45 reallocates a nearly fixed parameter budget from parallel width to
    # serial depth on top of v44.  Head dimension stays exactly 64 while the
    # transformer changes from 4L/6H/384D to 6L/5H/320D (with a 4x MLP).
    # Data, count support, full-support sampler, no-index trace, objective,
    # optimizer, independent training, and the fixed 8,000-step endpoint are
    # unchanged.  This is a targeted architecture test of whether serial trace
    # retrieval benefits without making direct broad aggregation still wider.
    "v45": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 6,
        "n_head": 5,
        "n_embd": 320,
        "n_inner": 1280,
        "train_steps": 8_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
        ),
    },
    # v46 returns to v35's deliberately small 4L/4H/256D capacity and keeps
    # the corrected full-support set x count sampler introduced in v43.  Its
    # task-only intervention is a fresh random permutation of each selected
    # 256-character counting window.  The character multiset, target count,
    # query, no-index separator trace, answer, optimizer, and independent
    # mode-specific training are unchanged.  This destroys corpus-window and
    # local n-gram shortcuts while preserving the broad-versus-serial retrieval
    # problem exactly at the token-multiset level.
    "v46": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 6_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
        ),
    },
    # v47 is the fixed-horizon convergence control for v46.  Both modes are
    # independently reinitialized and receive the same 10,000 optimizer
    # updates.  The cosine schedule therefore also spans 10,000 updates.
    # Counts, model capacity, full-support sampler, freshly permuted task
    # windows, no-index separator trace, objective, and inference are exactly
    # unchanged.  This tests the predeclared diagnosis that v46's long-count
    # serial trace was still improving when its 6,000-step learning rate
    # reached zero, without selecting or continuing an earlier checkpoint.
    "v47": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v48 is the parallel-capacity control for v47.  It preserves the fixed
    # 10,000-step horizon and every task, data, trace, sampler, objective, and
    # independent-training setting.  The only substantive change is the same
    # width/head expansion used by v41--v44: 4H/256D/1024-MLP becomes
    # 6H/384D/1536-MLP at four layers, retaining 64 dimensions per head.  This
    # tests whether v47's errors on the longest no-index marker sequences are a
    # parallel targeted-retrieval capacity bottleneck under shortcut-controlled
    # task windows.
    "v48": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 6,
        "n_embd": 384,
        "n_inner": 1536,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v49 returns to v47's compact 4L/4H/256D model and changes only the
    # semantic partition used by the component-normalized Thinking loss.
    # Separator tokens are grammar decisions (continue) rather than retrieved
    # marker identities, so v49 groups every <Sep> with </Think> and the other
    # structural tokens.  This balances continue against stop while reserving
    # the trace component for marker identities.  Serialized traces, targets,
    # data, optimizer, inference, and independent model training are unchanged.
    "v49": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "task_output_trace_delimiters_as_structure": True,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "n_inner": 1024,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v50 is the fixed-horizon control for v49.  The model, data, trace,
    # optimizer family, and grammar-balanced loss are unchanged; only the
    # cosine endpoint is extended so that the still-decreasing marker and
    # grammar objectives are not forced to zero learning rate at step 10k.
    "v50": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 8.0,
        "task_output_trace_delimiters_as_structure": True,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "n_inner": 1024,
        "train_steps": 20_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
            12_000,
            14_000,
            16_000,
            18_000,
            20_000,
        ),
    },
    # v51 returns to v49's fixed 10k schedule and changes one loss coefficient:
    # grammar/structure has weight 16 instead of 8.  This targets v49's nearly
    # symmetric +/-1 trace-length errors without changing tokens, marker loss,
    # data, architecture, or inference.
    "v51": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 16.0,
        "task_output_trace_delimiters_as_structure": True,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "n_inner": 1024,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v52 keeps v51's grammar emphasis and changes one loss coefficient:
    # marker-identity trace has weight 16 instead of 8.  This targets the
    # observed v51 gap between correct trace length and exact marker sequence
    # without changing tokens, data, architecture, schedule, or inference.
    "v52": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 16.0,
        "task_output_structure_weight": 16.0,
        "task_output_trace_delimiters_as_structure": True,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "n_inner": 1024,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v53 returns to v51's best 8/8/16 objective and changes one architecture
    # scalar: serial depth increases from four to six layers.  Width, head
    # bank, data, no-index trace, schedule, and inference remain unchanged.
    "v53": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 16.0,
        "task_output_trace_delimiters_as_structure": True,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 6,
        "n_head": 4,
        "n_embd": 256,
        "n_inner": 1024,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v54 returns to v51's best 4L/4H/256D architecture and changes one
    # training scalar: during the Thinking task-output phase, scheduled
    # sampling rises linearly from 0 to 0.1.  Gold targets, the serialized
    # no-index trace, data, Non-thinking path, endpoint, and inference remain
    # unchanged.  This is a direct teacher-forcing/free-running gap control.
    "v54": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 16.0,
        "task_output_trace_delimiters_as_structure": True,
        "task_output_scheduled_sampling_max_probability": 0.1,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "n_inner": 1024,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v55 returns to v51's pure gold-prefix training and changes one objective
    # coefficient: grammar/structure weight increases from 16 to 32.  This
    # directly targets the remaining continue-versus-close errors after v51,
    # while preserving marker-identity weight, count weight, data, no-index
    # trace, architecture, endpoint, and inference.
    "v55": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 32.0,
        "task_output_trace_delimiters_as_structure": True,
        "task_output_scheduled_sampling_max_probability": 0.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "n_inner": 1024,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v56 returns exactly to v51's best objective and changes one optimization
    # scalar: batch size increases from 128 to 256.  The larger independently
    # sampled batch reduces per-update set/count and long-trace gradient
    # variance without changing the sampler distribution, number of optimizer
    # updates, trace, architecture, learning-rate schedule, or inference.
    "v56": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 16.0,
        "task_output_trace_delimiters_as_structure": True,
        "task_output_scheduled_sampling_max_probability": 0.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "batch_size": 256,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "n_inner": 1024,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v57 returns to v51's batch-128 optimization path and changes one linked
    # architecture factor: the parallel head bank widens from 4H/256D to
    # 6H/384D while preserving 64 dimensions per head and the 4x MLP ratio.
    # Data, no-index trace, 8/8/16 objective, depth, schedule, and inference
    # remain unchanged.
    "v57": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 16.0,
        "task_output_trace_delimiters_as_structure": True,
        "task_output_scheduled_sampling_max_probability": 0.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "batch_size": 128,
        "n_layer": 4,
        "n_head": 6,
        "n_embd": 384,
        "n_inner": 1536,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
    # v58 continues the clean v51 -> v57 parallel-capacity sweep.  It widens
    # the head bank from 6H/384D to 8H/512D while preserving 64 dimensions per
    # head and the 4x MLP ratio.  Data, no-index trace, 8/8/16 objective,
    # depth, batch size, schedule, and inference remain unchanged.
    "v58": {
        "count_tokenization": "atomic",
        "trace_format": "separator",
        "count_max_threshold": 10,
        "needle_pool_frequency_threshold": 10.0 / 256.0,
        "training_count_distribution": "maxent_set_count",
        "joint_sampler_max_starts_per_cell": None,
        "permute_task_context_tokens": True,
        "task_output_loss_reduction": "component_normalized",
        "task_output_count_weight": 8.0,
        "task_output_trace_weight": 8.0,
        "task_output_structure_weight": 16.0,
        "task_output_trace_delimiters_as_structure": True,
        "task_output_scheduled_sampling_max_probability": 0.0,
        "tie_word_embeddings": True,
        "untie_atomic_count_readout": True,
        "batch_size": 128,
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 512,
        "n_inner": 2048,
        "train_steps": 10_000,
        "phase_cloud_steps": (
            0,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
        ),
    },
}
SUPPORTED_VERSIONS = tuple(VERSION_SPECS)
SUPPORTED_TRAINING_COUNT_DISTRIBUTIONS = (
    "natural",
    "uniform",
    "maxent_set_count",
)
SUPPORTED_TASK_OUTPUT_LOSS_REDUCTIONS = ("token_weighted_mean", "component_normalized")


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class V20Config:
    """Shared v20/v21/v22/v23/v24/v24.2/v24.3/v24.4/v24.5 configuration.

    v20 and v21 are deliberately paired.  The only task-grammar difference is
    ``count_tokenization``: v20 uses one atomic token per integer, whereas v21
    renders every trace index and final answer with shared decimal digit tokens.
    v22 is a second matched control: it keeps v20's atomic final answer while
    replacing every explicit trace index with the same separator token.
    v23 keeps that separator trace and trains both modes with an 8x final-count
    loss weight. v24 returns to unit loss weights and retrains both modes on the
    smaller count range 1..10. v24.2 changes only v24's training count
    distribution from natural to uniform. v24.3 changes only v24.2's
    post-boundary task-output loss reduction. v24.4 changes only v24.3's
    training sampler to balance both set and count marginals. v24.5 changes
    only the v24.4 needle-pool size from 100 to 20.
    """

    version: str = "v20"
    preset: str = "debug"
    seed: int = 1234
    seq_len: int = 256
    needle_set_size: int = 3
    needle_pool_size: int = 100
    needle_pool_frequency_threshold: float = 0.12
    needle_pool_frequency_bins: int = 20
    needle_pool_seed: int | None = None
    count_max_threshold: int = 30
    task_occurrence_ratio: float = 1.0
    training_count_distribution: str = "natural"
    corpus_train_fraction: float = 0.80
    corpus_validation_fraction: float = 0.10
    candidate_filter_max_attempts: int = 100_000
    # Maximum number of legal corpus starts retained inside each feasible
    # set x count cell of the maximum-entropy sampler.  None uses the full
    # within-cell support.  The historical 8,192 default exactly preserves
    # versions through v42 and old serialized configs.
    joint_sampler_max_starts_per_cell: int | None = 8_192
    shuffle_needle_set_order: bool = True
    # If true, each accepted counting-task source window is independently
    # permuted before rendering.  Counts and character multiplicities are
    # invariant, but contiguous-corpus and absolute-start shortcuts disappear.
    # Raw language-model examples retain their natural order.
    permute_task_context_tokens: bool = False
    # v20/v21 are intentionally RoPE-only so the comparison isolates output
    # tokenization rather than position encoding.
    position_encodings: tuple[str, ...] = ("rope",)
    enabled_model_variants: tuple[str, ...] = REFERENCE_MODEL_VARIANTS

    train_steps: int = 10_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    warmup_steps: int = 500
    # Optional horizon for cosine decay, independent of the number of
    # optimizer updates actually executed.  None preserves the historical
    # behavior where the schedule reaches zero at ``train_steps``.
    lr_decay_steps: int | None = None
    # Minimum cosine learning rate, also used as the constant tail after an
    # explicit decay horizon.  Zero exactly preserves all historical runs.
    min_lr: float = 0.0
    grad_clip: float = 1.0
    precision: str = "bf16"
    log_every: int = 50
    eval_every: int = 500
    ar_eval_every: int = 1_000
    # Dense model-only snapshots are the scientific checkpoints used for phase
    # transition analysis.  Full optimizer/RNG recovery checkpoints are less
    # frequent and are overwritten except for the objective boundary and final.
    checkpoint_every: int = 100
    recovery_every: int = 500
    snapshot_shard_every: int = 500
    snapshot_dtype: str = "float16"
    eval_examples_per_count: int = 10
    final_examples_per_count: int = 50
    ar_examples_per_count: int = 2
    permutation_examples_per_count: int = 1
    max_steps_for_language_pred: int = 1_500
    final_count_loss_weight: float = 1.0
    cot_trace_loss_weight: float = 1.0
    # These fields affect only steps after max_steps_for_language_pred.  The
    # legacy reduction remains the default so every earlier version is exactly
    # reproducible.  Component-normalized loss first averages each semantic
    # region within an example, then across the batch, and finally combines the
    # three region means with the coefficients below.
    task_output_loss_reduction: str = "token_weighted_mean"
    task_output_count_weight: float = 1.0
    task_output_trace_weight: float = 1.0
    task_output_structure_weight: float = 0.1
    # Historical component-normalized runs assign repeated <Sep> query tokens
    # to the trace region.  Enabling this flag keeps those grammar decisions
    # in the structure region with </Think>, without changing serialized data.
    task_output_trace_delimiters_as_structure: bool = False
    # Optional training-only supervised contrastive objective on the final
    # answer-query residual.  Zero exactly reproduces all versions through
    # v24.6.  It never changes inference or supplies a decoder.
    answer_query_contrastive_weight: float = 0.0
    answer_query_contrastive_temperature: float = 0.1
    # Exposure-bias control used only after the language-model phase.  At a
    # linearly increasing probability, a generated continuation input is
    # replaced by the model's own previous-token prediction while its target
    # remains the original gold token.  A zero maximum exactly reproduces all
    # versions through v32.
    task_output_scheduled_sampling_max_probability: float = 0.0

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_inner: int = 1024
    n_positions: int = 384
    max_relative_distance: int = 256
    rpe_max_update: bool = False
    rope_base: float = 10_000.0

    attention_examples_per_count: int = 10
    state_train_examples_per_count: int = 20
    state_eval_examples_per_count: int = 10
    phase_examples_per_count: int = 1
    phase_head_selection_examples_per_count: int = 2
    phase_cloud_steps: tuple[int, ...] = (
        0, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 5_000, 7_000, 10_000
    )
    analysis_batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model/data interface metadata. These are deliberately immutable in v20.
    noise_source: str = "shakespeare_char"
    task_type: str = "target_character_set"
    loss_scope: str = "all_sequence"
    query_layout: str = "query_first"
    count_tokenization: str = "atomic"
    trace_format: str = "indexed"
    use_sdpa: bool = True
    tie_word_embeddings: bool = True
    # If true, only the atomic count-token output vectors are independent of
    # the input embedding.  The complete model is still trained end-to-end;
    # this is an architectural parameterization, not a frozen readout tail.
    untie_atomic_count_readout: bool = False

    @property
    def count_min(self) -> int:
        return 1

    @property
    def count_max(self) -> int:
        """Compatibility alias; count_max_threshold is the only stored setting."""

        return int(self.count_max_threshold)

    @property
    def effective_needle_pool_seed(self) -> int:
        return int(self.seed + 20_000 if self.needle_pool_seed is None else self.needle_pool_seed)

    @property
    def corpus_test_fraction(self) -> float:
        return 1.0 - float(self.corpus_train_fraction) - float(self.corpus_validation_fraction)

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(mode for _, mode in self.model_variants))

    @property
    def model_variants(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (value.split("/", 1)[0], value.split("/", 1)[1])
            for value in self.enabled_model_variants
        )

    @property
    def max_render_len(self) -> int:
        # BOS + query + prompt + Think + (trace query + marker)*n +
        # close + Ans + final-count digits + EOS.  Query-first is fixed in both
        # families, and digitwise v21 remains within the 384-token context.
        def width(value: int) -> int:
            return 1 if self.count_tokenization == "atomic" else len(str(int(value)))
        prefix = 1 + (2 + self.needle_set_size) + self.seq_len
        direct = prefix + 1 + width(self.count_max_threshold) + 1
        trace = (
            2 * self.count_max_threshold
            if self.trace_format == "separator"
            else sum(width(index) + 1 for index in range(1, self.count_max_threshold + 1))
        )
        thinking = prefix + 1 + trace + 1 + 1 + width(self.count_max_threshold) + 1
        return max(direct, thinking)

    @property
    def count_bins(self) -> tuple[tuple[int, int], ...]:
        return ((1, self.count_max_threshold),)

    def count_bin(self, count: int) -> str:
        value = int(count)
        if not 1 <= value <= self.count_max_threshold:
            raise ValueError(f"count {value} is outside 1..{self.count_max_threshold}")
        return f"1-{self.count_max_threshold}"

    def validate(self) -> None:
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"V20Config.version must be one of {SUPPORTED_VERSIONS}")
        version_spec = VERSION_SPECS[self.version]
        expected_tokenization = version_spec["count_tokenization"]
        if self.count_tokenization != expected_tokenization:
            raise ValueError(
                f"{self.version} requires count_tokenization={expected_tokenization!r}"
            )
        if self.trace_format not in SUPPORTED_TRACE_FORMATS:
            raise ValueError(f"trace_format must be one of {SUPPORTED_TRACE_FORMATS}")
        expected_trace_format = version_spec["trace_format"]
        if self.trace_format != expected_trace_format:
            raise ValueError(
                f"{self.version} requires trace_format={expected_trace_format!r}"
            )
        if self.query_layout != "query_first":
            raise ValueError(
                "v20/v21/v22/v23/v24/v24.2/v24.3/v24.4 require query-first sequence construction"
            )
        if self.needle_set_size != 3:
            raise ValueError(
                "v20/v21/v22/v23/v24/v24.2/v24.3/v24.4 require exactly three distinct "
                "characters per needle set"
            )
        if self.needle_pool_size <= 0 or self.needle_pool_frequency_bins <= 0:
            raise ValueError("needle pool size and number of bins must be positive")
        if not 0.0 < self.needle_pool_frequency_threshold <= 1.0:
            raise ValueError("needle_pool_frequency_threshold must be in (0, 1]")
        if not 1 <= self.count_max_threshold <= self.seq_len:
            raise ValueError("count_max_threshold must satisfy 1 <= threshold <= seq_len")
        if not 0.0 <= self.task_occurrence_ratio <= 1.0:
            raise ValueError("task_occurrence_ratio must be in [0, 1]")
        if self.training_count_distribution not in SUPPORTED_TRAINING_COUNT_DISTRIBUTIONS:
            raise ValueError(
                "training_count_distribution must be one of "
                f"{SUPPORTED_TRAINING_COUNT_DISTRIBUTIONS}"
            )
        if self.training_count_distribution != "natural" and self.task_occurrence_ratio != 1.0:
            raise ValueError(
                "controlled count distributions require task_occurrence_ratio=1 so the "
                "requested example distribution is unambiguous"
            )
        if self.task_output_loss_reduction not in SUPPORTED_TASK_OUTPUT_LOSS_REDUCTIONS:
            raise ValueError(
                "task_output_loss_reduction must be one of "
                f"{SUPPORTED_TASK_OUTPUT_LOSS_REDUCTIONS}"
            )
        if self.corpus_train_fraction <= 0 or self.corpus_validation_fraction <= 0:
            raise ValueError("corpus train and validation fractions must be positive")
        if self.corpus_train_fraction + self.corpus_validation_fraction >= 1:
            raise ValueError("train + validation fractions must be less than one")
        if self.candidate_filter_max_attempts <= 0:
            raise ValueError("candidate_filter_max_attempts must be positive")
        if self.joint_sampler_max_starts_per_cell is not None and (
            type(self.joint_sampler_max_starts_per_cell) is not int
            or self.joint_sampler_max_starts_per_cell <= 0
        ):
            raise ValueError(
                "joint_sampler_max_starts_per_cell must be a positive integer or None"
            )
        if type(self.permute_task_context_tokens) is not bool:
            raise ValueError("permute_task_context_tokens must be a boolean")
        if type(self.task_output_trace_delimiters_as_structure) is not bool:
            raise ValueError(
                "task_output_trace_delimiters_as_structure must be a boolean"
            )
        if self.task_output_trace_delimiters_as_structure and (
            self.trace_format != "separator"
            or self.task_output_loss_reduction != "component_normalized"
        ):
            raise ValueError(
                "task_output_trace_delimiters_as_structure requires a separator "
                "trace and component-normalized task-output loss"
            )
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least two")
        canonical_n_layer = int(version_spec.get("n_layer", 4))
        canonical_n_head = int(version_spec.get("n_head", 4))
        canonical_n_embd = int(version_spec.get("n_embd", 256))
        canonical_n_inner = int(version_spec.get("n_inner", 1024))
        if (self.n_layer, self.n_head, self.n_embd, self.n_inner) != (
            canonical_n_layer,
            canonical_n_head,
            canonical_n_embd,
            canonical_n_inner,
        ):
            raise ValueError(
                f"{self.version} requires {canonical_n_layer} layers, "
                f"{canonical_n_head} heads, d_model={canonical_n_embd}, "
                f"MLP={canonical_n_inner}"
            )
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.max_render_len > self.n_positions:
            raise ValueError(
                f"max rendered length {self.max_render_len} exceeds n_positions={self.n_positions}"
            )
        if type(self.rpe_max_update) is not bool:
            raise ValueError("rpe_max_update must be a boolean")
        if self.rpe_max_update:
            raise ValueError("rpe_max_update is inapplicable because v20/v21 are RoPE-only")
        if type(self.max_relative_distance) is not int or self.max_relative_distance <= 0:
            raise ValueError("max_relative_distance must be a positive integer")
        if self.rpe_max_update and self.max_relative_distance != self.max_render_len - 1:
            raise ValueError(
                "rpe_max_update requires max_relative_distance == max_render_len - 1"
            )
        if not self.position_encodings:
            raise ValueError("at least one position encoding is required")
        invalid = sorted(set(self.position_encodings) - set(SUPPORTED_POSITION_ENCODINGS))
        if invalid:
            raise ValueError(f"unsupported position encodings: {invalid}")
        if not self.enabled_model_variants:
            raise ValueError("enabled_model_variants must contain at least one model")
        if len(set(self.enabled_model_variants)) != len(self.enabled_model_variants):
            raise ValueError("enabled_model_variants must not contain duplicates")
        invalid_variants = sorted(set(self.enabled_model_variants) - set(ALL_MODEL_VARIANTS))
        if invalid_variants:
            raise ValueError(f"unsupported model variants: {invalid_variants}")
        active_positions = tuple(
            position
            for position in SUPPORTED_POSITION_ENCODINGS
            if any(value.startswith(f"{position}/") for value in self.enabled_model_variants)
        )
        if self.position_encodings != active_positions:
            raise ValueError(
                "position_encodings must equal the position encodings used by "
                "enabled_model_variants"
            )
        if self.noise_source != "shakespeare_char" or self.task_type != "target_character_set":
            raise ValueError(
                "v20/v21/v22/v23/v24/v24.2/v24.3/v24.4 require the Shakespeare "
                "target-character-set task"
            )
        if self.loss_scope != "all_sequence":
            raise ValueError(
                "v20/v21/v22/v23/v24/v24.2/v24.3/v24.4 require all-sequence next-token loss metadata"
            )
        if self.precision not in {"float32", "bf16"}:
            raise ValueError("precision must be float32 or bf16")
        if self.snapshot_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("snapshot_dtype must be float16, bfloat16, or float32")
        if type(self.use_sdpa) is not bool:
            raise ValueError("use_sdpa must be a boolean")
        if type(self.tie_word_embeddings) is not bool:
            raise ValueError("tie_word_embeddings must be a boolean")
        canonical_tying = version_spec.get("tie_word_embeddings")
        if canonical_tying is not None and self.tie_word_embeddings is not canonical_tying:
            raise ValueError(
                f"{self.version} requires tie_word_embeddings={canonical_tying}"
            )
        if type(self.untie_atomic_count_readout) is not bool:
            raise ValueError("untie_atomic_count_readout must be a boolean")
        canonical_partial_untie = version_spec.get("untie_atomic_count_readout")
        if (
            canonical_partial_untie is not None
            and self.untie_atomic_count_readout is not canonical_partial_untie
        ):
            raise ValueError(
                f"{self.version} requires untie_atomic_count_readout="
                f"{canonical_partial_untie}"
            )
        if self.untie_atomic_count_readout and not self.tie_word_embeddings:
            raise ValueError(
                "untie_atomic_count_readout requires the remaining vocabulary rows "
                "to stay tied"
            )
        if self.untie_atomic_count_readout and self.count_tokenization != "atomic":
            raise ValueError(
                "untie_atomic_count_readout requires atomic count tokenization"
            )
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        for name in (
            "final_count_loss_weight",
            "cot_trace_loss_weight",
            "task_output_count_weight",
            "task_output_trace_weight",
            "task_output_structure_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if (
            not math.isfinite(float(self.answer_query_contrastive_weight))
            or self.answer_query_contrastive_weight < 0
        ):
            raise ValueError(
                "answer_query_contrastive_weight must be finite and nonnegative"
            )
        if (
            not math.isfinite(float(self.answer_query_contrastive_temperature))
            or self.answer_query_contrastive_temperature <= 0
        ):
            raise ValueError(
                "answer_query_contrastive_temperature must be finite and positive"
            )
        if (
            not math.isfinite(
                float(self.task_output_scheduled_sampling_max_probability)
            )
            or not 0.0
            <= float(self.task_output_scheduled_sampling_max_probability)
            <= 1.0
        ):
            raise ValueError(
                "task_output_scheduled_sampling_max_probability must be finite "
                "and in [0, 1]"
            )
        canonical_final_weight = version_spec.get("final_count_loss_weight")
        if (
            canonical_final_weight is not None
            and float(self.final_count_loss_weight) != float(canonical_final_weight)
        ):
            raise ValueError(
                f"{self.version} requires final_count_loss_weight="
                f"{canonical_final_weight:g}"
            )
        canonical_task_output_count_weight = version_spec.get(
            "task_output_count_weight"
        )
        if (
            canonical_task_output_count_weight is not None
            and float(self.task_output_count_weight)
            != float(canonical_task_output_count_weight)
        ):
            raise ValueError(
                f"{self.version} requires task_output_count_weight="
                f"{canonical_task_output_count_weight:g}"
            )
        canonical_task_output_trace_weight = version_spec.get(
            "task_output_trace_weight"
        )
        if (
            canonical_task_output_trace_weight is not None
            and float(self.task_output_trace_weight)
            != float(canonical_task_output_trace_weight)
        ):
            raise ValueError(
                f"{self.version} requires task_output_trace_weight="
                f"{canonical_task_output_trace_weight:g}"
            )
        canonical_task_output_structure_weight = version_spec.get(
            "task_output_structure_weight"
        )
        if (
            canonical_task_output_structure_weight is not None
            and float(self.task_output_structure_weight)
            != float(canonical_task_output_structure_weight)
        ):
            raise ValueError(
                f"{self.version} requires task_output_structure_weight="
                f"{canonical_task_output_structure_weight:g}"
            )
        canonical_count_max = version_spec.get("count_max_threshold")
        if (
            canonical_count_max is not None
            and int(self.count_max_threshold) != int(canonical_count_max)
        ):
            raise ValueError(
                f"{self.version} requires count_max_threshold={canonical_count_max}"
            )
        canonical_pool_threshold = version_spec.get("needle_pool_frequency_threshold")
        if (
            canonical_pool_threshold is not None
            and float(self.needle_pool_frequency_threshold)
            != float(canonical_pool_threshold)
        ):
            raise ValueError(
                f"{self.version} requires needle_pool_frequency_threshold="
                f"{canonical_pool_threshold:g}"
            )
        canonical_pool_size = version_spec.get("needle_pool_size")
        if (
            canonical_pool_size is not None
            and int(self.needle_pool_size) != int(canonical_pool_size)
        ):
            raise ValueError(
                f"{self.version} requires needle_pool_size={canonical_pool_size}"
            )
        canonical_count_distribution = version_spec.get("training_count_distribution")
        if (
            canonical_count_distribution is not None
            and self.training_count_distribution != canonical_count_distribution
        ):
            raise ValueError(
                f"{self.version} requires training_count_distribution="
                f"{canonical_count_distribution!r}"
            )
        if "joint_sampler_max_starts_per_cell" in version_spec:
            canonical_joint_sampler_cap = version_spec[
                "joint_sampler_max_starts_per_cell"
            ]
            if self.joint_sampler_max_starts_per_cell != canonical_joint_sampler_cap:
                raise ValueError(
                    f"{self.version} requires joint_sampler_max_starts_per_cell="
                    f"{canonical_joint_sampler_cap!r}"
                )
        canonical_context_permutation = version_spec.get(
            "permute_task_context_tokens"
        )
        if (
            canonical_context_permutation is not None
            and self.permute_task_context_tokens is not canonical_context_permutation
        ):
            raise ValueError(
                f"{self.version} requires permute_task_context_tokens="
                f"{canonical_context_permutation}"
            )
        canonical_task_output_reduction = version_spec.get("task_output_loss_reduction")
        if (
            canonical_task_output_reduction is not None
            and self.task_output_loss_reduction != canonical_task_output_reduction
        ):
            raise ValueError(
                f"{self.version} requires task_output_loss_reduction="
                f"{canonical_task_output_reduction!r}"
            )
        canonical_delimiter_partition = version_spec.get(
            "task_output_trace_delimiters_as_structure"
        )
        if (
            canonical_delimiter_partition is not None
            and self.task_output_trace_delimiters_as_structure
            is not canonical_delimiter_partition
        ):
            raise ValueError(
                f"{self.version} requires "
                "task_output_trace_delimiters_as_structure="
                f"{canonical_delimiter_partition}"
            )
        canonical_contrastive_weight = version_spec.get(
            "answer_query_contrastive_weight"
        )
        if (
            canonical_contrastive_weight is not None
            and float(self.answer_query_contrastive_weight)
            != float(canonical_contrastive_weight)
        ):
            raise ValueError(
                f"{self.version} requires answer_query_contrastive_weight="
                f"{canonical_contrastive_weight:g}"
            )
        canonical_contrastive_temperature = version_spec.get(
            "answer_query_contrastive_temperature"
        )
        if (
            canonical_contrastive_temperature is not None
            and float(self.answer_query_contrastive_temperature)
            != float(canonical_contrastive_temperature)
        ):
            raise ValueError(
                f"{self.version} requires answer_query_contrastive_temperature="
                f"{canonical_contrastive_temperature:g}"
            )
        canonical_scheduled_sampling = version_spec.get(
            "task_output_scheduled_sampling_max_probability"
        )
        if (
            canonical_scheduled_sampling is not None
            and float(self.task_output_scheduled_sampling_max_probability)
            != float(canonical_scheduled_sampling)
        ):
            raise ValueError(
                f"{self.version} requires "
                "task_output_scheduled_sampling_max_probability="
                f"{canonical_scheduled_sampling:g}"
            )
        canonical_train_steps = version_spec.get("train_steps")
        if (
            canonical_train_steps is not None
            and int(self.train_steps) != int(canonical_train_steps)
        ):
            raise ValueError(
                f"{self.version} requires train_steps={canonical_train_steps}"
            )
        canonical_batch_size = version_spec.get("batch_size")
        if (
            canonical_batch_size is not None
            and int(self.batch_size) != int(canonical_batch_size)
        ):
            raise ValueError(
                f"{self.version} requires batch_size={canonical_batch_size}"
            )
        if not math.isfinite(float(self.min_lr)) or not 0 <= self.min_lr < self.lr:
            raise ValueError("min_lr must be finite and satisfy 0 <= min_lr < lr")
        if self.lr_decay_steps is not None:
            if type(self.lr_decay_steps) is not int:
                raise ValueError("lr_decay_steps must be an integer or None")
            if self.lr_decay_steps <= self.warmup_steps:
                raise ValueError("lr_decay_steps must be greater than warmup_steps")
            if self.lr_decay_steps < self.train_steps and self.min_lr == 0:
                raise ValueError(
                    "lr_decay_steps shorter than train_steps requires positive min_lr"
                )
        canonical_lr_decay_steps = version_spec.get("lr_decay_steps")
        if (
            canonical_lr_decay_steps is not None
            and self.lr_decay_steps != int(canonical_lr_decay_steps)
        ):
            raise ValueError(
                f"{self.version} requires lr_decay_steps={canonical_lr_decay_steps}"
            )
        canonical_min_lr = version_spec.get("min_lr")
        if canonical_min_lr is not None and float(self.min_lr) != float(canonical_min_lr):
            raise ValueError(f"{self.version} requires min_lr={canonical_min_lr:g}")
        if type(self.max_steps_for_language_pred) is not int or self.max_steps_for_language_pred < 0:
            raise ValueError("max_steps_for_language_pred must be a nonnegative integer")
        if self.max_steps_for_language_pred < self.train_steps and self.task_occurrence_ratio == 0:
            raise ValueError(
                "task_occurrence_ratio must be positive when task-output-only training is scheduled"
            )
        if type(self.checkpoint_every) is not int or self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be a positive integer")
        if type(self.recovery_every) is not int or self.recovery_every <= 0:
            raise ValueError("recovery_every must be a positive integer")
        if type(self.snapshot_shard_every) is not int or self.snapshot_shard_every <= 0:
            raise ValueError("snapshot_shard_every must be a positive integer")
        if self.snapshot_shard_every % self.checkpoint_every:
            raise ValueError("snapshot_shard_every must be divisible by checkpoint_every")
        if self.recovery_every % self.checkpoint_every:
            raise ValueError("recovery_every must be divisible by checkpoint_every")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        for name in (
            "train_steps",
            "batch_size",
            "log_every",
            "eval_every",
            "ar_eval_every",
            "eval_examples_per_count",
            "final_examples_per_count",
            "phase_examples_per_count",
            "phase_head_selection_examples_per_count",
            "permutation_examples_per_count",
            "analysis_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position_encodings"] = list(self.position_encodings)
        result["enabled_model_variants"] = list(self.enabled_model_variants)
        result["count_max"] = self.count_max
        result["count_max_alias"] = "read-only alias of count_max_threshold"
        result["effective_needle_pool_seed"] = self.effective_needle_pool_seed
        result["corpus_test_fraction"] = self.corpus_test_fraction
        if self.max_steps_for_language_pred < self.train_steps:
            result["training_objective"] = (
                "teacher-forced weighted next-token cross-entropy over every non-padding "
                f"token through step {self.max_steps_for_language_pred}; from step "
                f"{self.max_steps_for_language_pred + 1}, task-output targets only, "
                "starting inclusively at <Ans> for nonthinking and <Think> for thinking"
            )
            result["training_loss_schedule"] = (
                f"steps 1-{self.max_steps_for_language_pred}: all_sequence; steps "
                f"{self.max_steps_for_language_pred + 1}-{self.train_steps}: task_output"
            )
        else:
            result["training_objective"] = (
                "teacher-forced weighted next-token cross-entropy over every non-padding "
                "token for all configured training steps"
            )
            result["training_loss_schedule"] = (
                f"steps 1-{self.train_steps}: all_sequence; task-output switch is after training"
            )
        result["task_output_scope"] = {
            "nonthinking": "<Ans> through <EOS>, inclusive",
            "thinking": "<Think> through <EOS>, inclusive",
        }
        result["task_output_component_partition"] = {
            "final_count": "atomic answer token",
            "trace": (
                "retrieved marker identities only"
                if self.task_output_trace_delimiters_as_structure
                else "trace query/delimiter tokens plus retrieved marker identities"
            ),
            "structure": (
                "<Think>, repeated <Sep>, </Think>, <Ans>, and <EOS>"
                if self.task_output_trace_delimiters_as_structure
                else "output tags and <EOS> not assigned to count or trace"
            ),
        }
        result["answer_query_contrastive_objective"] = {
            "active": bool(self.answer_query_contrastive_weight > 0),
            "activation_phase": "task_output",
            "query_position": "<Ans> input position whose logits predict the count token",
            "weight": self.answer_query_contrastive_weight,
            "temperature": self.answer_query_contrastive_temperature,
            "inference_change": False,
        }
        result["scheduled_sampling_objective"] = {
            "active": bool(
                self.task_output_scheduled_sampling_max_probability > 0
            ),
            "activation_phase": "task_output",
            "schedule": "linear_from_zero",
            "maximum_probability": (
                self.task_output_scheduled_sampling_max_probability
            ),
            "scope": "generated continuation inputs after the fixed prompt prefix",
            "targets_changed": False,
            "inference_change": False,
        }
        result["task_occurrence_ratio_definition"] = (
            "example-level probability of formatting a training corpus window as a counting task"
        )
        result["joint_sampler_within_cell_policy"] = (
            "all_legal_starts"
            if self.joint_sampler_max_starts_per_cell is None
            else (
                "deterministic_evenly_spaced_cap_"
                f"{self.joint_sampler_max_starts_per_cell}"
            )
        )
        result["task_context_order"] = (
            "fresh_random_permutation_of_each_selected_source_window"
            if self.permute_task_context_tokens
            else "natural_contiguous_source_window"
        )
        result["sequence_layout"] = "query_first"
        thinking_trace = (
            "(<Sep> marker)*n"
            if self.trace_format == "separator"
            else "(number marker)*n"
        )
        result["sequence_templates"] = {
            "nonthinking": f"<BOS> query[5] data[{self.seq_len}] <Ans> number <EOS>",
            "thinking": (
                f"<BOS> query[5] data[{self.seq_len}] <Think> {thinking_trace} "
                "</Think> <Ans> number <EOS>"
            ),
        }
        result["checkpoint_policy"] = {
            "analysis_snapshot_every": self.checkpoint_every,
            "optimizer_recovery_every": self.recovery_every,
            "snapshot_shard_every": self.snapshot_shard_every,
            "snapshot_dtype": self.snapshot_dtype,
        }
        result["readout_parameterization"] = (
            "atomic_count_rows_untied; all_other_rows_tied"
            if self.untie_atomic_count_readout
            else ("fully_tied" if self.tie_word_embeddings else "fully_untied")
        )
        return result


def preset_config(preset: str = "debug", **overrides: Any) -> V20Config:
    cfg = V20Config(preset="main")
    if preset == "debug":
        cfg = replace(
            cfg,
            preset="debug",
            seq_len=48,
            count_max_threshold=4,
            n_positions=96,
            max_relative_distance=96,
            train_steps=6,
            batch_size=4,
            warmup_steps=2,
            log_every=1,
            eval_every=3,
            ar_eval_every=3,
            checkpoint_every=3,
            recovery_every=3,
            snapshot_shard_every=3,
            precision="float32",
            eval_examples_per_count=2,
            ar_examples_per_count=1,
            permutation_examples_per_count=1,
            attention_examples_per_count=1,
            state_train_examples_per_count=2,
            state_eval_examples_per_count=1,
            analysis_batch_size=8,
            phase_examples_per_count=1,
            phase_head_selection_examples_per_count=1,
            phase_cloud_steps=(0, 3, 6),
        )
    elif preset != "main":
        raise ValueError(f"unknown preset: {preset}")
    unknown = sorted(set(overrides) - set(cfg.__dataclass_fields__))
    if unknown:
        raise TypeError(f"unknown V20Config overrides: {unknown}")
    if "enabled_model_variants" in overrides:
        overrides["enabled_model_variants"] = tuple(overrides["enabled_model_variants"])
        if "position_encodings" in overrides:
            overrides["position_encodings"] = tuple(overrides["position_encodings"])
        if "position_encodings" not in overrides:
            overrides["position_encodings"] = tuple(
                position
                for position in SUPPORTED_POSITION_ENCODINGS
                if any(
                    value.startswith(f"{position}/")
                    for value in overrides["enabled_model_variants"]
                )
            )
    elif "position_encodings" in overrides:
        overrides["position_encodings"] = tuple(overrides["position_encodings"])
        overrides["enabled_model_variants"] = tuple(
            f"{position}/{mode}"
            for position in overrides["position_encodings"]
            for mode in SUPPORTED_MODES
        )
    cfg = replace(cfg, **overrides)
    if cfg.rpe_max_update:
        cfg = replace(cfg, max_relative_distance=cfg.max_render_len - 1)
    cfg.validate()
    return cfg


def config_from_dict(values: dict[str, Any]) -> V20Config:
    data = dict(values)
    legacy_loss_schedule = "max_steps_for_language_pred" not in data
    alias = data.pop("count_max", None)
    threshold = int(data["count_max_threshold"])
    if alias is not None and int(alias) != threshold:
        raise ValueError("serialized count_max alias disagrees with count_max_threshold")
    for derived in (
        "count_max_alias",
        "effective_needle_pool_seed",
        "corpus_test_fraction",
        "training_objective",
        "training_loss_schedule",
        "task_output_scope",
        "task_occurrence_ratio_definition",
        "sequence_layout",
        "sequence_templates",
        "checkpoint_policy",
        "answer_query_contrastive_objective",
        "scheduled_sampling_objective",
        "readout_parameterization",
        "joint_sampler_within_cell_policy",
        "task_context_order",
        "task_output_component_partition",
    ):
        data.pop(derived, None)
    data["position_encodings"] = tuple(data["position_encodings"])
    if "enabled_model_variants" in data:
        data["enabled_model_variants"] = tuple(data["enabled_model_variants"])
    else:
        data["enabled_model_variants"] = tuple(
            f"{position}/{mode}"
            for position in data["position_encodings"]
            for mode in SUPPORTED_MODES
        )
    data.setdefault("final_count_loss_weight", 1.0)
    data.setdefault("cot_trace_loss_weight", 1.0)
    data.setdefault("task_output_loss_reduction", "token_weighted_mean")
    data.setdefault("task_output_count_weight", 1.0)
    data.setdefault("task_output_trace_weight", 1.0)
    data.setdefault("task_output_structure_weight", 0.1)
    data.setdefault("task_output_trace_delimiters_as_structure", False)
    data.setdefault("answer_query_contrastive_weight", 0.0)
    data.setdefault("answer_query_contrastive_temperature", 0.1)
    data.setdefault("task_output_scheduled_sampling_max_probability", 0.0)
    data.setdefault("weight_decay", 0.01)
    data.setdefault("lr_decay_steps", None)
    data.setdefault("min_lr", 0.0)
    # Retained only to reject accidental RPE-era configs with a clear message.
    data.setdefault("rpe_max_update", False)
    # Before revision 5, the main cadence was 1,000 steps. Preserve that value
    # when loading a rare hand-written legacy config that omitted the field.
    data.setdefault("checkpoint_every", 100)
    data.setdefault("recovery_every", 500)
    data.setdefault("snapshot_shard_every", 500)
    data.setdefault("snapshot_dtype", "float16")
    data.setdefault("final_examples_per_count", 50)
    data.setdefault("ar_examples_per_count", 2)
    data.setdefault("permutation_examples_per_count", 1)
    data.setdefault("phase_examples_per_count", 1)
    data.setdefault("phase_head_selection_examples_per_count", 2)
    data.setdefault("phase_cloud_steps", (0, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 5_000, 7_000, 10_000))
    data["phase_cloud_steps"] = tuple(data["phase_cloud_steps"])
    data.setdefault("query_layout", "query_first")
    version_spec = VERSION_SPECS.get(data.get("version", "v20"), VERSION_SPECS["v20"])
    data.setdefault("count_tokenization", version_spec["count_tokenization"])
    data.setdefault("trace_format", version_spec["trace_format"])
    data.setdefault("use_sdpa", True)
    data.setdefault("tie_word_embeddings", True)
    data.setdefault("untie_atomic_count_readout", False)
    data.setdefault("training_count_distribution", "natural")
    data.setdefault("joint_sampler_max_starts_per_cell", 8_192)
    data.setdefault("permute_task_context_tokens", False)
    if legacy_loss_schedule:
        data["max_steps_for_language_pred"] = int(data["train_steps"])
    cfg = V20Config(**data)
    if cfg.rpe_max_update:
        cfg = replace(cfg, max_relative_distance=cfg.max_render_len - 1)
    cfg.validate()
    return cfg


def default_run_name(cfg: V20Config) -> str:
    variants = "-".join(value.replace("nonthinking", "nt").replace("thinking", "t") for value in cfg.enabled_model_variants)
    eval_size = cfg.eval_examples_per_count * cfg.count_max_threshold
    rpe_distance_tag = f"_rped{cfg.max_relative_distance}" if cfg.rpe_max_update else ""
    schedule_tag = (
        "allseq-taskout"
        if cfg.max_steps_for_language_pred < cfg.train_steps
        else "all_sequence"
    )
    trace_tag = "" if cfg.trace_format == "indexed" else f"_trace-{cfg.trace_format}"
    component_loss_tag = (
        ""
        if cfg.task_output_loss_reduction == "token_weighted_mean"
        else (
            f"_taskloss-{cfg.task_output_loss_reduction}"
            f"-c{_float_tag(cfg.task_output_count_weight)}"
            f"-t{_float_tag(cfg.task_output_trace_weight)}"
            f"-s{_float_tag(cfg.task_output_structure_weight)}"
        )
    )
    readout_tag = (
        "_untied-count-readout"
        if cfg.untie_atomic_count_readout
        else ("" if cfg.tie_word_embeddings else "_untied-lm-head")
    )
    contrastive_tag = (
        ""
        if cfg.answer_query_contrastive_weight == 0
        else (
            f"_answer-supcon-w{_float_tag(cfg.answer_query_contrastive_weight)}"
            f"-t{_float_tag(cfg.answer_query_contrastive_temperature)}"
        )
    )
    scheduled_sampling_tag = (
        ""
        if cfg.task_output_scheduled_sampling_max_probability == 0
        else (
            "_scheduled-sampling-p"
            f"{_float_tag(cfg.task_output_scheduled_sampling_max_probability)}"
        )
    )
    context_order_tag = (
        "_taskctx-permuted" if cfg.permute_task_context_tokens else ""
    )
    delimiter_partition_tag = (
        "_trace-markers_grammar-delimiters"
        if cfg.task_output_trace_delimiters_as_structure
        else ""
    )
    lr_decay_tag = (
        "" if cfg.lr_decay_steps is None else f"_lrdecay{cfg.lr_decay_steps}"
    )
    min_lr_tag = "" if cfg.min_lr == 0 else f"_minlr{_float_tag(cfg.min_lr)}"
    return (
        f"{cfg.version}_{cfg.preset}_L{cfg.seq_len}_pool{cfg.needle_pool_size}x{cfg.needle_set_size}_"
        f"pf{_float_tag(cfg.needle_pool_frequency_threshold)}_count1-{cfg.count_max_threshold}{rpe_distance_tag}_"
        f"taskr{_float_tag(cfg.task_occurrence_ratio)}_wd{_float_tag(cfg.weight_decay)}_"
        f"countdist-{cfg.training_count_distribution}_"
        f"fcw{_float_tag(cfg.final_count_loss_weight)}_"
        f"cotw{_float_tag(cfg.cot_trace_loss_weight)}_langsteps{cfg.max_steps_for_language_pred}_"
        f"steps{cfg.train_steps}{lr_decay_tag}{min_lr_tag}_snap{cfg.checkpoint_every}_recover{cfg.recovery_every}_"
        f"evaln{eval_size}_{variants.replace('/', '-')}_{cfg.count_tokenization}{trace_tag}"
        f"{component_loss_tag}{readout_tag}{contrastive_tag}{scheduled_sampling_tag}"
        f"{context_order_tag}{delimiter_partition_tag}_"
        f"query-first_{schedule_tag}_seed{cfg.seed}"
    )


def prepare_run_dir(out_root: str | Path, cfg: V20Config, run_name: str | None = None) -> Path:
    path = Path(out_root) / (run_name or default_run_name(cfg))
    for subdir in ("tables", "figures", "checkpoints", "analysis", "logs", "data"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path
