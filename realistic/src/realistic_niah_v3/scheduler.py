from __future__ import annotations

"""Pure resource-allocation logic for the V3 multi-GPU scheduler."""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Allocation:
    task_id: str
    gpu_ids: tuple[int, ...]


def _choose_gpu_block(
    free_gpu_ids: list[int],
    count: int,
) -> tuple[int, ...] | None:
    if count < 1 or len(free_gpu_ids) < count:
        return None
    if count == 1:
        return (free_gpu_ids[0],)
    for start in range(len(free_gpu_ids) - count + 1):
        block = free_gpu_ids[start : start + count]
        if all(
            right == left + 1
            for left, right in zip(block, block[1:])
        ):
            return tuple(block)
    return tuple(free_gpu_ids[:count])


def allocate_pending_tasks(
    tasks: Iterable[dict[str, Any]],
    *,
    visible_gpu_ids: Iterable[int],
    busy_gpu_ids: Iterable[int] = (),
) -> list[Allocation]:
    """Greedily pack pending tasks without overlapping GPU assignments.

    Tasks are ranked by the frozen plan priority.  If a high-priority TP task
    cannot fit in the remaining GPUs, smaller tasks may backfill the gap.
    """

    visible = tuple(sorted(set(int(gpu) for gpu in visible_gpu_ids)))
    busy = set(int(gpu) for gpu in busy_gpu_ids)
    if not busy.issubset(set(visible)):
        raise ValueError("Busy GPUs must be a subset of visible GPUs")
    free = [gpu for gpu in visible if gpu not in busy]
    allocations: list[Allocation] = []
    ordered = sorted(
        tasks,
        key=lambda task: (-int(task["priority"]), str(task["task_id"])),
    )
    for task in ordered:
        required = int(task["gpus_required"])
        tensor_parallel = int(task["tensor_parallel_size"])
        if required != tensor_parallel or required < 1:
            raise ValueError(
                f"Invalid GPU/TP profile for task {task['task_id']}"
            )
        block = _choose_gpu_block(free, required)
        if block is None:
            continue
        allocations.append(
            Allocation(str(task["task_id"]), block)
        )
        allocated = set(block)
        free = [gpu for gpu in free if gpu not in allocated]
        if not free:
            break
    return allocations


def validate_resource_plan(
    tasks: Iterable[dict[str, Any]],
    *,
    maximum_supported_gpus: int = 8,
) -> None:
    task_ids: set[str] = set()
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in task_ids:
            raise ValueError(f"Duplicate scheduler task id: {task_id}")
        task_ids.add(task_id)
        required = int(task["gpus_required"])
        tensor_parallel = int(task["tensor_parallel_size"])
        if not 1 <= required <= maximum_supported_gpus:
            raise ValueError(f"Invalid GPU count for {task_id}: {required}")
        if tensor_parallel != required:
            raise ValueError(
                f"TP size must equal allocated GPUs for {task_id}"
            )
        if int(task["request_batch_size"]) < 1:
            raise ValueError(f"Invalid request batch size for {task_id}")
        if int(task["max_num_seqs"]) < 1:
            raise ValueError(f"Invalid max_num_seqs for {task_id}")
        utilization = float(task["gpu_memory_utilization"])
        if not 0.5 <= utilization < 1.0:
            raise ValueError(
                f"Invalid GPU memory utilization for {task_id}"
            )
