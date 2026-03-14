# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alex Raistrick: refactor, local rendering, video rendering
# - Lahav Lipson: stereo version, local rendering
# - Hei Law: initial version

import logging
import subprocess
import time
from pathlib import Path

import gin

from infinigen.datagen.util.submitit_emulator import LocalJob

logger = logging.getLogger(__name__)

_SEFF_SUBPROCESS_TIMEOUT_SECONDS: int = 30
_SEFF_RETRY_INTERVAL_SECONDS: int = 1
_SEFF_DEFAULT_MAX_RETRIES: int = 10


class JobState:
    NotQueued = "notqueued"
    Queued = "queued"
    Running = "running"
    Succeeded = "succeeded"
    Failed = "crashed"
    Cancelled = "cancelled"


class SceneState:
    NotDone = "notdone"
    Done = "done"
    Crashed = "crashed"


CONCLUDED_JOBSTATES = {JobState.Succeeded, JobState.Failed, JobState.Cancelled}
JOB_OBJ_SUCCEEDED = "MARK_AS_SUCCEEDED"


# Will throw exception if the scene was not found. Sometimes this happens if the scene was queued very very recently
# Keys: JobID ArrayJobID User Group State Clustername Ncpus Nnodes Ntasks Reqmem PerNode Cput Walltime Mem ExitStatus
@gin.configurable
def seff(job_obj, retry_on_error=True, max_retries=_SEFF_DEFAULT_MAX_RETRIES):
    scene_id = job_obj.job_id
    if not scene_id.isdigit():
        raise ValueError(f"Invalid {scene_id=}, expected a numeric SLURM job ID")
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    for attempt in range(max_retries):
        try:
            seff_out = subprocess.check_output(
                f"/usr/bin/seff -d {scene_id}".split(),
                timeout=_SEFF_SUBPROCESS_TIMEOUT_SECONDS,
                stderr=subprocess.PIPE,
            ).decode()
            lines = seff_out.splitlines()
            return dict(zip(lines[0].split(" ")[2:], lines[1].split(" ")[2:]))["State"]
        except subprocess.TimeoutExpired:
            logger.warning(f"seff timed out for {scene_id} (attempt {attempt + 1}/{max_retries})")
            if not retry_on_error or attempt >= max_retries - 1:
                raise
            time.sleep(_SEFF_RETRY_INTERVAL_SECONDS)
        except (subprocess.CalledProcessError, KeyError, IndexError) as e:
            logger.warning(f"seff failed for {scene_id}: {e} (attempt {attempt + 1}/{max_retries})")
            if not retry_on_error or attempt >= max_retries - 1:
                raise
            time.sleep(_SEFF_RETRY_INTERVAL_SECONDS)


def get_scene_state(scene: dict, taskname: str, scene_folder: Path):
    if not scene.get(f"{taskname}_submitted", False):
        return JobState.NotQueued
    elif scene.get(f"{taskname}_crash_recorded", False):
        return JobState.Failed
    elif scene.get(f"{taskname}_force_cancelled", False):
        return JobState.Cancelled

    # if scene['all_done']:
    #    return JobState.Succeeded # TODO Hacky / incorrect for nonfatal

    job_obj = scene[f"{taskname}_job_obj"]

    # for when both local and slurm scenes are being mixed
    if isinstance(job_obj, str):
        assert job_obj == JOB_OBJ_SUCCEEDED
        return JobState.Succeeded
    elif isinstance(job_obj, LocalJob):
        res = job_obj.status()
    elif hasattr(job_obj, "job_id"):
        res = seff(job_obj)
    else:
        raise TypeError(f"Unrecognized {job_obj=}")

    # map from submitit's scene state strings to our JobState enum
    if res in {"PENDING", "REQUEUED"}:
        return JobState.Queued
    elif res == "RUNNING":
        return JobState.Running
    elif not (scene_folder / "logs" / f"FINISH_{taskname}").exists():
        return JobState.Failed

    return JobState.Succeeded


def cancel_job(job_obj):
    if isinstance(job_obj, str):
        assert job_obj == JOB_OBJ_SUCCEEDED
        return JobState.Succeeded
    elif isinstance(job_obj, LocalJob):
        job_obj.kill()
    elif hasattr(job_obj, "job_id"):
        # TODO: does submitit have a cancel?
        subprocess.check_call(["/usr/bin/scancel", str(job_obj.job_id)])
    else:
        raise TypeError(f"Unrecognized {job_obj=}")
