# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def canonicalize_reason(reason):
    reason = reason[:60]
    return reason


def get_configs(log_path, stage):
    scene_folder = log_path.parent.parent
    run_pipeline = scene_folder / "run_pipeline.sh"

    if not run_pipeline.exists():
        return None

    text = run_pipeline.read_text()

    for line in text.split("\n"):
        if "--task " + stage not in line:
            continue

        regex = re.compile("-g(.*)-p")
        match = regex.search(line)

        if not match:
            continue

        return match.groups()[0]


def parse_run_folder(run_folder: Path, args: argparse.Namespace):
    crash_reasons = run_folder / "crash_summaries.txt"

    if not crash_reasons.exists():
        logger.info(f'Could not find crash reasons for {run_folder}')
        return

    crash_reasons = crash_reasons.read_text().split("\n")

    regex = re.compile(
        ".*\s.*\s(.*\/([a-zA-Z0-9]*)\/logs\/(.*))\sreason=[\"'](.*)[\"']\snode='(.*)'"
    )

    records = []
    for x in crash_reasons:
        if not x:
            continue

        match = regex.match(x)
        if match:
            log_path, job_id, stage, reason, node = match.groups()
            stage = stage.split(".")[0].split("_")[0]
            configs = get_configs(Path(log_path), stage)
            record = {
                "log_path": log_path,
                "job_id": job_id,
                "stage": stage,
                "reason": reason,
                "node": node,
                "configs": configs,
                "scenetype": configs.split()[0] if configs else None,
            }
            records.append(record)
        else:
            logger.info(f'Could not match: {x}')

    df = pd.DataFrame.from_records(records)

    return df


def visualize_results(df: pd.DataFrame, args: argparse.Namespace):
    df["reason_canonical"] = df["reason"].apply(canonicalize_reason)

    logger.info('COMMON CRASH REASONS')
    logger.info(df.value_counts('reason_canonical'))
    logger.info('')

    logger.info('CRASH SCENETYPES')
    logger.info(df.value_counts('scenetype'))
    logger.info('')

    for x in df["reason_canonical"].unique():
        logger.info(f'REASON: {x}')

        examples = df[df["reason_canonical"] == x]
        sample_idxs = np.random.choice(
            len(examples), min(10, len(examples)), replace=False
        )

        stages = examples.value_counts("stage")
        if len(stages) > 1:
            logger.info(stages)

        scenetype = examples.value_counts("scenetype")
        if len(scenetype) > 1:
            logger.info(scenetype)

        for idx in sample_idxs:
            row = " ".join(
                [
                    str(x)
                    for x in examples.iloc[idx][
                        ["log_path", "stage", "scenetype"]
                    ].values
                ]
            )
            logger.info(f'  {row}')
        logger.info('')


def main(args):
    run_dfs = [parse_run_folder(run_folder, args) for run_folder in args.input_folder]
    run_dfs = [x for x in run_dfs if x is not None]

    df = pd.concat(run_dfs)
    visualize_results(df, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=Path, required=True, nargs="+")
    args = parser.parse_args()

    main(args)
