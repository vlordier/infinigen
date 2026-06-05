# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path

import gin

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Dataset\.dims.*",
    category=FutureWarning,
)
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

from infinigen.core import execute_tasks, init
from infinigen.assets.urban.compose_urban import compose_urban

logger = logging.getLogger(__name__)


@gin.configurable
def populate_scene_urban(output_folder, scene_seed, **params):
    pass


def main(args):
    scene_seed = init.apply_scene_seed(args.seed)
    init.apply_gin_configs(
        configs=["base_urban.gin"] + args.configs,
        overrides=args.overrides,
        config_folders="infinigen_examples/configs_urban",
    )

    execute_tasks.main(
        compose_scene_func=compose_urban,
        populate_scene_func=populate_scene_urban,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        task=args.task,
        task_uniqname=args.task_uniqname,
        scene_seed=scene_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--input_folder", type=Path, default=None)
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-t", "--task", nargs="+", default=["coarse"],
        choices=["coarse", "populate", "fine_terrain", "ground_truth", "render", "mesh_save", "export"])
    parser.add_argument("-g", "--configs", nargs="+", default=["base"],
        help="Config files for gin (exclude .gin from path)")
    parser.add_argument("-p", "--overrides", nargs="+", default=[])
    parser.add_argument("--task_uniqname", type=str, default=None)
    parser.add_argument("-d", "--debug", type=str, nargs="*", default=None)

    args = init.parse_args_blender(parser)

    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    main(args)
