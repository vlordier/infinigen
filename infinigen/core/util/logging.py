# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson: logging formats, timer format
# - Alex Raistrick: Timer
# - Alejandro Newell: Suppress
# - Lingjie Mei: disable


import logging
import os
import sys
import time
import typing
from datetime import timedelta
from pathlib import Path

import bpy
import gin


def lazydebug(logger: logging.Logger, msg: typing.Callable, *args, **kwargs):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(msg(), *args, **kwargs)


@gin.configurable
class Timer:
    def __init__(self, desc, disable_timer=False, logger=None):
        self.disable_timer = disable_timer
        if self.disable_timer:
            return
        self.name = f"[{desc}]"
        if logger is None:
            logger = logging.getLogger("infinigen.times")
        self.logger = logger

    def __enter__(self):
        if self.disable_timer:
            return
        self._start = time.perf_counter()
        self.logger.info(f"{self.name}")

    def __exit__(self, exc_type, exc_val, traceback):
        if self.disable_timer:
            return
        elapsed = time.perf_counter() - self._start
        self.duration = timedelta(seconds=elapsed)
        if exc_type is None:
            self.logger.info(f"{self.name} finished in {str(self.duration)}")
        else:
            self.logger.info(f"{self.name} failed with {exc_type}")


class Suppress:
    def __init__(self, logfile=os.devnull):
        self.logfile = logfile

    def __enter__(self):
        self.old = os.dup(1)
        sys.stdout.flush()
        devnull_fd = os.open(self.logfile, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.close(devnull_fd)
        self.level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, traceback):
        sys.stdout.flush()
        os.dup2(self.old, 1)
        os.close(self.old)
        logging.disable(self.level)


class LogLevel:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.orig_level = None

    def __enter__(self):
        self.orig_level = self.logger.level
        self.logger.setLevel(self.level)

    def __exit__(self, *_):
        self.logger.setLevel(self.orig_level)


def save_polycounts(file):
    for col in bpy.data.collections:
        polycount = sum(
            len(obj.data.polygons)
            for obj in col.all_objects
            if (obj.type == "MESH" and obj.data is not None)
        )
        file.write(f"{col.name}: {polycount:,}\n")
    for stat in bpy.context.scene.statistics(bpy.context.view_layer).split(" | ")[2:]:
        file.write(stat + "\n")


@gin.configurable
def create_text_file(log_dir, filename, text=None):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / filename).touch()
    if text is not None:
        (log_dir / filename).write_text(text)


class BadSeedError(ValueError):
    pass
