# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Pipeline stage result caching.

Caches intermediate ``RandomStageExecutor`` results on disk so that unchanged
upstream stages can be skipped on subsequent runs.  Each cached entry stores:

- the stage name
- a content hash of the stage's input parameters
- the serialised result (via ``pickle``)

This speeds up iterative workflows where only downstream stages (e.g. render)
are modified but terrain / population stages remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel indicating a cache miss
_MISS = object()


def _params_hash(params: dict[str, Any]) -> str:
    """Create a deterministic hash of stage parameters.

    Non-serialisable values are converted to their ``repr()`` string
    so the hash function never fails.
    """
    try:
        raw = json.dumps(params, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        raw = repr(sorted(params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class StageCache:
    """Disk-backed cache for pipeline stage results.

    Parameters
    ----------
    cache_dir : Path or str
        Directory where cache files are stored.
    enabled : bool
        If ``False``, all operations are no-ops (cache is transparent).
    """

    def __init__(self, cache_dir: Path | str, *, enabled: bool = True):
        self._cache_dir = Path(cache_dir)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0
        if enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    def _key_path(self, stage_name: str, params_hash: str) -> Path:
        return self._cache_dir / f"{stage_name}_{params_hash}.pkl"

    def get(self, stage_name: str, params: dict[str, Any]) -> Any:
        """Return cached result for *stage_name* or the sentinel :data:`_MISS`.

        Parameters
        ----------
        stage_name : str
            Pipeline stage identifier.
        params : dict
            The stage's input parameters (used for invalidation).
        """
        if not self._enabled:
            self._misses += 1
            return _MISS

        h = _params_hash(params)
        path = self._key_path(stage_name, h)

        if not path.exists():
            self._misses += 1
            logger.debug("Cache miss for stage '%s' (hash=%s)", stage_name, h)
            return _MISS

        try:
            with open(path, "rb") as f:
                result = pickle.load(f)
            self._hits += 1
            logger.info("Cache hit for stage '%s' (hash=%s)", stage_name, h)
            return result
        except Exception:
            logger.warning(
                "Corrupt cache entry for '%s', treating as miss", stage_name
            )
            path.unlink(missing_ok=True)
            self._misses += 1
            return _MISS

    def put(self, stage_name: str, params: dict[str, Any], result: Any) -> None:
        """Store *result* in the cache for *stage_name*."""
        if not self._enabled:
            return

        h = _params_hash(params)
        path = self._key_path(stage_name, h)

        try:
            with open(path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Cached result for stage '%s' (hash=%s)", stage_name, h)
        except Exception:
            logger.warning("Failed to cache result for stage '%s'", stage_name)

    def invalidate(self, stage_name: str | None = None) -> int:
        """Remove cached entries.

        Parameters
        ----------
        stage_name : str or None
            If provided, only remove entries matching this stage.
            If ``None``, remove *all* entries.

        Returns
        -------
        int
            Number of entries removed.
        """
        if not self._enabled:
            return 0

        removed = 0
        pattern = f"{stage_name}_*.pkl" if stage_name else "*.pkl"
        for path in self._cache_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1

        logger.info("Invalidated %d cache entries (stage=%s)", removed, stage_name)
        return removed

    def is_miss(self, value: Any) -> bool:
        """Return ``True`` if *value* is the cache-miss sentinel."""
        return value is _MISS
