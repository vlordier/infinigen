# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import io
import os
import sys

import pytest

from infinigen.core import tags as t
from infinigen.datagen.util import submitit_emulator as submitit


def test_subpart_deprecated_member_removed():
    assert "StaircaseWall" not in t.Subpart.__members__


def test_filetee_writes_both_streams_without_closing_outer_stream(tmp_path):
    inner_path = tmp_path / "inner.log"
    outer_stream = io.StringIO()

    with inner_path.open("w") as inner:
        tee = submitit.FileTee(inner, outer_stream)
        tee.write("hello\n")
        tee.flush()
        tee.close()

    assert inner_path.read_text() == "hello\n"
    assert outer_stream.getvalue() == "hello\n"

    outer_stream.write("still-open")
    assert outer_stream.getvalue().endswith("still-open")


def test_job_wrapper_passthrough_streams_files_and_console(tmp_path, monkeypatch):
    stdout_file = tmp_path / "stdout.log"
    stderr_file = tmp_path / "stderr.log"
    stdout_stream = io.StringIO()
    stderr_stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_stream)
    monkeypatch.setattr(sys, "stderr", stderr_stream)

    class SuccessfulPopen:
        def __init__(self, *args, **kwargs):
            self.stdout = iter(["hello stdout\n", "second stdout line\n"])
            self.stderr = iter(["hello stderr\n", "second stderr line\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(submitit.subprocess, "Popen", SuccessfulPopen)

    submitit.job_wrapper(
        command=["ignored"],
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        stdout_passthrough=True,
    )

    assert stdout_file.read_text() == "hello stdout\nsecond stdout line\n"
    assert stderr_file.read_text() == "hello stderr\nsecond stderr line\n"
    assert stdout_stream.getvalue() == "hello stdout\nsecond stdout line\n"
    assert stderr_stream.getvalue() == "hello stderr\nsecond stderr line\n"


def test_job_wrapper_passthrough_propagates_exit_code(tmp_path, monkeypatch):
    stdout_file = tmp_path / "stdout.log"
    stderr_file = tmp_path / "stderr.log"
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    class FailingPopen:
        def __init__(self, *args, **kwargs):
            self.stdout = iter(["partial stdout\n"])
            self.stderr = iter(["partial stderr\n"])

        def wait(self):
            return 7

    monkeypatch.setattr(submitit.subprocess, "Popen", FailingPopen)

    with pytest.raises(SystemExit, match="7"):
        submitit.job_wrapper(
            command=["ignored"],
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            stdout_passthrough=True,
        )

    assert stdout_file.read_text() == "partial stdout\n"
    assert stderr_file.read_text() == "partial stderr\n"


def test_job_wrapper_propagates_non_passthrough_exit_code(tmp_path):
    stdout_file = tmp_path / "stdout.log"
    stderr_file = tmp_path / "stderr.log"
    command = [sys.executable, "-c", "import sys; sys.exit(5)"]

    with pytest.raises(SystemExit, match="5"):
        submitit.job_wrapper(
            command=command,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )


def test_job_wrapper_sets_cuda_visible_devices(tmp_path):
    stdout_file = tmp_path / "stdout.log"
    stderr_file = tmp_path / "stderr.log"
    command = [
        sys.executable,
        "-c",
        f"import os; print(os.environ.get('{submitit.CUDA_VARNAME}', ''))",
    ]

    submitit.job_wrapper(
        command=command,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        cuda_devices=[1, 3],
    )

    assert stdout_file.read_text() == "1,3\n"
    assert stderr_file.read_text() == ""
    assert os.environ.get(submitit.CUDA_VARNAME) is None
