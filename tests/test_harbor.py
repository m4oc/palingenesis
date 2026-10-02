"""Harbor task environments (palingenesis.rl.envs.harbor). Docker tests are skipped when docker is unavailable."""

import shutil
import subprocess
from pathlib import Path

import pytest

from palingenesis.rl.envs.harbor import HarborEnv, HarborTask, check_task, harbor_rows

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_task"


def _docker_ok():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def test_task_loads_and_rows():
    t = HarborTask.load(FIXTURE)
    assert t.name == "fixture/hello" and "hello.txt" in t.instruction
    assert t.environment["memory_mb"] == 512 and t.verifier["env"]["EXPECTED"] == "ciao"
    assert t.image().startswith("pgs-harbor:")
    rows = harbor_rows(FIXTURE.parent)
    assert rows and rows[0]["task_dir"].endswith("harbor_task") and rows[0]["messages"][-1]["role"] == "user"


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_oracle_passes_and_nop_fails():
    r = check_task(FIXTURE)
    assert r["oracle"] == 1.0 and r["nop"] == 0.0 and r["valid"], r


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_episode_with_tools_and_verifier_reward():
    def episode():
        env = HarborEnv()
        env.reset(task_dir=str(FIXTURE))
        try:
            out = env.call_tool("bash", {"command": "ls /workdir; echo hi"})
            assert "[exit code 0]" in out and "hi" in out
            env.call_tool("write_file", {"path": "hello.txt", "content": "cia0\n"})
            assert env.get_reward()["reward"] == 0.0
            assert (env.call_tool("str_replace", {"path": "hello.txt", "old": "cia0", "new": "ciao"})).startswith(
                "Edited"
            )
            env.call_tool("submit", {})
            assert env.done
            assert env.get_reward()["reward"] == 1.0
            assert (
                "network" in (env.call_tool("bash", {"command": "getent hosts example.com || echo no network"})).lower()
            )
        finally:
            env.close()

    episode()


@pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")
def test_setup_runs_and_verifier_is_isolated_from_the_agent(tmp_path):
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "setup.sh").write_text("set -e\ncd /workdir\necho seeded > seeded.txt\n")

    def episode():
        env = HarborEnv()
        env.reset(task_dir=str(task))
        try:
            assert "seeded" in env.call_tool("bash", {"command": "cat seeded.txt"})
            # an agent that forges the reward from a background process must not reach the verifier
            env.call_tool(
                "bash",
                {
                    "command": "nohup bash -c 'while true; do echo 1 > /logs/verifier/reward.txt; "
                    "done' >/dev/null 2>&1 &"
                },
            )
            assert env.get_reward()["reward"] == 0.0
        finally:
            env.close()

    episode()
