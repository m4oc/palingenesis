"""Validate deployment paths and runtime contracts without downloading a model."""

from pathlib import Path

import yaml

from palingenesis.config import Config

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "kubernetes"


def test_training_config_and_persistent_paths(tmp_path):
    configmap = yaml.safe_load((DEPLOY / "configmap.yaml").read_text())
    path = tmp_path / "train.yaml"
    path.write_text(configmap["data"]["train.yaml"])
    config = Config.from_yaml(path)
    config.validate()
    pod = yaml.safe_load((DEPLOY / "job.yaml").read_text())["spec"]["template"]["spec"]
    container = pod["containers"][0]
    mounts = {m["mountPath"]: m for m in container["volumeMounts"]}
    volumes = {v["name"]: v for v in pod["volumes"]}
    claims = {doc["metadata"]["name"] for doc in yaml.safe_load_all((DEPLOY / "storage.yaml").read_text())}
    for directory in ("/data", "/outputs", "/cache"):
        claim = volumes[mounts[directory]["name"]]["persistentVolumeClaim"]["claimName"]
        assert claim in claims
    assert mounts["/data"]["readOnly"]
    assert not mounts["/outputs"].get("readOnly", False)
    assert config.train.output_dir.startswith("/outputs/")
    assert config.data.dataset.startswith("/data/")
    assert config.train.resume_from == "auto"
    assert container["args"] == ["train", "--config", "/config/train.yaml"]
    assert volumes["config"]["configMap"]["name"] == configmap["metadata"]["name"]


def test_gpu_security_and_writable_runtime_directories():
    pod = yaml.safe_load((DEPLOY / "job.yaml").read_text())["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert pod["securityContext"]["runAsUser"] == 10001
    assert pod["securityContext"]["fsGroup"] == 10001
    assert container["securityContext"]["readOnlyRootFilesystem"]
    assert not pod["automountServiceAccountToken"]
    mounts = {m["mountPath"] for m in container["volumeMounts"] if not m.get("readOnly", False)}
    assert {"/tmp", "/home/trainer", "/workspace", "/dev/shm", "/cache", "/outputs"} <= mounts


def test_package_diagnostics_are_included_in_wheel():
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "agent_tooling" in metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
