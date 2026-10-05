# Containers and Kubernetes training

The application is a batch training CLI, so deploy it as a Kubernetes Job.
The image contains the installed application, diagnostics and the `train` extra;
models, datasets, credentials and checkpoints stay outside the image.

## Build and publish

```bash
docker build -t palingenesis:local .
docker run --rm palingenesis:local version
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/configs:/config:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/outputs:/outputs" \
  -v "$PWD/cache:/cache" \
  palingenesis:local train --config /config/train.yaml
```

Create these host directories first and make `outputs` and `cache` writable by
UID/GID 10001. Create `configs/train.yaml` using the ConfigMap's `train.yaml` example.
The YAML must use container paths, not host paths.

`.github/workflows/container.yaml` runs Python/YAML/Dockerfile lint, formatting, configuration/deployment
tests, a Docker build, installed-wheel smoke tests and CPU training regressions.
PRs build and test without publishing. Pushes to `main`, `v*` tags and manual
dispatch publish to `ghcr.io/<lowercase-owner>/<lowercase-repository>` using
`GITHUB_TOKEN` with `packages: write`. No Docker Hub credentials are needed.
Tags include `sha-<full-commit>`, a version for `v*` releases and `latest` on main.
For production, use the published digest rather than a mutable tag.

The build uses `uv sync --locked` with the committed lockfile (Python 3.12,
PyTorch CUDA 12.9), and fails if dependency metadata and lockfile diverge.
The image targets Linux amd64. Base image tags and Debian security packages can
change between builds; pin their digests and snapshot repositories if byte-for-byte
reproducibility is required.

## Runtime inputs

| Input | Kubernetes mechanism | Container path |
| --- | --- | --- |
| Training YAML | ConfigMap `palingenesis-config` | `/config/train.yaml` |
| Local JSONL/Parquet data, optional local model | PVC `palingenesis-data`, read-only | `/data` |
| Checkpoints and final export | PVC `palingenesis-outputs`, writable | `/outputs` |
| Hugging Face downloads and Torch/Triton compile caches | PVC `palingenesis-cache`, writable | `/cache` |
| DataLoader shared memory | Memory-backed `emptyDir` | `/dev/shm` |
| Temporary files, home, working directory | Ephemeral `emptyDir` volumes | `/tmp`, `/home/trainer`, `/workspace` |
| Hugging Face access token | Optional Secret `palingenesis-credentials`, key `HF_TOKEN` | Environment |
| Private GHCR pull credentials | `imagePullSecrets` | Used by kubelet |

Never put tokens in ConfigMaps, build arguments or committed YAML. For gated models,
create the optional application Secret from a local file containing the token:

```bash
kubectl create secret generic palingenesis-credentials --from-file=HF_TOKEN=/secure/hf-token
```

For private images create `ghcr-pull` in the Job's namespace using a GitHub personal
access token (classic) with `read:packages`, package access and any required SSO
authorization. Uncomment `imagePullSecrets` in `job.yaml`. Public packages can be
pulled without authentication. A newly published GHCR package may need its visibility
or repository access configured in GitHub's package settings.

## Submit a training run

1. Install the NVIDIA driver, NVIDIA Container Toolkit and Kubernetes NVIDIA device
   plugin on GPU nodes. The image supplies user-space CUDA libraries, not a driver.
   Use a CUDA 12.9-compatible driver (the project's documented baseline is >=575).
2. Edit `storage.yaml` for your storage class, capacity and access modes. The sample
   uses ReadWriteOnce and one trainer Pod; concurrent runs and multiple nodes need
   suitable storage and separate output directories.
3. Apply `storage.yaml`, then populate `palingenesis-data` with `train.jsonl` using a
   temporary uploader Pod or your storage system. Each JSONL row must contain
   `messages`, for example:

   ```json
   {"messages":[{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"}]}
   ```

4. Edit `configmap.yaml` for the model, dataset and run ID. A Hugging Face dataset
   identifier can replace the local path; set its `dataset_split` as needed. For an
   offline model, set `model.name_or_path: /data/models/my-model`, stage all artifacts
   and set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` in the Job.
5. Replace the Job image with your repository's published digest. Adjust GPU, CPU,
   memory and shared-memory sizes for the actual model. The sample requests one GPU
   and is intended for a small model; GPU VRAM is separate from the Pod's RAM limit.
6. Run pre-training diagnostics on the target GPU node with the same image, mounts
   and credentials: override the Job args with
   `[diagnose, --config, /config/train.yaml, --mode, pre, --json]` in a separate Job.
   Review the result before submitting training. Diagnostics may download model
   metadata/tokenizer and read data; CI does not replace this check.

```bash
kubectl apply -f deploy/kubernetes/storage.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/job.yaml
kubectl logs -f job/palingenesis-train
```

The storage must allow UID/GID 10001 to write cache and outputs; `fsGroup` handles
this for supporting CSI drivers. Configure ownership separately if your driver
does not honor it. Memory-backed shared memory counts toward the RAM limit.
Network access to Hugging Face is required unless all artifacts are staged offline.

Logs go to stdout/stderr and can be collected by the cluster logging system. To run
post-training log diagnostics, save `kubectl logs job/palingenesis-train` to a file
and invoke `pgs loss --log_file <file>` or the post-mode diagnostic.
The Job has no automatic retries. `resume_from: auto` reloads the latest checkpoint
from `/outputs/run-001` when you explicitly submit another run. Use a new run ID for
unrelated experiments. Termination does not guarantee an emergency checkpoint;
recovery starts from the last saved checkpoint.

## Multiple GPUs and optional features

For multiple GPUs on one node, change the GPU limit, override `command` to
`[torchrun]` and set args to
`[--standalone, --nproc_per_node=4, -m, palingenesis.train, --config, /config/train.yaml]`.
Configure the training parallelism in YAML. Multiple nodes require explicit
rendezvous and distributed job orchestration; the sample Job does not provide it.

The default image supports SFT/DPO training and the bundled diagnostics. Extras for
vLLM distillation, RL rewards, logging, float8, preparation and MCP are not included.
Build a separate variant by adding the needed `--extra` arguments to `uv sync`;
keep the same lockfile. `hybrid` (causal-conv1d) and FlashAttention need a CUDA devel
build environment and are not supported by this slim build. Qwen3.5 packing needs
the hybrid extension. Use SDPA and disable packing for those models in this image.
Harbor/Docker environments and Kubernetes Agent Sandbox have additional runtime
requirements and need separate deployment designs.

## Quality checks

```bash
ruff check .
ruff format --check .
pytest -q tests/test_container_contract.py tests/test_config_validation.py
```

CI deliberately runs a CPU regression subset that does not download models. GPU
kernels, real model masking and memory capacity must be checked on the target
hardware with the diagnostic tools described in `AGENTS.md`.

References: [GitHub image publishing](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images),
[GHCR authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry),
[Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/),
[Kubernetes volumes](https://kubernetes.io/docs/concepts/storage/volumes/).
