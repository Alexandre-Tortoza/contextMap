"""Measure the scale of SAM3's ``masks_logits`` and ``masks`` tensors (issue #610).

``Sam3ImageProcessorRuntime`` thresholds ``masks_logits`` (or ``masks`` without it) directly
with ``Sam3Config.mask_threshold``, as if the values were probabilities. Nothing in the repository
records what the SDK actually returns. This probe runs the real SAM3 image processor through the
repository's own runtime path (``Sam3ImageProcessorRuntime.predict``: model placement check,
``torch.inference_mode``, autocast, ``set_image``, ``set_confidence_threshold``,
``set_text_prompt``, the current thresholding), over a small declared image set, and keeps each
raw ``set_text_prompt`` output next to what the runtime made of it.

It records, for ``masks_logits`` and ``masks``: dtype, shape, min/max, a histogram, the share of
values outside ``[0, 1]`` and whether the tensor is binary; how the current rule
(``x >= mask_threshold``) and the sigmoid rule (``sigmoid(x) >= mask_threshold``) binarize it,
and how each agrees with the SDK's own ``masks``; and the checkpoint and SDK identity. A few
windows of the raw tensors are written as a compact JSON fixture for the contract test in
``tests/visual_perception/backends/test_sam3.py``.

The probe never changes ``sam3.py``: the correction waits for the measurement.

Usage:
    python sam3_mask_scale.py --sequence DIR --model-version REVISION
        [--frames ID ...] [--prompts TEXT ...] [--processor module:attr] [--run-number 1]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contextmap.evaluation import DriverRun
from contextmap.ingestion import (
    ImageEncoding,
    ImageObservation,
    SequenceArtifactReader,
    SourceObservationId,
)
from contextmap.region_perception import DiscoveryInput, build_discovery_passes
from contextmap.region_perception.adapters.sam3 import (
    Sam3Config,
    Sam3ImageProcessorRuntime,
    Sam3Strategy,
)
from contextmap.shared import ArtifactReference
from contextmap.visual_perception import SourceImage, prepare_image

EXPERIMENT = "sam3-mask-scale-20260927"
ISSUE = 610

DEFAULT_FRAMES = (
    "camera_1_image_raw-000215",
    "camera_1_image_raw-001295",
    "camera_1_image_raw-002267",
)
"""The first, eleventh and last of the 20 frames scenario 1.0.4 froze for corridor-02."""

DEFAULT_PROMPTS = ("floor", "wall", "door")
"""Declared text prompts: surfaces every corridor frame shows. Not tuned; they only have to
produce masks."""

RAW_KEYS = ("masks_logits", "masks", "boxes", "scores")

HISTOGRAM_EDGES = (
    -20.0,
    -10.0,
    -5.0,
    -2.0,
    -1.0,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
)
"""Inner bin edges; the first and last bins are unbounded, so every finite value is counted."""

SCALE_RULE = (
    "probability when every finite value of the tensor the runtime consumes lies in [0, 1]; "
    "logit when any lies outside; undetermined without any finite value"
)


@dataclass
class TensorStatistics:
    """Running statistics of one kind of mask tensor, over any number of SDK outputs."""

    dtypes: set[str] = field(default_factory=set)
    shapes: list[list[int]] = field(default_factory=list)
    count: int = 0
    nonfinite: int = 0
    minimum: float | None = None
    maximum: float | None = None
    below_zero: int = 0
    above_one: int = 0
    non_binary: int = 0
    histogram: list[int] = field(default_factory=lambda: [0] * (len(HISTOGRAM_EDGES) + 1))

    def update(self, array: np.ndarray[Any, Any], *, dtype: str) -> None:
        """Add one tensor, already detached to NumPy; ``dtype`` is its SDK dtype."""
        values = np.asarray(array, dtype=np.float64).ravel()
        finite = values[np.isfinite(values)]
        self.dtypes.add(dtype)
        self.shapes.append([int(size) for size in np.shape(array)])
        self.count += int(values.size)
        self.nonfinite += int(values.size - finite.size)
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
            self.minimum = low if self.minimum is None else min(self.minimum, low)
            self.maximum = high if self.maximum is None else max(self.maximum, high)
        self.below_zero += int((finite < 0).sum())
        self.above_one += int((finite > 1).sum())
        self.non_binary += int(((finite != 0) & (finite != 1)).sum())
        # searchsorted à direita: o valor v cai no bin [edges[i-1], edges[i]).
        bins = np.searchsorted(HISTOGRAM_EDGES, finite, side="right")
        counts = np.bincount(bins, minlength=len(HISTOGRAM_EDGES) + 1)
        self.histogram = [int(a + b) for a, b in zip(self.histogram, counts, strict=True)]

    @property
    def finite(self) -> int:
        """Return how many finite values were seen."""
        return self.count - self.nonfinite

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible statistics."""
        edges: list[float | None] = [None, *HISTOGRAM_EDGES, None]
        return {
            "dtype": sorted(self.dtypes),
            "shape": self.shapes,
            "count": self.count,
            "nonfinite": self.nonfinite,
            "min": self.minimum,
            "max": self.maximum,
            "below_zero": self.below_zero,
            "above_one": self.above_one,
            "outside_unit_interval_fraction": (
                None if self.finite == 0 else (self.below_zero + self.above_one) / self.finite
            ),
            "binary": self.finite > 0 and self.non_binary == 0,
            "histogram": [
                {"from": edges[index], "to": edges[index + 1], "count": count}
                for index, count in enumerate(self.histogram)
            ],
        }


def observed_scale(statistics: TensorStatistics) -> str:
    """Classify a tensor by :data:`SCALE_RULE`."""
    if statistics.finite == 0:
        return "undetermined"
    if statistics.below_zero or statistics.above_one:
        return "logit"
    return "probability"


def sigmoid(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return the logistic function, written with ``tanh`` so it never overflows."""
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * np.asarray(values, dtype=np.float64))))


def binarization_comparison(
    values: np.ndarray[Any, Any], masks: np.ndarray[Any, Any] | None, *, threshold: float
) -> dict[str, Any]:
    """Compare the current rule, the sigmoid rule and the SDK's own ``masks``.

    Args:
        values: The tensor the runtime thresholds.
        masks: The SDK's ``masks`` of the same proposals, when it returned them.
        threshold: ``Sam3Config.mask_threshold``.

    Returns:
        Foreground counts under both rules, how many pixels they disagree on, and, when the SDK
        ``masks`` are binary and aligned, the share of pixels each rule agrees with them on.
    """
    x = np.asarray(values, dtype=np.float64)
    current = x >= threshold
    by_sigmoid = sigmoid(x) >= threshold
    reference = None if masks is None else np.asarray(masks)
    binary = (
        reference is not None
        and reference.shape == x.shape
        and bool(np.isin(reference, (0, 1)).all())
    )
    comparable = binary and x.size > 0
    aligned = reference.astype(bool) if comparable and reference is not None else None
    return {
        "pixels": int(x.size),
        "current_rule_foreground": int(current.sum()),
        "sigmoid_rule_foreground": int(by_sigmoid.sum()),
        "rules_disagree": int((current != by_sigmoid).sum()),
        "masks_binary": binary,
        "masks_foreground": None if aligned is None else int(aligned.sum()),
        "masks_equal_current_rule": None if aligned is None else float((aligned == current).mean()),
        "masks_equal_sigmoid_rule": (
            None if aligned is None else float((aligned == by_sigmoid).mean())
        ),
    }


def sample_window(
    values: np.ndarray[Any, Any], *, threshold: float, size: int, scale: str
) -> tuple[int, int, int, int] | None:
    """Choose the window of one proposal's mask tensor that the fixture keeps.

    Under a logit scale the window is centred on a pixel where the current rule and the sigmoid
    rule disagree (``0 <= x < threshold``), so the fixture pins exactly what a correction would
    change. Otherwise, or when no such pixel exists, it is centred on the boundary of the mask the
    current rule draws. The window keeps its size inside the image.

    Returns:
        ``(x_min, y_min, x_max, y_max)``, half-open, or ``None`` when the mask has no boundary.
    """
    x = np.asarray(values, dtype=np.float64)
    current = x >= threshold
    candidates = np.empty((0, 2), dtype=np.int64)
    if scale == "logit":
        candidates = np.argwhere(current != (sigmoid(x) >= threshold))
    if candidates.size == 0:
        padded = np.pad(current, 1, mode="edge")
        background_neighbour = (
            ~padded[:-2, 1:-1] | ~padded[2:, 1:-1] | ~padded[1:-1, :-2] | ~padded[1:-1, 2:]
        )
        candidates = np.argwhere(current & background_neighbour)
    if candidates.size == 0:
        return None
    row, column = (int(value) for value in candidates[len(candidates) // 2])
    height, width = x.shape
    window_height, window_width = min(size, height), min(size, width)
    y0 = min(max(row - window_height // 2, 0), height - window_height)
    x0 = min(max(column - window_width // 2, 0), width - window_width)
    return (x0, y0, x0 + window_width, y0 + window_height)


def fixture_sample(
    *,
    frame: str,
    prompt: str,
    proposal_index: int,
    box: Sequence[float],
    score: float,
    logits: np.ndarray[Any, Any],
    masks: np.ndarray[Any, Any] | None,
    window: tuple[int, int, int, int],
    logits_dtype: str,
    masks_dtype: str | None,
) -> dict[str, Any]:
    """Keep one window of one proposal's raw tensors, with exact values.

    Values are stored as the float64 of the SDK value: every float32, float16 and bfloat16
    number is exact in float64 and in JSON, so no rounding moves a pixel across a threshold.
    """
    x0, y0, x1, y1 = window
    tensor = np.asarray(logits, dtype=np.float64)
    return {
        "frame": frame,
        "prompt": prompt,
        "proposal_index": proposal_index,
        "box": [float(value) for value in box],
        "score": float(score),
        "window": [x0, y0, x1, y1],
        "width": x1 - x0,
        "height": y1 - y0,
        "masks_logits_dtype": logits_dtype,
        "masks_dtype": masks_dtype,
        "masks_logits": tensor[y0:y1, x0:x1].tolist(),
        "masks": None if masks is None else np.asarray(masks)[y0:y1, x0:x1].tolist(),
        "full_min": float(np.nanmin(tensor)),
        "full_max": float(np.nanmax(tensor)),
    }


def fixture_document(
    *,
    samples: Sequence[Mapping[str, Any]],
    scale: str,
    config: Sam3Config,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the fixture the contract test reads: the scale, the configuration, the samples."""
    return {
        "experiment": EXPERIMENT,
        "issue": ISSUE,
        "observed_scale": scale,
        "rule": SCALE_RULE,
        "checkpoint": config.checkpoint,
        "model_version": config.model_version,
        "device": config.device,
        "precision": config.precision,
        "score_threshold": config.score_threshold,
        "mask_threshold": config.mask_threshold,
        "config_digests": {
            prompt: dataclasses.replace(config, prompts=(prompt,)).digest
            for prompt in sorted({sample["prompt"] for sample in samples})
        },
        "identity": dict(identity),
        "samples": list(samples),
    }


def rgb_array(observation: ImageObservation) -> np.ndarray[Any, Any]:
    """Decode one canonical image into the HWC ``uint8`` RGB array the SDK receives.

    Raises:
        ValueError: If the encoding is not 8-bit RGB, BGR or mono.
    """
    data = np.frombuffer(observation.data, dtype=np.uint8)
    shape = (observation.height, observation.width)
    if observation.encoding is ImageEncoding.RGB8:
        return data.reshape(*shape, 3).copy()
    if observation.encoding is ImageEncoding.BGR8:
        return data.reshape(*shape, 3)[:, :, ::-1].copy()
    if observation.encoding is ImageEncoding.MONO8:
        return np.repeat(data.reshape(*shape, 1), 3, axis=2)
    raise ValueError(f"the probe decodes 8-bit RGB, BGR or mono images, not {observation.encoding}")


def checkpoint_identity(repo_id: str, *, cache: Path) -> dict[str, Any]:
    """Read the revision and file blobs of a Hugging Face checkpoint from the local cache.

    A cached file is a link to ``blobs/<hash>``, where the hash is the file's SHA-256 for a large
    (LFS) file: recording the blob name identifies the weights without reading them.
    """
    folder = cache / f"models--{repo_id.replace('/', '--')}"
    reference = folder / "refs" / "main"
    if not reference.is_file():
        return {"repo_id": repo_id, "revision": None, "files": {}}
    revision = reference.read_text(encoding="utf-8").strip()
    snapshot = folder / "snapshots" / revision
    files: dict[str, str] = {}
    if snapshot.is_dir():
        for path in sorted(snapshot.rglob("*")):
            name = path.relative_to(snapshot).as_posix()
            if path.is_symlink():
                files[name] = Path(os.readlink(path)).name
            elif path.is_file():
                files[name] = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    return {"repo_id": repo_id, "revision": revision, "files": files}


def _detached(value: object) -> tuple[str, np.ndarray[Any, Any]] | None:
    """Move one SDK output to NumPy, keeping its dtype name; floats are widened, never rounded."""
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        tensor = detach().to("cpu")
        dtype = str(tensor.dtype)
        if tensor.dtype.is_floating_point:
            tensor = tensor.float()
        return dtype, np.asarray(tensor.numpy())
    array = np.asarray(value)
    return str(array.dtype), array


class RecordingProcessor:
    """Forward the official processor surface and keep each raw ``set_text_prompt`` output."""

    def __init__(self, processor: Any) -> None:
        """Wrap the loaded official processor."""
        self._processor = processor
        self.outputs: list[dict[str, tuple[str, np.ndarray[Any, Any]] | None]] = []

    @property
    def model(self) -> object:
        """Return the wrapped processor's model, whose placement the runtime verifies."""
        return self._processor.model

    @property
    def device(self) -> object:
        """Return the device the wrapped processor moves images to."""
        return self._processor.device

    def set_image(self, image: object) -> object:
        """Encode one image through the wrapped processor."""
        return self._processor.set_image(image)

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        """Apply the proposal threshold through the wrapped processor."""
        return self._processor.set_confidence_threshold(threshold, state=state)

    def reset_all_prompts(self, state: object) -> None:
        """Remove every prompt and result through the wrapped processor."""
        self._processor.reset_all_prompts(state)

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        """Run the prompt and keep the raw tensors before the runtime reads them."""
        output: Mapping[str, object] = self._processor.set_text_prompt(state=state, prompt=prompt)
        self.outputs.append({key: _detached(output.get(key)) for key in RAW_KEYS})
        return output


def _per_proposal(array: np.ndarray[Any, Any], index: int) -> np.ndarray[Any, Any]:
    """Return one proposal's mask as ``H x W``, dropping the SDK's single channel axis."""
    mask = np.asarray(array[index])
    return np.asarray(mask[0]) if mask.ndim == 3 and mask.shape[0] == 1 else mask


def run_probe(
    *,
    sequence: Path,
    frames: Sequence[str],
    prompts: Sequence[str],
    config: Sam3Config,
    processor: Any,
    to_image: Callable[[np.ndarray[Any, Any]], object],
    samples: int,
    window: int,
    identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run every declared prompt on every declared frame and measure the raw mask tensors.

    Args:
        sequence: The sequence artifact holding the frames.
        frames: Observation ids of the declared images.
        prompts: Declared text prompts.
        config: The SAM3 configuration; each call replaces its phrases by one declared
            prompt, so every raw ``set_text_prompt`` output belongs to one call.
        processor: The loaded official SAM3 image processor.
        to_image: Turns the decoded ``H x W x 3`` array into what the processor accepts.
        samples: How many windows the fixture keeps, at most one per frame and prompt.
        window: Side of each window, in pixels.
        identity: Checkpoint, SDK and machine identity, recorded in both documents.

    Returns:
        The measurement, and the fixture or ``None`` when the scale cannot be determined.

    Raises:
        ValueError: If a frame is not an image observation.
    """
    reader = SequenceArtifactReader(sequence)
    recording = RecordingProcessor(processor)
    images: dict[str, object] = {}
    runtime = Sam3ImageProcessorRuntime(processor=recording)
    totals = {"masks_logits": TensorStatistics(), "masks": TensorStatistics()}
    consumed_totals = TensorStatistics()
    calls: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for frame in frames:
        observation = reader.get_observation(SourceObservationId(frame))
        if not isinstance(observation, ImageObservation):
            raise ValueError(f"{frame} is not an image observation")
        images[frame] = to_image(rgb_array(observation))
        image_sha256 = hashlib.sha256(observation.data).hexdigest()
        prepared = prepare_image(
            SourceImage(
                source_observation_id=frame,
                image=ArtifactReference(
                    uri=f"sequence://{frame}",
                    sha256=image_sha256,
                    media_type=f"image/{observation.encoding.value}",
                ),
                width=observation.width,
                height=observation.height,
            ),
            operations=(),
        )
        (full_frame,) = build_discovery_passes(prepared)
        for prompt in prompts:
            prompt_config = dataclasses.replace(config, prompts=(prompt,))
            native = runtime.predict(
                DiscoveryInput(
                    prepared_image=prepared,
                    discovery_pass=full_frame,
                    perception_run_id=EXPERIMENT,
                    perception_result_id=f"{EXPERIMENT}--{frame}",
                ),
                images[frame],
                prompt_config,
            )
            raw = recording.outputs[-1]
            # O runtime consome masks_logits quando existe, senão masks: a escala medida é essa.
            consumed_key = "masks_logits" if raw["masks_logits"] is not None else "masks"
            consumed = raw[consumed_key]
            if consumed is None:
                raise ValueError(f"SAM3 returned neither masks_logits nor masks for {frame}")
            call: dict[str, Any] = {
                "frame": frame,
                "image_sha256": image_sha256,
                "prompt": prompt,
                "config_digest": prompt_config.digest,
                "consumed": consumed_key,
                "proposals": len(consumed[1]),
            }
            for key in ("masks_logits", "masks"):
                detached = raw[key]
                if detached is None:
                    call[key] = None
                    continue
                statistics = TensorStatistics()
                statistics.update(detached[1], dtype=detached[0])
                totals[key].update(detached[1], dtype=detached[0])
                call[key] = statistics.to_record()
            consumed_totals.update(consumed[1], dtype=consumed[0])
            masks = raw["masks"]
            call["binarization"] = binarization_comparison(
                consumed[1],
                None if masks is None or consumed_key == "masks" else masks[1],
                threshold=config.mask_threshold,
            )
            call["runtime_proposals"] = len(native.proposals)
            call["runtime_foreground"] = int(
                sum(int(proposal.mask.as_array().sum()) for proposal in native.proposals)
            )
            calls.append(call)
            candidates.extend(_best_candidate(frame, prompt, raw, consumed, consumed_key))
    scale = observed_scale(consumed_totals)
    measurement = {
        "experiment": EXPERIMENT,
        "issue": ISSUE,
        "frames": list(frames),
        "prompts": list(prompts),
        "config": {
            "checkpoint": config.checkpoint,
            "model_version": config.model_version,
            "device": config.device,
            "precision": config.precision,
            "score_threshold": config.score_threshold,
            "mask_threshold": config.mask_threshold,
        },
        "identity": dict(identity or {}),
        "calls": calls,
        "total": {key: statistics.to_record() for key, statistics in totals.items()},
        "consumed_total": consumed_totals.to_record(),
        "observed_scale": scale,
        "rule": SCALE_RULE,
    }
    if scale == "undetermined":
        return measurement, None
    kept = []
    for candidate in candidates:
        chosen = sample_window(
            candidate["values"], threshold=config.mask_threshold, size=window, scale=scale
        )
        if chosen is None:
            continue
        kept.append(
            fixture_sample(
                frame=candidate["frame"],
                prompt=candidate["prompt"],
                proposal_index=candidate["index"],
                box=candidate["box"],
                score=candidate["score"],
                logits=candidate["values"],
                masks=candidate["masks"],
                window=chosen,
                logits_dtype=candidate["dtype"],
                masks_dtype=candidate["masks_dtype"],
            )
        )
        if len(kept) == samples:
            break
    if not kept:
        return measurement, None
    return measurement, fixture_document(
        samples=kept, scale=scale, config=config, identity=identity or {}
    )


def _best_candidate(
    frame: str,
    prompt: str,
    raw: Mapping[str, tuple[str, np.ndarray[Any, Any]] | None],
    consumed: tuple[str, np.ndarray[Any, Any]],
    consumed_key: str,
) -> list[dict[str, Any]]:
    """Return the highest-scoring proposal of one call as a fixture candidate, if any."""
    scores, boxes, masks = raw["scores"], raw["boxes"], raw["masks"]
    if scores is None or boxes is None or len(scores[1]) == 0:
        return []
    index = int(np.argmax(np.asarray(scores[1], dtype=np.float64)))
    aligned_masks = None if masks is None or consumed_key == "masks" else masks
    return [
        {
            "frame": frame,
            "prompt": prompt,
            "index": index,
            "box": [float(value) for value in np.asarray(boxes[1][index]).ravel()],
            "score": float(scores[1][index]),
            "values": _per_proposal(consumed[1], index),
            "dtype": consumed[0],
            "masks": None if aligned_masks is None else _per_proposal(aligned_masks[1], index),
            "masks_dtype": None if aligned_masks is None else aligned_masks[0],
        }
    ]


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write a JSON document into a file that must not exist yet.

    Raises:
        FileExistsError: If the file exists: a measurement is never overwritten.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=1, sort_keys=True) + "\n")


def _load_callable(target: str) -> Callable[[], Any]:
    """Import ``module:attribute`` and return it; it must be callable with no argument."""
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"--processor must look like 'module:attribute', got {target!r}")
    loaded = getattr(importlib.import_module(module_name), attribute)
    if not callable(loaded):
        raise ValueError(f"--processor {target!r} is not callable")
    return loaded  # type: ignore[no-any-return]


def official_processor() -> object:
    """Build the official image processor exactly as the SAM3 SDK documents it."""
    builder = importlib.import_module("sam3.model_builder")
    processor_module = importlib.import_module("sam3.model.sam3_image_processor")
    return processor_module.Sam3Processor(builder.build_sam3_image_model())


def _git_commit(directory: Path) -> str | None:
    """Return the commit of the Git checkout holding ``directory``, or ``None`` outside one."""
    git = shutil.which("git")
    if git is None or not directory.is_dir():
        return None
    result = subprocess.run(
        [git, "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_identity(repo_id: str) -> dict[str, Any]:
    """Record the checkpoint, the SDK and the machine; nothing here stops the probe."""
    spec = importlib.util.find_spec("sam3")
    origin = None if spec is None or spec.origin is None else Path(spec.origin).parent
    torch = importlib.import_module("torch")
    cuda = bool(torch.cuda.is_available())
    executable = shutil.which("nvidia-smi")
    gpus = None
    if executable is not None:
        query = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        gpus = query.stdout.strip().splitlines()
    cache = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    )
    return {
        "checkpoint": checkpoint_identity(repo_id, cache=cache),
        "sam3_sdk": {
            "version": _version("sam3"),
            "git_commit": None if origin is None else _git_commit(origin),
        },
        "contextmap": {
            "version": _version("contextmap"),
            "git_commit": _git_commit(Path(__file__).resolve().parent),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda else None,
        },
        "nvidia_smi": gpus,
        "packages": {name: _version(name) for name in ("numpy", "Pillow", "transformers")},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Probe the real processor and write ``measurement.json`` and ``samples.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", default=list(DEFAULT_FRAMES))
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--checkpoint", default="facebook/sam3")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--processor", default=None, help="module:attribute building the processor")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--run-number", type=int, default=1)
    parser.add_argument("--outputs-root", type=Path, default=None)
    options = parser.parse_args(argv)

    run = DriverRun.for_driver(
        Path(__file__),
        experiment=EXPERIMENT,
        run_number=options.run_number,
        outputs_root=options.outputs_root,
    )
    root = run.outputs_root / run.experiment / run.run_id
    if root.exists():
        raise FileExistsError(f"{root} already exists: a measurement is never overwritten")
    config = Sam3Config(
        checkpoint=options.checkpoint,
        model_version=options.model_version,
        device=options.device,
        precision=options.precision,
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompts=(options.prompts[0],),
        score_threshold=options.score_threshold,
    )
    # Os frames são conferidos antes de carregar o modelo: um id errado não custa o load.
    reader = SequenceArtifactReader(options.sequence)
    for frame in options.frames:
        if not isinstance(reader.get_observation(SourceObservationId(frame)), ImageObservation):
            raise ValueError(f"{frame} is not an image observation of {options.sequence}")
    build = official_processor if options.processor is None else _load_callable(options.processor)
    processor = build()
    image_module = importlib.import_module("PIL.Image")
    identity = environment_identity(options.checkpoint)
    identity["processor"] = options.processor or "sam3 SDK: Sam3Processor(build_sam3_image_model())"
    identity["sequence_artifact_id"] = str(reader.manifest.artifact_id)
    measurement, fixture = run_probe(
        sequence=options.sequence,
        frames=options.frames,
        prompts=options.prompts,
        config=config,
        processor=processor,
        # Um array HWC uint8 de 3 canais já é RGB para o Pillow.
        to_image=image_module.fromarray,
        samples=options.samples,
        window=options.window,
        identity=identity,
    )
    write_json(root / "measurement.json", measurement)
    print(f"measurement: {root / 'measurement.json'} (scale: {measurement['observed_scale']})")
    if fixture is None:
        print("no fixture: no proposal had a mask to sample", file=sys.stderr)
        return 1
    write_json(root / "samples.json", fixture)
    print(f"fixture: {root / 'samples.json'} ({len(fixture['samples'])} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
