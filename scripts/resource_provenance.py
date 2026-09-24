#!/usr/bin/env python3
"""Record and select bounded Slurm resources using measured job evidence."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


CPU_CAP = 64
MEMORY_CAP_GB = 64
GPU_CAP = 1
MIN_ACTIVE_CPU_EFFICIENCY = 0.50
MIN_ACTIVE_GPU_UTILIZATION_PERCENT = 50.0
MIN_ACTIVE_GPU_SAMPLES = 3
PREPARE_REQUEST = {"cpus": 16, "memory_gb": 32, "gpus": 0}
FINALIZE_REQUEST = {"cpus": 4, "memory_gb": 32, "gpus": 0}


class ResourceError(RuntimeError):
    pass


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_stage(args):
    manifest_path = args.run_dir / f"{args.stage}_stage_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ResourceError(f"Missing {args.stage} stage manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("stage") != args.stage or payload.get("status") != "PASS":
        raise ResourceError(f"Invalid {args.stage} stage manifest")
    for name, expected in payload.get("artifacts", {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ResourceError(f"Unsafe {args.stage} artifact path")
        path = args.run_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ResourceError(f"Changed {args.stage} artifact: {name}")
    runtime_path = args.run_dir / "runtime_manifest.json"
    if not runtime_path.is_file():
        raise ResourceError("Resume runtime manifest is missing")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if (
        payload.get("run_id") != args.run_id
        or runtime.get("run_id") != args.run_id
        or runtime.get("git_commit") != args.git_commit
        or runtime.get("source_inventory_sha256") != args.source_inventory
    ):
        raise ResourceError("Resume source context differs from the prepared run")
    return {"status": "PASS", "stage": args.stage, "artifact_count": len(payload["artifacts"])}


def verify_profile(args):
    path = args.profile_dir / f"{args.name}.json"
    benchmark_path = args.profile_dir / f"{args.name}-benchmark.json"
    if not path.is_file() or not benchmark_path.is_file():
        raise ResourceError(f"Missing profile evidence for {args.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    expected = {"cpus": args.cpus, "memory_gb": args.memory_gb, "gpus": args.gpus}
    expected_partition = "gpu" if args.gpus else "cpu"
    if (
        payload.get("status") != "PASS"
        or payload.get("hostname") != "compute-0-2"
        or payload.get("partition") != expected_partition
        or payload.get("requested") != expected
        or payload.get("source_git_commit") != args.git_commit
        or payload.get("source_inventory_sha256") != args.source_inventory
        or payload.get("run_id") != args.run_id
        or payload.get("benchmark") != benchmark
        or benchmark.get("status") != "PASS"
    ):
        raise ResourceError(f"Invalid profile evidence for {args.name}")
    return {"status": "PASS", "profile": args.name}


def verify_selected_profile(args):
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    path = args.profile_dir / "selected-model.json"
    benchmark_path = args.profile_dir / "selected-model-benchmark.json"
    if not path.is_file() or not benchmark_path.is_file():
        raise ResourceError("Selected model concurrency profile is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    selected = selection.get("selected", {})
    fit = selection.get("fit_profile", {})
    workers = selection.get("parallel_workers")
    candidates = selection.get("parallel_candidates")
    expected_concurrent = (
        workers * candidates
        if isinstance(workers, int)
        and not isinstance(workers, bool)
        and isinstance(candidates, int)
        and not isinstance(candidates, bool)
        else None
    )
    gpu = payload.get("measured", {}).get("gpu", {})
    if (
        selection.get("status") != "PASS"
        or payload.get("stage") != "selected-model"
        or payload.get("status") != "PASS"
        or payload.get("hostname") != "compute-0-2"
        or payload.get("partition") != ("gpu" if selected.get("gpus") else "cpu")
        or payload.get("requested") != selected
        or payload.get("run_id") != selection.get("run_id")
        or payload.get("source_git_commit") != selection.get("source_git_commit")
        or payload.get("source_inventory_sha256")
        != selection.get("source_inventory_sha256")
        or payload.get("benchmark") != benchmark
        or benchmark.get("status") != "PASS"
        or benchmark.get("fit_threads") != fit.get("cpus")
        or benchmark.get("parallel_workers") != workers
        or benchmark.get("parallel_candidates") != candidates
        or expected_concurrent is None
        or benchmark.get("concurrent_xgboost_fits") != expected_concurrent
        or payload.get("measured", {}).get("max_rss_gb", math.inf)
        > selected.get("memory_gb", 0)
    ):
        raise ResourceError("Selected model concurrency profile does not match its resource selection")
    if selected.get("gpus") == 0:
        if benchmark.get("xgboost_active_cpu_efficiency", 0) <= MIN_ACTIVE_CPU_EFFICIENCY:
            raise ResourceError("Selected concurrent XGBoost phase did not exceed 50% active CPU efficiency")
    elif (
        gpu.get("active_samples", 0) < MIN_ACTIVE_GPU_SAMPLES
        or gpu.get("mean_active_utilization_percent", 0)
        <= MIN_ACTIVE_GPU_UTILIZATION_PERCENT
    ):
        raise ResourceError("Selected GPU profile did not meet active utilization gates")
    return {
        "status": "PASS",
        "profile": "selected-model",
        "active_cpu_efficiency": benchmark.get("xgboost_active_cpu_efficiency"),
    }


def parse_time(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue  # GNU time prefixes failed commands with a diagnostic sentence.
        key, value = line.split("=", 1)
        values[key] = float(value)
    required = {"elapsed_seconds", "user_seconds", "system_seconds", "max_rss_kb", "exit_status"}
    if set(values) != required or values["elapsed_seconds"] <= 0:
        raise ResourceError("Invalid /usr/bin/time evidence")
    return values


def parse_gpu(path):
    if path is None or not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return {
            "samples": 0,
            "active_samples": 0,
            "mean_utilization_percent": None,
            "mean_active_utilization_percent": None,
            "max_memory_used_mib": None,
            "memory_total_mib": None,
        }
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(tuple(float(value.strip()) for value in line.split(",")))
        except ValueError as exc:
            raise ResourceError("Invalid nvidia-smi sample") from exc
    active = [row for row in rows if row[0] > 0]
    return {
        "samples": len(rows),
        "active_samples": len(active),
        "mean_utilization_percent": sum(row[0] for row in rows) / len(rows),
        "mean_active_utilization_percent": (
            sum(row[0] for row in active) / len(active) if active else 0.0
        ),
        "max_memory_used_mib": max(row[1] for row in rows),
        "memory_total_mib": max(row[2] for row in rows),
    }


def record_stage(args):
    timing = parse_time(args.time_file)
    cpus = args.cpus
    memory_gb = args.memory_gb
    gpus = args.gpus
    if not 1 <= cpus <= CPU_CAP or not 1 <= memory_gb <= MEMORY_CAP_GB or not 0 <= gpus <= GPU_CAP:
        raise ResourceError("Requested resources exceed the approved bounds")
    gpu = parse_gpu(args.gpu_log)
    if gpus == 1 and gpu["samples"] == 0:
        raise ResourceError("GPU allocation lacks utilization samples")
    payload = {
        "stage": args.stage,
        "status": "PASS" if int(timing["exit_status"]) == 0 else "FAIL",
        "hostname": args.hostname,
        "partition": args.partition,
        "slurm_job_id": args.job_id,
        "run_id": os.environ.get("RUN_ID", "unset"),
        "source_git_commit": os.environ.get("SOURCE_GIT_COMMIT", "unset"),
        "source_inventory_sha256": os.environ.get("SOURCE_INVENTORY_SHA256", "unset"),
        "requested": {"cpus": cpus, "memory_gb": memory_gb, "gpus": gpus},
        "measured": {
            **timing,
            "total_cpu_seconds": timing["user_seconds"] + timing["system_seconds"],
            "cpu_efficiency": (timing["user_seconds"] + timing["system_seconds"]) / (timing["elapsed_seconds"] * cpus),
            "max_rss_gb": timing["max_rss_kb"] / (1024 ** 2),
            "memory_fraction_of_request": timing["max_rss_kb"] / (memory_gb * 1024 ** 2),
            "gpu": gpu,
        },
    }
    if args.benchmark_file is not None:
        if not args.benchmark_file.is_file():
            raise ResourceError("Benchmark result is missing")
        payload["benchmark"] = json.loads(args.benchmark_file.read_text(encoding="utf-8"))
    if args.hostname != "compute-0-2":
        payload["status"] = "FAIL"
        payload["failure"] = "scientific computation is restricted to compute-0-2"
    atomic_json(args.output, payload)
    return payload


def profile_key(payload):
    requested = payload["requested"]
    return requested["gpus"], requested["cpus"], requested["memory_gb"]


def select_profile(args):
    paths = [args.profile_dir / f"{kind}{cpu}.json" for kind in ("cpu", "gpu") for cpu in (8, 16, 32)]
    if any(not path.is_file() for path in paths):
        raise ResourceError("All six CPU/GPU benchmark profiles must exist")
    profiles = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(profiles) != 6 or any(profile.get("status") != "PASS" for profile in profiles):
        raise ResourceError("All six CPU/GPU benchmark profiles must pass")
    expected = {(gpus, cpu) for gpus in (0, 1) for cpu in (8, 16, 32)}
    contexts = {
        (
            profile.get("run_id"),
            profile.get("source_git_commit"),
            profile.get("source_inventory_sha256"),
        )
        for profile in profiles
    }
    if (
        {(profile["requested"]["gpus"], profile["requested"]["cpus"]) for profile in profiles} != expected
        or any(
            profile.get("hostname") != "compute-0-2"
            or profile.get("partition") != ("gpu" if profile["requested"]["gpus"] else "cpu")
            or profile["requested"].get("memory_gb") != 32
            or profile.get("benchmark", {}).get("status") != "PASS"
            for profile in profiles
        )
        or len(contexts) != 1
    ):
        raise ResourceError("Benchmark profile grid is incomplete")
    run_id, source_git_commit, source_inventory = contexts.pop()
    if (
        not isinstance(run_id, str) or not run_id or run_id == "unset"
        or not isinstance(source_git_commit, str) or len(source_git_commit) != 40
        or not isinstance(source_inventory, str) or len(source_inventory) != 64
        or run_id != args.run_id
        or source_git_commit != args.git_commit
        or source_inventory != args.source_inventory
    ):
        raise ResourceError("Benchmark profiles lack one valid source context")

    def smallest_within_five(candidates, kind):
        if kind == "cpu":
            candidates = [
                profile for profile in candidates
                if profile["benchmark"].get("active_cpu_efficiency", 0)
                > MIN_ACTIVE_CPU_EFFICIENCY
            ]
            if not candidates:
                raise ResourceError("No CPU profile exceeded 50% utilization during active compute")
        else:
            candidates = [
                profile for profile in candidates
                if profile["measured"]["gpu"].get("active_samples", 0)
                >= MIN_ACTIVE_GPU_SAMPLES
                and profile["measured"]["gpu"].get(
                    "mean_active_utilization_percent", 0
                ) > MIN_ACTIVE_GPU_UTILIZATION_PERCENT
            ]
            if not candidates:
                return None
        fastest = min(profile["measured"]["elapsed_seconds"] for profile in candidates)
        near = [profile for profile in candidates if profile["measured"]["elapsed_seconds"] <= fastest * 1.05]
        return min(near, key=profile_key)

    best_cpu = smallest_within_five(
        [profile for profile in profiles if profile["requested"]["gpus"] == 0],
        "cpu",
    )
    best_gpu = smallest_within_five(
        [profile for profile in profiles if profile["requested"]["gpus"] == 1],
        "gpu",
    )
    gpu_speedup = (
        best_cpu["measured"]["elapsed_seconds"]
        / best_gpu["measured"]["elapsed_seconds"]
        if best_gpu is not None else None
    )
    selected = best_gpu if best_gpu is not None and gpu_speedup > 1.05 else best_cpu
    peak = max(
        float(selected["measured"]["max_rss_gb"]),
        float(selected.get("benchmark", {}).get("estimated_full_peak_gb", 0)),
    )
    requested_memory = max(2, int(math.ceil((peak * 1.20) / 2) * 2))
    if requested_memory > MEMORY_CAP_GB:
        raise ResourceError("Measured/estimated memory plus 20% exceeds the approved 64 GB cap")
    fit_profile = {
        "cpus": selected["requested"]["cpus"],
        "memory_gb": requested_memory,
        "gpus": selected["requested"]["gpus"],
    }
    parallel_workers = 1
    if fit_profile["gpus"] == 0:
        parallel_workers = min(
            5,
            CPU_CAP // fit_profile["cpus"],
            MEMORY_CAP_GB // fit_profile["memory_gb"],
        )
    if parallel_workers < 1:
        raise ResourceError("Measured per-fit resources leave no valid model worker")
    parallel_candidates = 1
    if fit_profile["gpus"] == 0:
        parallel_candidates = min(
            2,
            CPU_CAP // (fit_profile["cpus"] * parallel_workers),
        )
    if parallel_candidates < 1:
        raise ResourceError("Measured fold workers leave no valid candidate worker")
    model_resources = {
        "cpus": fit_profile["cpus"] * parallel_workers * parallel_candidates,
        "memory_gb": fit_profile["memory_gb"] * parallel_workers,
        "gpus": fit_profile["gpus"],
    }
    payload = {
        "status": "PASS",
        "selection_rule": (
            "smallest profile within 5% of fastest; CPU active compute efficiency >50%; "
            "GPU mean active utilization >50% with >=3 active samples and >5% speedup; "
            "independent CPU folds run concurrently only while measured per-fit CPU and "
            "memory plus 20% remain within the 64 CPU/64 GB job caps"
        ),
        "selected": model_resources,
        "fit_profile": fit_profile,
        "parallel_workers": parallel_workers,
        "parallel_candidates": parallel_candidates,
        "selected_profile_job_id": selected["slurm_job_id"],
        "selected_profile_elapsed_seconds": selected["measured"]["elapsed_seconds"],
        "selected_profile_cpu_efficiency": selected["measured"]["cpu_efficiency"],
        "selected_profile_active_cpu_efficiency": selected["benchmark"].get(
            "active_cpu_efficiency"
        ),
        "selected_profile_active_gpu_utilization_percent": selected["measured"][
            "gpu"
        ].get("mean_active_utilization_percent"),
        "selected_profile_active_gpu_samples": selected["measured"]["gpu"].get(
            "active_samples", 0
        ),
        "best_cpu_elapsed_seconds": best_cpu["measured"]["elapsed_seconds"],
        "best_gpu_elapsed_seconds": (
            best_gpu["measured"]["elapsed_seconds"] if best_gpu is not None else None
        ),
        "gpu_speedup_over_best_cpu": gpu_speedup,
        "memory_basis_peak_gb": peak,
        "memory_margin": 0.20,
        "run_id": run_id,
        "source_git_commit": source_git_commit,
        "source_inventory_sha256": source_inventory,
        "profiles": profiles,
    }
    atomic_json(args.output, payload)
    return payload


def aggregate(args):
    stages = {}
    for stage in ("prepare", "model", "finalize"):
        path = args.resources_dir / f"{stage}.json"
        if not path.is_file():
            raise ResourceError(f"Missing resource evidence for {stage}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_partition = "gpu" if payload.get("requested", {}).get("gpus") else "cpu"
        if (
            payload.get("status") != "PASS"
            or payload.get("hostname") != "compute-0-2"
            or payload.get("partition") != expected_partition
        ):
            raise ResourceError(f"Invalid resource evidence for {stage}")
        stages[stage] = payload
    profile = json.loads(args.profile_selection.read_text(encoding="utf-8"))
    if profile.get("status") != "PASS":
        raise ResourceError("Resource profile selection did not pass")
    verify_selected_profile(argparse.Namespace(
        profile_dir=args.profile_selection.parent / "profiles",
        selection=args.profile_selection,
    ))
    selected = profile.get("selected", {})
    selected_profile_path = args.profile_selection.parent / "profiles" / "selected-model.json"
    selected_benchmark_path = (
        args.profile_selection.parent / "profiles" / "selected-model-benchmark.json"
    )
    if not selected_profile_path.is_file() or not selected_benchmark_path.is_file():
        raise ResourceError("Selected model concurrency evidence is missing")
    selected_validation = json.loads(selected_profile_path.read_text(encoding="utf-8"))
    selected_benchmark = json.loads(selected_benchmark_path.read_text(encoding="utf-8"))
    stage_context = next(iter(stages.values()))
    if (
        stages["prepare"]["requested"] != PREPARE_REQUEST
        or stages["model"]["requested"] != selected
        or stages["finalize"]["requested"] != FINALIZE_REQUEST
        or len({stage["run_id"] for stage in stages.values()}) != 1
        or len({stage["source_git_commit"] for stage in stages.values()}) != 1
        or len({stage["source_inventory_sha256"] for stage in stages.values()}) != 1
        or profile.get("run_id") != stage_context.get("run_id")
        or profile.get("source_git_commit") != stage_context.get("source_git_commit")
        or profile.get("source_inventory_sha256") != stage_context.get("source_inventory_sha256")
        or selected_validation.get("status") != "PASS"
        or selected_validation.get("requested") != selected
        or selected_validation.get("benchmark") != selected_benchmark
        or selected_validation.get("run_id") != stage_context.get("run_id")
        or selected_validation.get("source_git_commit")
        != stage_context.get("source_git_commit")
        or selected_validation.get("source_inventory_sha256")
        != stage_context.get("source_inventory_sha256")
    ):
        raise ResourceError("Stage resources do not match the selected profile and source context")
    payload = {
        "status": "PASS",
        "caps": {"maximum_cpus": CPU_CAP, "maximum_memory_gb": MEMORY_CAP_GB, "maximum_gpus": GPU_CAP},
        "profile_selection": profile,
        "selected_model_validation": selected_validation,
        "stages": stages,
        "no_artificial_memory_fill": True,
    }
    atomic_json(args.output, payload)
    return payload


def parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command")
    commands.required = True  # Python 3.6 on the CEDIA login node.
    stage = commands.add_parser("stage")
    stage.add_argument("--stage", required=True)
    stage.add_argument("--time-file", type=Path, required=True)
    stage.add_argument("--gpu-log", type=Path)
    stage.add_argument("--benchmark-file", type=Path)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--cpus", type=int, required=True)
    stage.add_argument("--memory-gb", type=int, required=True)
    stage.add_argument("--gpus", type=int, required=True)
    stage.add_argument("--hostname", required=True)
    stage.add_argument("--job-id", required=True)
    stage.add_argument("--partition", choices=("cpu", "gpu"), required=True)
    selection = commands.add_parser("select-profile")
    selection.add_argument("--profile-dir", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--run-id", required=True)
    selection.add_argument("--git-commit", required=True)
    selection.add_argument("--source-inventory", required=True)
    final = commands.add_parser("aggregate")
    final.add_argument("--resources-dir", type=Path, required=True)
    final.add_argument("--profile-selection", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-stage")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--stage", choices=("prepare", "model", "finalize"), required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--git-commit", required=True)
    verify.add_argument("--source-inventory", required=True)
    profile = commands.add_parser("verify-profile")
    profile.add_argument("--profile-dir", type=Path, required=True)
    profile.add_argument("--name", required=True)
    profile.add_argument("--cpus", type=int, required=True)
    profile.add_argument("--memory-gb", type=int, required=True)
    profile.add_argument("--gpus", type=int, required=True)
    profile.add_argument("--run-id", required=True)
    profile.add_argument("--git-commit", required=True)
    profile.add_argument("--source-inventory", required=True)
    selected_profile = commands.add_parser("verify-selected-profile")
    selected_profile.add_argument("--profile-dir", type=Path, required=True)
    selected_profile.add_argument("--selection", type=Path, required=True)
    return root


def main():
    args = parser().parse_args()
    try:
        commands = {
            "stage": record_stage,
            "select-profile": select_profile,
            "aggregate": aggregate,
            "verify-stage": verify_stage,
            "verify-profile": verify_profile,
            "verify-selected-profile": verify_selected_profile,
        }
        payload = commands[args.command](args)
    except ResourceError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
