"""
Run benchmark scripts from a create_pr_from_seed result on a GPU cluster node.

Reads test_scripts from the seed result JSON, creates a k8s ConfigMap with the
script code, submits a Job using the primus image, streams the logs, and prints
the results summary for pasting into the PR description.

Usage:
    uv run python pipeline/run_benchmarks.py \\
        --seed-result /tmp/minimax_dry_run.json \\
        --node smc300x-ccs-aus-a16-01 \\
        --kubeconfig /tmp/cluster.yaml

    # Or as a CLI entry point:
    run-benchmarks --seed-result /tmp/minimax_dry_run.json --node smc300x-ccs-aus-a16-01
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "k8s"
_JOB_TEMPLATE = _SCRIPT_DIR / "benchmark-job.yaml"


def _kubectl(
    args: list[str],
    kubeconfig: str,
    *,
    check: bool = True,
    capture: bool = False,
    control_node: str = "",
):
    """Run kubectl — either locally (with SSH tunnel) or via SSH on a control node.

    control_node: SSH destination like 'a15' or 'root@10.235.192.141'.
    When set, the manifest/command runs on that host which has direct k8s API access.
    """
    if control_node:
        # Build the kubectl command as a shell string and run it remotely via SSH
        kubectl_args = " ".join(f"'{a}'" for a in args)
        apply_flag = " --validate=false" if "apply" in args else ""
        remote_cmd = f"kubectl --request-timeout=60s {kubectl_args}{apply_flag}"
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", control_node, remote_cmd]
        if capture:
            return subprocess.run(ssh_cmd, capture_output=True, text=True, check=check)
        return subprocess.run(ssh_cmd, check=check)

    kubectl_bin = Path(__file__).resolve().parent.parent / "kubectl"
    if not kubectl_bin.exists():
        kubectl_bin = Path("kubectl")
    cmd = [str(kubectl_bin), f"--kubeconfig={kubeconfig}", "--request-timeout=60s"] + args
    if "apply" in args:
        cmd.append("--validate=false")
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, check=check)
    return subprocess.run(cmd, check=check)


def _safe_name(s: str) -> str:
    """Convert string to a k8s-safe name segment."""
    return re.sub(r"[^a-z0-9-]", "-", s.lower())[:40].strip("-")


def run_benchmarks(
    seed_result_path: str,
    node: str,
    kubeconfig: str = "/tmp/cluster.yaml",
    namespace: str = "default-default",
    timeout_minutes: int = 30,
    image: str = "rocm/pytorch:latest",
    control_node: str = "",
) -> dict:
    """
    Submit benchmark scripts from a seed result JSON as a k8s Job.

    Returns dict with keys: job_name, node, results (list of {script, output}).
    """
    with open(seed_result_path) as f:
        result = json.load(f)

    scripts = result.get("test_scripts", [])
    if not scripts:
        raise ValueError("No test_scripts found in seed result — run create-pr-from-seed first")

    branch = result.get("branch_name", "unknown")
    job_suffix = _safe_name(branch)[:20]
    job_name = f"bench-{job_suffix}-{int(time.time()) % 100000}"

    logger.info("Preparing %d benchmark script(s) for node %s", len(scripts), node)

    # Build ConfigMap data with script code
    cm_data: dict[str, str] = {}
    for s in scripts:
        name = s.get("name", f"script_{len(cm_data)}.py")
        if not name.endswith(".py"):
            name += ".py"
        cm_data[name] = s.get("code", "")

    if not any(cm_data.values()):
        raise ValueError("All test scripts have empty code — check suggest_tests output")

    # Create ConfigMap
    cm_name = f"pr-pundit-bench-scripts-{job_name}"
    cm_manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": cm_name, "namespace": namespace},
        "data": cm_data,
    }
    import tempfile, os
    kw = {"control_node": control_node}

    def _kubectl_apply_content(content: str, suffix: str) -> None:
        """Apply a manifest — SCP to control node if needed, else apply locally."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            local_file = f.name
        try:
            if control_node:
                remote_path = f"/tmp/pr-pundit-{os.path.basename(local_file)}"
                subprocess.run(
                    ["scp", "-o", "StrictHostKeyChecking=no", local_file, f"{control_node}:{remote_path}"],
                    check=True, capture_output=True,
                )
                subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", control_node,
                     f"kubectl apply -f {remote_path} -n {namespace} --validate=false && rm {remote_path}"],
                    check=True,
                )
            else:
                _kubectl(["apply", "-f", local_file, "-n", namespace], kubeconfig, **kw)
        finally:
            os.unlink(local_file)

    logger.info("Creating ConfigMap %s with %d scripts...", cm_name, len(cm_data))
    _kubectl_apply_content(json.dumps(cm_manifest), ".json")

    # Render job YAML
    template = _JOB_TEMPLATE.read_text()
    import re as _re
    job_yaml = template
    job_yaml = job_yaml.replace("${JOB_NAME}", job_name)
    job_yaml = job_yaml.replace("${NODE_NAME}", node)
    job_yaml = _re.sub(r"\$\{BENCH_IMAGE:-[^}]*\}", image, job_yaml)

    logger.info("Submitting Job pr-pundit-bench-%s...", job_name)
    _kubectl_apply_content(job_yaml, ".yaml")

    full_job_name = f"pr-pundit-bench-{job_name}"
    logger.info("Job submitted: %s — waiting for pod...", full_job_name)

    # Wait for pod to appear
    pod_name = None
    for _ in range(60):
        r = _kubectl(
            ["get", "pods", "-n", namespace, "-l", f"job-name={full_job_name}",
             "-o", "jsonpath={.items[0].metadata.name}"],
            kubeconfig, check=False, capture=True, **kw,
        )
        if r.stdout.strip():
            pod_name = r.stdout.strip()
            break
        time.sleep(5)

    if not pod_name:
        raise RuntimeError(f"Pod for job {full_job_name} never appeared")

    logger.info("Pod: %s — streaming logs...", pod_name)
    print(f"\n{'='*60}")
    print(f"Job: {full_job_name}")
    print(f"Pod: {pod_name}  Node: {node}")
    print(f"{'='*60}\n")

    # Wait for pod to be running
    for _ in range(120):
        r = _kubectl(
            ["get", "pod", pod_name, "-n", namespace,
             "-o", "jsonpath={.status.phase}"],
            kubeconfig, check=False, capture=True, **kw,
        )
        phase = r.stdout.strip()
        if phase in ("Running", "Succeeded", "Failed"):
            break
        logger.info("  Pod phase: %s — waiting...", phase)
        time.sleep(10)

    # Stream logs
    if control_node:
        log_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", control_node,
                   f"kubectl logs -n {namespace} -f {pod_name}"]
    else:
        kubectl_bin = Path(__file__).resolve().parent.parent / "kubectl"
        log_cmd = [str(kubectl_bin), f"--kubeconfig={kubeconfig}", "logs", "-n", namespace, "-f", pod_name]
    log_proc = subprocess.Popen(log_cmd, text=True)

    # Wait for job completion
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        r = _kubectl(
            ["get", "job", full_job_name, "-n", namespace,
             "-o", "jsonpath={.status.conditions[0].type}"],
            kubeconfig, check=False, capture=True, **kw,
        )
        status = r.stdout.strip()
        if status in ("Complete", "Failed"):
            break
        time.sleep(15)

    log_proc.terminate()

    # Get full logs
    r = _kubectl(
        ["logs", "-n", namespace, pod_name],
        kubeconfig, check=False, capture=True, **kw,
    )
    full_log = r.stdout

    # Parse per-script results from log
    script_results = []
    current_script = None
    current_lines: list[str] = []
    for line in full_log.splitlines():
        m = re.match(r"^RUNNING: (.+)$", line)
        if m:
            if current_script:
                script_results.append({"script": current_script, "output": "\n".join(current_lines)})
            current_script = m.group(1)
            current_lines = []
        elif current_script:
            current_lines.append(line)
    if current_script:
        script_results.append({"script": current_script, "output": "\n".join(current_lines)})

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    for sr in script_results:
        print(f"\n--- {sr['script']} ---")
        # Print last 20 lines of output (the summary table)
        lines = sr["output"].splitlines()
        print("\n".join(lines[-20:]) if len(lines) > 20 else sr["output"])

    return {
        "job_name": full_job_name,
        "pod_name": pod_name,
        "node": node,
        "scripts": list(cm_data.keys()),
        "results": script_results,
        "full_log": full_log,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run benchmark scripts from a seed result on a GPU k8s node")
    p.add_argument("--seed-result", required=True, help="Path to JSON from create-pr-from-seed --out")
    p.add_argument("--node", required=True, help="k8s node name to pin the job to (e.g. smc300x-ccs-aus-a16-01)")
    p.add_argument("--control-node", default="", dest="control_node",
                   help="SSH target of k8s control node to run kubectl on (e.g. a15). "
                        "Use this instead of --kubeconfig when an SSH tunnel is not available.")
    p.add_argument("--kubeconfig", default="/tmp/cluster.yaml", help="Path to kubeconfig file")
    p.add_argument("--namespace", default="default-default", help="k8s namespace")
    p.add_argument("--image", default="rocm/pytorch:latest",
                   help="Container image to use (must have ROCm + Python)")
    p.add_argument("--timeout", type=int, default=60, help="Max minutes to wait for job completion")
    p.add_argument("--out", default=None, help="Write full results JSON to this file")
    args = p.parse_args()

    result = run_benchmarks(
        args.seed_result,
        node=args.node,
        kubeconfig=args.kubeconfig,
        namespace=args.namespace,
        timeout_minutes=args.timeout,
        image=args.image,
        control_node=args.control_node,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
