"""Central ONNX Runtime session factory (torch-free, reproducible).

Every runtime session is created here so the execution policy has a
single source of truth: two intra-op threads, one inter-op thread and
sequential execution. The fixed policy keeps CPU inference reproducible
across machines and prevents ORT from oversubscribing cores.

Build-time exporters keep their own helpers in ``scripts/`` (torch
allowed there); this module is runtime-only.
"""
import os


def session_options(*, lower_memory=False, basic_optimization=False):
    """Return ORT SessionOptions with the project-wide execution policy."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR only: hide benign optimizer chatter
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = not lower_memory
    so.enable_mem_pattern = not lower_memory
    if basic_optimization:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return so


def create_session(path, *, lower_memory=False, basic_optimization=False):
    """Open one CPU ONNX session with the shared deterministic policy."""
    import onnxruntime as ort

    return ort.InferenceSession(
        os.path.abspath(path),
        sess_options=session_options(lower_memory=lower_memory,
                                     basic_optimization=basic_optimization),
        providers=["CPUExecutionProvider"],
    )
