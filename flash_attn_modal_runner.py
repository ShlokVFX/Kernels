import modal
app = modal.App("gluon-flash-attn-bench")

image = (
    # Match the CUDA toolkit version to torch cu128 so flash-attn compiles cleanly.
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "cmake", "ninja-build", "zlib1g-dev")
    .pip_install("numpy", "pandas", "pytest")
    .pip_install(
        "torch",
        extra_options="--pre --index-url https://download.pytorch.org/whl/nightly/cu128",
    )
    .pip_install("pybind11", "wheel", "setuptools")
    .run_commands(
        "pip install --no-build-isolation 'git+https://github.com/triton-lang/triton.git'"
    )
    .add_local_dir("triton/python/tutorials/gluon", "/tutorials")
    .add_local_python_source("flash_attn_gluon")
)

@app.function(image=image, gpu="B200", timeout=7200)
def run_triton():
    from importlib import import_module
    m = import_module("flash_attn_gluon")
    m.run_triton_impl()

@app.local_entrypoint()
def main():
    run_triton.remote()
