import modal
import subprocess
import sys


app = modal.App(name="flash-attn")


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(["nvidia-cutlass-dsl==4.3.0.dev0", "torch", "einops"])
    .add_local_dir("./flash_attn/cute", remote_path="/root/flash_attn/cute", copy=True)
    .pip_install_from_pyproject("./flash_attn/cute/pyproject.toml")
)


@app.function(
    gpu="B200",
    timeout=3600,
    image=image.add_local_file("bench.py", remote_path="/root/bench.py"),
)
def run(
    page_size: int = 128,
    num_ptr_calculations: int = None,
    gmem_threads_per_row: int = None,
    ablate_page_table_load: bool = False,
    return_time: bool = False,
):
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    if num_ptr_calculations is not None:
        os.environ["OVERRIDE_NUM_PTR_CALCULATIONS"] = str(num_ptr_calculations)
    if gmem_threads_per_row is not None:
        os.environ["OVERRIDE_GMEM_THREADS_PER_ROW"] = str(gmem_threads_per_row)
    os.environ["ABLATE_PAGE_TABLE_LOAD"] = "1" if ablate_page_table_load else "0"

    command = [
        sys.executable, "bench.py",
        "--page_size", str(page_size),
    ]

    print("Command:", " ".join(command))

    import re

    if return_time:
        result = subprocess.run(command, stdout=subprocess.PIPE, encoding="utf-8")
        match = re.search(r"^\s*Avg time:\s*([0-9.eE+-]+)\s*ms", result.stdout, re.MULTILINE)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not find 'Avg time' in output: {result.stdout}")
    else:
        subprocess.run(command)

@app.local_entrypoint()
def main():
    tasks = []
    for gmem_threads_per_row in [1, 2, 4, 8]:
        num_ptr_calculations = gmem_threads_per_row * 4
        while num_ptr_calculations <= 256:
            tasks.append((1, num_ptr_calculations, gmem_threads_per_row, False, True))
            num_ptr_calculations *= 2

    avg_times = list(run.starmap(tasks))

    print("page_size,num_ptr_calculations,gmem_threads_per_row,avg_time")
    for (page_size, num_ptr_calculations, gmem_threads_per_row, _, _), avg_time in zip(tasks, avg_times):
        print(f"{page_size},{num_ptr_calculations},{gmem_threads_per_row},{avg_time}")