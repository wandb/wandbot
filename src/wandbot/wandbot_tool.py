"Light‑weight public surface of heavy_lib."
__all__ = ["WANDBOT_TOOL_DESCRIPTION", "public_function"]


def public_function(x: int) -> int:
    "Cheap helper that does not need torch, numpy, …"
    return x + 1

WANDBOT_TOOL_DESCRIPTION = """Query the Weights & Biases support bot api for help with questions about the
Weights & Biases platform and how to use W&B Models and W&B Weave.

W&B features mentioned could include:
- Experiment tracking with Runs and Sweeps
- Model management with Models
- Model management and Data versioning with Artifacts and Registry
- Collaboration with Teams, Organizations and Reports
- Visualization with Tables and Charts
- Tracing and logging with Weave
- Evaluation and Scorers with Weave Evaluations
- Weave Datasets

Parameters
----------
question : str
    Users question about a Weights & Biases product or feature

Returns
-------
str
    newer to the user's question
"""