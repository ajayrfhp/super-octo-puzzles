# Data Science Research Codebase Organization Notes

## Core Philosophy: Progressive Refactoring
Do not force rigid structure at the start. Allow for initial chaos in notebooks, but implement a system to periodically refactor code into a stable structure. **Engineering supports research velocity; it does not replace it.**

## I. Project Structure & Hygiene
### 1. The "Cookiecutter" Standard
Adopt a standardized folder structure to eliminate decision fatigue.
* **`data/`**: `raw/` (immutable), `processed/` (clean), `interim/` (checkpoints).
* **`notebooks/`**: For experimentation.
* **`src/`**: Source code module (`__init__.py`, `data/`, `models/`).
* **`results/`**: Weights, metrics, figures.

### 2. Notebook Protocols
* **Numbering:** Use sequential prefixes (e.g., `01_cleaning.ipynb`, `02_analysis.ipynb`) to define order.
* **The "Sandpit":** Move failed/dead-end notebooks to an `archive/` or `scratch/` subfolder immediately.
* **Metadata Header:** Every notebook must state: Goal, Input Data, Output Artifacts.

### 3. The "Rule of Three" (Refactoring)
* **Workflow:** Write function in notebook -> Use it -> If used a second time, move to `src/`.
* **Autoreload:** Use the magic command to edit `.py` files without restarting kernels:
    ```python
    %load_ext autoreload
    %autoreload 2
    ```

## II. Pipeline Management
### 1. Externalize "Order of Operations"
Stop relying on memory to know which script runs first.
* **Level 1: Makefile (Recommended):** A simple text file defining commands (e.g., `make clean`, `make train`). Acts as documentation and execution tool.
* **Level 2: Controller Notebook:** A master notebook that uses `%run ./notebooks/01_task.ipynb` to trigger steps sequentially.
* **Level 3: DVC (Data Version Control):** For complex dependencies and caching.

### 2. The Checkpoint Strategy
Never overwrite data files. Save outputs of distinct stages to distinct folders.
* `01_raw` -> `02_intermediate` -> `03_features` (tensorized) -> `04_output`

## III. Handling Custom Datasets
### 1. The "Intermediate Format" Strategy
Avoid complex DataLoaders that parse raw files (PDF/JSON/TXT) on the fly.
* **Action:** Write a script *once* to convert raw data into **Hugging Face Datasets (Arrow)** or **Parquet**.
* **Benefit:** Fast loading (memory mapping), easier debugging, lazy loading.

### 2. Offline vs. Online Processing
* **Offline:** Heavy, deterministic tasks (resizing, tokenizing). Do this once before training.
* **Online:** Stochastic tasks (random masking, augmentations). Do this during training.

### 3. The "Golden Sample"
Create a debug dataset containing 5–100 perfectly verified examples. Build the entire pipeline against this small set first to ensure end-to-end functionality before using the full dataset.

---

# Q&A: Common Frustrations & Solutions

**Q: My codebase is full of unorganized, unrelated notebooks. How do I fix this without slowing down?**
**A:** Stop treating notebooks as final products. Treat them as lab journals. Use the **Rule of Three**: as soon as a block of code is useful in more than one place, refactor it into a Python script in a `src/` folder. Use `%autoreload` so you can keep working in the notebook while calling functions from the script.

**Q: I lose track of what I did in previous sessions. How do I remember my experiments?**
**A:** Stop tracking results in your head or notebook outputs. Implement a simple **Experiment Log** (even just a CSV file) that records the timestamp, commit hash, hyperparameters, and resulting metrics for every run.

**Q: I have long pipelines (Download -> Clean -> Caption). I forget which script to run first.**
**A:** Create a **Makefile**. This allows you to define dependencies (e.g., "cleaning depends on download"). You can then run the whole pipeline with a single command like `make all`, and the file serves as documentation for the correct order of operations.

**Q: Custom dataset processing takes too much time and feels like it delays "real results."**
**A:** Reframe your mindset: with custom data, the pipeline *is* the result. To move faster, convert your raw data immediately into a binary format like **Parquet or Arrow**. This removes the need to write complex loading logic during the modeling phase and speeds up training significantly.

**Q: My training crashes halfway through because of bad data. How do I prevent this?**
**A:** Use **Pydantic** or similar validation libraries during the data ingestion phase to "fail fast" if data is missing or malformed. Additionally, always build your pipeline using a **"Golden Sample"** (a tiny, verified subset of data) to debug logic errors instantly before running on the full dataset.
