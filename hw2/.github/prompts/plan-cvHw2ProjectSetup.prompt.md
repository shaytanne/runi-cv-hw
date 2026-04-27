# Plan: CV HW2 Project Setup

## Goal
Set up uv-managed Python environment + Git/GitHub for a Jupyter notebook CV assignment.

## Decisions
- Python 3.11 (pinned)
- Private GitHub repo
- Images committed directly (no LFS)
- No README (not requested)
- `uv.lock` committed (reproducibility)
- No `gh` CLI — manual GitHub push

## Files to create
1. `pyproject.toml` — uv project with all deps
2. `.python-version` — "3.11" (uv uses this to pin python)
3. `.gitignore` — exclude .venv, .ipynb_checkpoints, __pycache__, uv cache

## Dependencies (from notebook imports)
- numpy (>=1.3.2)
- opencv-python
- matplotlib
- scipy
- jupyter (to run the notebook)
- ipykernel (so venv kernel is available in Jupyter)

## Steps
### Phase 1 – uv project
1. Create `pyproject.toml` with [project] metadata and dependencies
2. Create `.python-version` with "3.11"
3. Run `uv sync` to create `.venv` and lock file

### Phase 2 – Git setup
4. Create `.gitignore`
5. Run `git init`, `git add .`, `git commit -m "Initial commit"`
6. Manual GitHub steps:
   - Create empty private repo on github.com (no README, no license)
   - `git remote add origin <your-repo-url>`
   - `git push -u origin main`

### Phase 3 – Jupyter kernel
7. Register venv as Jupyter kernel so notebook uses it
   - `uv run python -m ipykernel install --user --name hw2`

## Verification
- Select `hw2` kernel in the notebook
- Run the first cell — cv2, matplotlib, numpy, scipy all import successfully
- Image loads: `images/Sudoku.PNG` prints shape without error
