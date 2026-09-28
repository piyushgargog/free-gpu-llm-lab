# notebooks/

This repo's actual experiments were run via scripts (`experiments/*/`) in
Colab/Kaggle notebook cells, not as committed `.ipynb` files — the scripts
in this repo ARE the reproducible record.

If you export a Colab/Kaggle notebook for a future experiment, place it
here. A few notes:

- Strip cell outputs that could contain long model-generation text or
  anything sensitive before committing (Colab/Kaggle notebook JSON embeds
  outputs by default).
- Do not commit a notebook that contains an `HF_TOKEN` or any other
  credential in a cell or its output.
- Prefer keeping the actual logic in `experiments/<name>/*.py` and using
  the notebook as a thin driver that calls those scripts — that way the
  logic stays testable/reviewable outside the notebook format.

No notebooks currently exist in this repo.
