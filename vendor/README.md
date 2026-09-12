# Model Microscope library

**This wheel is not the active install source.** Pebby resolves `model-microscope` from the
sibling checkout on the filesystem, so `uv sync` links the live source and UI edits appear
without a rebuild:

```toml
[tool.uv.sources]
model-microscope = { path = "../pytorch_visualizer", editable = true }
```

`model_microscope-0.1.3-py3-none-any.whl` is a current build of that checkout, kept here as a
standalone fallback for when the sibling checkout is not available. It contains the model-map
interface (`static/mapview.js`).

To switch Pebby onto the wheel instead of the checkout:

```bash
uv add ./vendor/model_microscope-0.1.3-py3-none-any.whl
```

To refresh this wheel after changing the visualizer:

```bash
# in the visualizer checkout: bump the version in pyproject.toml first
uv build --wheel
rm ../Pebby/vendor/model_microscope-<old-version>-py3-none-any.whl
cp dist/model_microscope-<new-version>-py3-none-any.whl ../Pebby/vendor/
```

The lockfile verifies the wheel hash; do not replace an existing version's wheel in place.
Always publish a newly versioned artifact.
