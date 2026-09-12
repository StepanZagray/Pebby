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

## HostAI provider SDK

`hostai-0.1.0-py3-none-any.whl` is the active HostAI dependency, built from the
HostAI repository's `python/` package. `pyproject.toml` selects this relative
wheel path and `uv.lock` records its hash. `uv sync --locked` therefore works
without another local checkout or a published PyPI package. No model weights
are included.

This is the initial, unpublished SDK build. For a subsequent release, bump its
version, build a new wheel with `uv build --project python --wheel --no-sources`,
add it here, update `vendor/.gitignore` to include that artifact, then update
the dependency/source and run `uv lock`. Keep the SDK source repository as the
source of truth; never patch files inside an installed wheel.
