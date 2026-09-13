# Viewer and HostAI integration

`serve.py` and `ui/` provide a local viewer for generated and shipped LS20 levels.
The viewer replays the level through the real engine, so it is useful for inspecting
game behavior and generated-level proof metadata. Its stored solution route is an
oracle replay, not a learned controller evaluation.

## Standalone viewer

```bash
uv run serve.py --port 11435
# open http://127.0.0.1:11435/ui/index.html
```

To load a trained policy into the same runtime:

```bash
uv run serve.py --port 11435 --checkpoint checkpoints/ls20-looped-policy.pt
```

The server is stateless. The browser owns the selected level and action history,
while each play request replays that history through the engine. `/health` reports
whether the server is reachable, and `/predict` remains available as the raw-input
alias of the inference route.

## HostAI runtime

The locked environment installs the HostAI SDK dependency. Start Pebby's server,
then start HostAI separately with its runtime URL pointed at Pebby:

```bash
HOSTAI_RUNTIME_URLS=http://127.0.0.1:11435
```

When embedded, the UI uses the injected `window.hostai` bridge. In standalone mode,
it sends the same request envelope to Pebby's local inference endpoint. The provider
declaration in `pebby/hostai_provider.py` is the source of truth for the current
operation and schema details; use the HostAI CLI's `describe` command rather than
duplicating that schema here.
