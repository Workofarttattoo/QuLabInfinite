"""Backward-compatible wrapper around the canonical QuLabInfinite runtime entrypoint."""

from qulab_runtime import app, build_registry, cli, registry

__all__ = ["app", "build_registry", "cli", "registry"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8102)
