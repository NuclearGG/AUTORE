"""
FastAPI application for the AUTORE Environment.

Endpoints:
    POST /reset  - Reset the environment
    POST /step   - Execute an action
    GET  /state  - Get current environment state
    GET  /schema - Get action/observation schemas
    WS   /ws     - WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import AutoreAction, AutoreObservation
    from .AUTORE_environment import AutoreEnvironment
except (ImportError, ModuleNotFoundError):
    from models import AutoreAction, AutoreObservation
    from server.AUTORE_environment import AutoreEnvironment


app = create_app(
    AutoreEnvironment,
    AutoreAction,
    AutoreObservation,
    env_name="AUTORE",
    max_concurrent_envs=1,
)


def main():
    """Entry point — strictly callable with no arguments for openenv validate."""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args, _ = parser.parse_known_args()
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()