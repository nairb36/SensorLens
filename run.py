#!/usr/bin/env python3
import argparse

from sensorlens.app import create_app


def main():
    parser = argparse.ArgumentParser(description="SensorLens — 3D Multi-Object Tracking Visualizer")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"Starting SensorLens on http://{args.host}:{args.port}")

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
