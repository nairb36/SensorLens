#!/usr/bin/env python3
import argparse

from sensorlens.app import create_app


def main():
    parser = argparse.ArgumentParser(description="SensorLens — 3D Multi-Object Tracking Visualizer")
    parser.add_argument("--dataroot", default="/workspace/data/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
    parser.add_argument("--gt", default=None, help="Path to GT detections JSON")
    parser.add_argument("--tracker", default=None, help="Path to tracker results JSON")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    if not args.gt and not args.tracker:
        parser.error("At least one of --gt or --tracker must be provided")

    print(f"Starting SensorLens on http://{args.host}:{args.port}")
    print(f"  dataroot: {args.dataroot}")
    print(f"  version:  {args.version}")
    if args.gt:
        print(f"  GT file:  {args.gt}")
    if args.tracker:
        print(f"  Tracker:  {args.tracker}")

    app = create_app(
        dataroot=args.dataroot,
        version=args.version,
        gt_path=args.gt,
        tracker_path=args.tracker,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
