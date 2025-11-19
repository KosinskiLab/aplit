"""
Command-line interface for APLit
"""

import sys
import os
import argparse
import subprocess

from pathlib import Path


def main():
    """Launch the AP streamlit application"""
    parser = argparse.ArgumentParser(
        description="APLit - AlphaPulldown Structure Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aplit
  aplit --directory /path/to/predictions
  aplit --directory /path/to/predictions --port 8502

For more information, visit: https://github.com/KosinskiLab/aplit
        """,
    )

    parser.add_argument(
        "--directory",
        type=str,
        default="",
        help="Path to directory containing AlphaPulldown predictions",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the server on (default: 8501)",
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost",
        help="Server address (default: localhost)",
    )

    parser.add_argument(
        "--browser",
        action="store_true",
        help="Automatically open browser (default: True for local, False for remote)",
    )

    parser.add_argument(
        "--no-browser", action="store_true", help="Do not automatically open browser"
    )

    args = parser.parse_args()

    # Get the path to app.py
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Build streamlit command
    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.server_address,
    ]

    # Handle browser settings
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    elif args.browser:
        cmd.extend(["--server.headless", "false"])

    # Add directory if provided
    if args.directory:
        if not Path(args.directory).exists():
            print(
                f"Error: Directory '{args.directory}' does not exist", file=sys.stderr
            )
            sys.exit(1)
        cmd.extend(["--", "--directory", args.directory])

    # Prepare environment
    env = os.environ.copy()
    if args.directory:
        env["APLIT_DEFAULT_DIRECTORY"] = str(Path(args.directory).resolve())

    # Print startup message
    print("=" * 70)
    print("APLit - AlphaPulldown Structure Viewer")
    print("=" * 70)
    if args.directory:
        print(f"Directory: {args.directory}")
    print(f"Server: http://{args.server_address}:{args.port}")
    print("=" * 70)
    print("\nStarting server... Press Ctrl+C to stop")
    print()

    # Run streamlit
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running APLit: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
