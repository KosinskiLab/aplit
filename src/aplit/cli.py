#!/usr/bin/env python3
"""
Command-line interface for APLit
"""

import argparse
import os
import secrets
import subprocess
import sys
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
        help="Force opening a browser (disable headless mode).",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Force headless mode (never open a browser).",
    )

    parser.add_argument(
        "--site-config",
        type=str,
        default="",
        help="Path to the APLit submission site YAML configuration.",
    )

    parser.add_argument(
        "--access-token",
        type=str,
        default="",
        help="Prototype access token; generated automatically when the site requires one.",
    )

    args = parser.parse_args()

    if args.browser and args.no_browser:
        print("Error: use only one of --browser / --no-browser", file=sys.stderr)
        sys.exit(2)

    if not (1 <= args.port <= 65535):
        print(f"Error: invalid port {args.port}", file=sys.stderr)
        sys.exit(2)

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

    site_config_path = None
    access_token = args.access_token
    if args.site_config:
        site_config_path = Path(args.site_config).expanduser().resolve()
        if not site_config_path.is_file():
            print(
                f"Error: site configuration '{site_config_path}' does not exist",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            from aplit.submission.errors import SubmissionError
            from aplit.submission.models import SiteConfig

            site = SiteConfig.load(site_config_path)
        except (SubmissionError, TypeError, ValueError) as exc:
            print(f"Error: invalid site configuration: {exc}", file=sys.stderr)
            sys.exit(2)
        if site.access_token_required and not access_token:
            access_token = secrets.token_urlsafe(32)

    # Prepare environment
    env = os.environ.copy()
    if args.directory:
        env["APLIT_DEFAULT_DIRECTORY"] = str(Path(args.directory).resolve())
    if site_config_path:
        env["APLIT_SITE_CONFIG"] = str(site_config_path)
    if access_token:
        env["APLIT_ACCESS_TOKEN"] = access_token

    # Print startup message
    print("=" * 70)
    print("APLit - AlphaPulldown Structure Viewer")
    print("=" * 70)
    if args.directory:
        print(f"Directory: {args.directory}")
    print(f"Server: http://{args.server_address}:{args.port}")
    if site_config_path:
        print(f"Site config: {site_config_path}")
    if access_token:
        print(
            f"Authenticated URL: http://{args.server_address}:{args.port}/?token={access_token}"
        )
    print("=" * 70)
    print("\nStarting server... Press Ctrl+C to stop")
    print()

    # Run streamlit
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"\nError running APLit: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
