#!/usr/bin/env python3
"""Draw measured PID responses from a saved controller recording."""

import argparse

from pid_lib.plotting import plot_pid_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    try:
        directory = plot_pid_response(args.run_directory)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"PID response plots saved in {directory}")


if __name__ == "__main__":
    main()
