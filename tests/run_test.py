import os
import sys
import unittest
from argparse import ArgumentParser


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "paths",
        nargs='*',
        help="paths to tests",
        default=False,
    )
    args = parser.parse_args(args or sys.argv[1:])

    root_dir = os.path.dirname(__file__)
    if args.paths:
        suite = unittest.TestSuite()
        for test_path in args.paths:
            full_path = os.path.join(root_dir, test_path)
            suite.addTests(unittest.TestLoader().discover(os.path.dirname(full_path), pattern=str(test_path)))
    else:
        suite = unittest.TestLoader().discover(root_dir)

    # patch path to have sources in it
    try:
        sys_path = sys.path
        sys.path.insert(0, os.path.join(__file__, '..', '..'))
        test_runner = unittest.TextTestRunner().run(suite)
        return test_runner.wasSuccessful()
    finally:
        sys.path = sys_path


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1:]) else 1)
