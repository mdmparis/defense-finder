import os
import sys
import unittest
from contextlib import contextmanager
from io import StringIO


class FakeExitCodeException(Exception):
    pass


class TooledTest(unittest.TestCase):
    _tests_dir = os.path.normpath(os.path.dirname(__file__))
    _data_dir = os.path.join(_tests_dir, "data")
    _real_exit = None

    def setUp(self):
        self._real_exit = sys.exit
        sys.exit = self.fake_exit

    def tearDown(self):
        sys.exit = self._real_exit

    @staticmethod
    def fake_exit(*args, **kwargs):
        returncode = args[0]
        raise FakeExitCodeException(returncode)

    @contextmanager
    def catch_io(self, out=False, err=False):
        """
        Catch stderr and stdout of the code running within this block.
        """
        old_out = sys.stdout
        new_out = old_out
        old_err = sys.stderr
        new_err = old_err
        if out:
            new_out = StringIO()
        if err:
            new_err = StringIO()
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err
