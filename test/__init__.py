import os
import sys
import unittest


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
