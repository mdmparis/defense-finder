import os

from defense_finder_cli.main import run
from tests import TooledTest, FakeExitCodeException


def cleaned_lines(expected):
    return set(sorted([line.upper() for line in expected.readlines()]))


class Test(TooledTest):
    data_dir = "tests/data"
    exp_dir = "expected_results"
    prd_dir = "produced_results"

    def test_data_content(self):
        data_dirs = []
        for subdir in os.listdir(self.data_dir):
            # get all subdir of data
            subdir = os.path.join(self.data_dir, subdir)
            # but only if it contains expected_dir
            if not os.path.exists(os.path.join(subdir, self.exp_dir)):
                continue
            for input_file in os.listdir(subdir):
                # all all fna and faa files, append them
                if input_file.endswith(".fna") or input_file.endswith(".faa"):
                    data_dirs.append((subdir, input_file))
        # one subtest per file
        for subdir, input_file in data_dirs:
            with self.subTest(subdir=subdir):
                self._test_subdir(subdir=subdir, input_file=input_file)

    def _test_subdir(self, subdir, input_file):
        with self.catch_io(out=True, err=True) as (out, err):
            with self.assertRaises(FakeExitCodeException) as ctx:
                run(
                    [
                        os.path.join(subdir, input_file),
                        "-o",
                        os.path.join(subdir, self.prd_dir),
                    ]
                )
        self.assertEqual(str(ctx.exception), "0")  # check the returncode

        try:
            for file in os.listdir(os.path.join(subdir, self.exp_dir)):
                with open(
                    os.path.join(subdir, self.exp_dir, file), "r"
                ) as expected, open(
                    os.path.join(os.path.join(subdir, self.prd_dir), file), "r"
                ) as produced:
                    expected_lines = cleaned_lines(expected)
                    produced_lines = cleaned_lines(produced)
                    self.assertSetEqual(expected_lines, produced_lines, file)
        except AssertionError as ae:
            print(out.getvalue().strip(), err.getvalue().strip())
            raise ae
