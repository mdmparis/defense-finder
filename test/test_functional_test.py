import os

from defense_finder_cli.main import run
from test import TooledTest, FakeExitCodeException


class Test(TooledTest):
    def test_data_content(self):
        subdir = "tests/data/nt/"
        input_file = "df_test_nt.fna"
        expected_results_dir = "expected_results/"

        with self.assertRaises(FakeExitCodeException) as ctx:
            run(
                [
                    subdir + input_file,
                    "-o",
                    subdir,
                ]
            )
        self.assertEqual(str(ctx.exception), "0")  # check the returncode

        for file in os.listdir(os.path.join(subdir, expected_results_dir)):
            with open(
                os.path.join(subdir, expected_results_dir, file), "r"
            ) as expected, open(os.path.join(subdir, file), "r") as produced:
                expected_lines = set(expected.readlines())
                produced_lines = set(produced.readlines())
                self.assertSetEqual(expected_lines, produced_lines, file)
