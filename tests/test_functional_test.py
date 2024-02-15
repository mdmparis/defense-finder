import csv
import os
from tempfile import TemporaryDirectory

from defense_finder_cli.main import run
from tests import TooledTest, FakeExitCodeException


def cleaned_lines(file, filename="", col_names_to_del=None):
    lines = set()
    reader = csv.reader(file, delimiter='\t')
    col_ids_to_del = []
    header = None
    col_names_to_del = col_names_to_del or [
        'sys_id',
        'used_in',
    ]
    for row in reader:
        if header is None:
            header = row
            for col_name in col_names_to_del:
                try:
                    col_ids_to_del.append(header.index(col_name))
                except ValueError:
                    pass

            col_ids_to_del = list(reversed(sorted(col_ids_to_del)))
            assert filename
        else:
            for col_id in col_ids_to_del:
                del row[col_id]
            lines.add('\t'.join(row))
    return sorted(lines)


class TestCleanedLine(TooledTest):

    def test_sorted(self):
        with TemporaryDirectory() as tempdir:
            tmp_file_path = os.path.join(tempdir, 'file.csv')
            with open(tmp_file_path, 'w') as f:
                f_csv = csv.writer(f, delimiter='\t')
                f_csv.writerow(["header1", "header2", "header3", "header4"])
                f_csv.writerow(l_i1 := ["a", "z", "e", "y"])
                f_csv.writerow(l_i2 := ["b", "a", "t", "56"])
                f_csv.writerow(l_i3 := ["a", "c", "u", "op"])
            with open(tmp_file_path, 'r') as f:
                lines = list(cleaned_lines(f, tmp_file_path))
                self.assertEqual(len(lines), 3)
                self.assertEqual(lines[0], '\t'.join(l_i3))
                self.assertEqual(lines[1], '\t'.join(l_i1))
                self.assertEqual(lines[2], '\t'.join(l_i2))

    def test_col_removed_1(self):
        with TemporaryDirectory() as tempdir:
            tmp_file_path = os.path.join(tempdir, 'file.csv')
            with open(tmp_file_path, 'w') as f:
                f_csv = csv.writer(f, delimiter='\t')
                f_csv.writerow(["header1", "header2", "header3", "header4"])
                f_csv.writerow(["a", "z", "e", "y"])
                f_csv.writerow(["b", "a", "t", "56"])
                f_csv.writerow(["a", "c", "u", "op"])
            with open(tmp_file_path, 'r') as f:
                lines = list(
                    cleaned_lines(
                        f,
                        tmp_file_path,
                        col_names_to_del=[
                            "header2",
                        ],
                    )
                )
                self.assertEqual(len(lines), 3)
                self.assertEqual(lines[0], "a\te\ty")
                self.assertEqual(lines[1], "a\tu\top")
                self.assertEqual(lines[2], "b\tt\t56")

    def test_col_removed_2(self):
        with TemporaryDirectory() as tempdir:
            tmp_file_path = os.path.join(tempdir, 'file.csv')
            with open(tmp_file_path, 'w') as f:
                f_csv = csv.writer(f, delimiter='\t')
                f_csv.writerow(["header1", "header2", "header3", "header4"])
                f_csv.writerow(["a", "z", "e", "y"])
                f_csv.writerow(["b", "a", "t", "56"])
                f_csv.writerow(["a", "c", "u", "op"])
            with open(tmp_file_path, 'r') as f:
                lines = list(
                    cleaned_lines(
                        f,
                        tmp_file_path,
                        col_names_to_del=[
                            "header2",
                            "header4",
                            "header666",
                        ],
                    )
                )
                self.assertEqual(len(lines), 3)
                self.assertEqual(lines[0], 'a\te')
                self.assertEqual(lines[1], 'a\tu')
                self.assertEqual(lines[2], 'b\tt')


class Test(TooledTest):
    data_dir = "tests/data"
    exp_dir = "expected_results"
    prd_dir = "produced_results"
    maxDiff = None

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
                with self.subTest(file=file), open(
                    os.path.join(subdir, self.exp_dir, file),
                    "r",
                ) as expected, open(
                    os.path.join(os.path.join(subdir, self.prd_dir), file),
                    "r",
                ) as produced:
                    expected_lines = cleaned_lines(expected, filename=file)
                    produced_lines = cleaned_lines(produced, filename=file)
                    self.assertEqual(expected_lines, produced_lines, file)
        except AssertionError as ae:
            print(out.getvalue().strip(), err.getvalue().strip())
            raise ae
