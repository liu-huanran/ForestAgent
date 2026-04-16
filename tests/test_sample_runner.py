"""Tests for running tasks against cataloged samples."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from forestagent.backends.mock_backend import MockBackend
from forestagent.data_catalog import DataCatalog
from forestagent.sample_runner import run_task_for_sample


class SampleRunnerTests(unittest.TestCase):
    def test_run_task_for_sample_returns_reference_and_task_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            ground_dir = data_dir / "ground_points"
            air_dir = data_dir / "air_points"
            (ground_dir / "150").mkdir(parents=True)
            (air_dir / "150").mkdir(parents=True)
            (ground_dir / "150" / "150_64.las").write_text("", encoding="utf-8")
            (air_dir / "150" / "150_64.las").write_text("", encoding="utf-8")

            workbook_path = data_dir / "reference.xlsx"
            self._create_workbook(
                workbook_path,
                [(150, "银杏", "150_64.las", 1, 1.0, 2.0, 3.0, 10.0, 8.0, 2.0, 4.0)],
            )

            catalog = DataCatalog.from_data_dir(data_dir)
            payload = run_task_for_sample(
                task_id="q1_dbh",
                sample_id="150_64",
                modality="ground",
                data_dir=data_dir,
                backend=MockBackend(),
                catalog=catalog,
            )

            self.assertEqual(payload["sample"]["sample_id"], "150_64")
            self.assertEqual(payload["sample"]["modality"], "ground")
            self.assertEqual(payload["measured_reference"]["dbh_cm"], 10.0)
            self.assertEqual(payload["task_result"]["status"], "success")
            self.assertEqual(payload["task_result"]["output_type"], "scalar")
            self.assertIn("28.40 cm", payload["answer_text"])

    def test_run_task_for_sample_raises_when_modality_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            ground_dir = data_dir / "ground_points"
            air_dir = data_dir / "air_points"
            (ground_dir / "151").mkdir(parents=True)
            air_dir.mkdir(parents=True)
            (ground_dir / "151" / "151_01.las").write_text("", encoding="utf-8")

            workbook_path = data_dir / "reference.xlsx"
            self._create_workbook(
                workbook_path,
                [(151, "银杏", "151_01.las", 2, 4.0, 5.0, 6.0, 20.0, 12.0, 3.0, 5.0)],
            )

            with self.assertRaises(FileNotFoundError):
                run_task_for_sample(
                    task_id="q1_dbh",
                    sample_id="151_01",
                    modality="air",
                    data_dir=data_dir,
                    backend=MockBackend(),
                )

    @staticmethod
    def _create_workbook(path: Path, rows: list[tuple[object, ...]]) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet2"
        sheet.append(
            [
                "样地号",
                "树种",
                "对应的文件名",
                "编号",
                "X",
                "Y",
                "Z",
                "胸径cm",
                "树高m",
                "东西冠幅m",
                "南北冠幅m",
            ]
        )
        for row in rows:
            sheet.append(list(row))
        workbook.save(path)


if __name__ == "__main__":
    unittest.main()

