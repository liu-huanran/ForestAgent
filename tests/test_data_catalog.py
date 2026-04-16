"""Tests for the real-sample data catalog."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from forestagent.data_catalog import DataCatalog


class DataCatalogTests(unittest.TestCase):
    def test_catalog_loads_records_and_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            ground_dir = data_dir / "ground_points"
            air_dir = data_dir / "air_points"
            (ground_dir / "150").mkdir(parents=True)
            (air_dir / "150").mkdir(parents=True)
            (ground_dir / "151").mkdir(parents=True)
            (air_dir / "151").mkdir(parents=True)

            (ground_dir / "150" / "150_64.las").write_text("", encoding="utf-8")
            (air_dir / "150" / "150_64.las").write_text("", encoding="utf-8")
            (ground_dir / "151" / "151_01.las").write_text("", encoding="utf-8")

            workbook_path = data_dir / "reference.xlsx"
            self._create_workbook(
                workbook_path,
                [
                    (150, "species_a", "150_64.las", 1, 1.0, 2.0, 3.0, 10.0, 8.0, 2.0, 4.0),
                    (151, "species_b", "151_01.las", 2, 4.0, 5.0, 6.0, 20.0, 12.0, 3.0, 5.0),
                ],
            )

            catalog = DataCatalog.from_data_dir(data_dir)

            self.assertEqual(catalog.summary()["record_count"], 2)
            self.assertEqual(catalog.summary()["dual_modality_count"], 1)

            record_a = catalog.get_record("150_64")
            self.assertEqual(record_a.available_modalities(), ["ground", "air"])
            self.assertEqual(record_a.measured.crown_width_mean_m, 3.0)

            record_b = catalog.get_record("151_01")
            self.assertEqual(record_b.available_modalities(), ["ground"])
            self.assertIsNone(record_b.air_point_cloud_path)

    def test_catalog_accepts_tls_and_uls_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            ground_dir = data_dir / "TLS"
            air_dir = data_dir / "ULS"
            (ground_dir / "150").mkdir(parents=True)
            (air_dir / "150").mkdir(parents=True)
            (ground_dir / "150" / "150_64.las").write_text("", encoding="utf-8")
            (air_dir / "150" / "150_64.las").write_text("", encoding="utf-8")

            workbook_path = data_dir / "reference.xlsx"
            self._create_workbook(
                workbook_path,
                [(150, "species", "150_64.las", 1, 1.0, 2.0, 3.0, 10.0, 8.0, 2.0, 4.0)],
            )

            catalog = DataCatalog.from_data_dir(data_dir)

            record = catalog.get_record("150_64")
            self.assertEqual(record.available_modalities(), ["ground", "air"])

    def test_catalog_ignores_unrecognized_top_level_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            ground_dir = data_dir / "TLS"
            air_dir = data_dir / "ULS"
            ignored_dir = data_dir / "extra_dataset"
            (ground_dir / "150").mkdir(parents=True)
            (air_dir / "150").mkdir(parents=True)
            ignored_dir.mkdir(parents=True)
            (ground_dir / "150" / "150_64.las").write_text("", encoding="utf-8")
            (air_dir / "150" / "150_64.las").write_text("", encoding="utf-8")
            (ignored_dir / "standalone.las").write_text("", encoding="utf-8")

            workbook_path = data_dir / "reference.xlsx"
            self._create_workbook(
                workbook_path,
                [(150, "species", "150_64.las", 1, 1.0, 2.0, 3.0, 10.0, 8.0, 2.0, 4.0)],
            )

            catalog = DataCatalog.from_data_dir(data_dir)

            record = catalog.get_record("150_64")
            self.assertEqual(record.available_modalities(), ["ground", "air"])

    @staticmethod
    def _create_workbook(path: Path, rows: list[tuple[object, ...]]) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet2"
        sheet.append(DataCatalog._EXPECTED_HEADERS)
        for row in rows:
            sheet.append(list(row))
        workbook.save(path)


if __name__ == "__main__":
    unittest.main()
