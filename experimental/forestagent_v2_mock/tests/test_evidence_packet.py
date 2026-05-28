"""Tests for ForestAgent v2 evidence packets."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from forestagent.evidence.packet import EvidenceItem, EvidencePacket


class EvidencePacketTests(unittest.TestCase):
    def test_measurement_evidence_serializes_to_json(self) -> None:
        packet = EvidencePacket(request_id="request-1")
        packet.add_item(
            EvidenceItem(
                evidence_id="measurement.dbh",
                source_tool="q1_dbh",
                evidence_type="measurement",
                value=23.4,
                unit="cm",
                status="ok",
                explanation="DBH estimated by frozen q1 geometry baseline.",
            )
        )

        payload = json.loads(packet.model_dump_json())

        self.assertEqual(payload["request_id"], "request-1")
        self.assertEqual(payload["items"][0]["evidence_id"], "measurement.dbh")
        self.assertEqual(payload["items"][0]["value"], 23.4)

    def test_unavailable_evidence_serializes_without_value(self) -> None:
        item = EvidenceItem(
            evidence_id="unavailable.q1_dbh",
            source_tool="q1_dbh",
            evidence_type="unavailable",
            value=None,
            unit=None,
            status="unavailable",
            explanation="Required input missing.",
        )

        self.assertIsNone(item.value)
        self.assertIsNone(item.unit)

    def test_invalid_status_and_type_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EvidenceItem(
                evidence_id="bad.status",
                source_tool="q1_dbh",
                evidence_type="measurement",
                status="pretend",
            )

        with self.assertRaises(ValidationError):
            EvidenceItem(
                evidence_id="bad.type",
                source_tool="q1_dbh",
                evidence_type="diagnosis",
                status="ok",
            )

    def test_failed_or_unavailable_evidence_cannot_carry_fake_values(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must not carry values"):
            EvidenceItem(
                evidence_id="unavailable.q2_height",
                source_tool="q2_height",
                evidence_type="unavailable",
                value=12.5,
                unit="m",
                status="unavailable",
                explanation="Missing point cloud.",
            )

    def test_packet_can_aggregate_multiple_tool_evidence(self) -> None:
        packet = EvidencePacket()
        packet.add_item(
            EvidenceItem(
                evidence_id="measurement.dbh",
                source_tool="q1_dbh",
                evidence_type="measurement",
                value=23.4,
                unit="cm",
                status="ok",
                explanation="Mock DBH evidence.",
            )
        )
        packet.add_item(
            EvidenceItem(
                evidence_id="measurement.height",
                source_tool="q2_height",
                evidence_type="measurement",
                value=12.5,
                unit="m",
                status="ok",
                explanation="Mock height evidence.",
            )
        )

        self.assertEqual(len(packet.ok_measurements()), 2)
        self.assertEqual(packet.get_item("measurement.height").source_tool, "q2_height")
        self.assertEqual(len(packet.by_source_tool("q1_dbh")), 1)


if __name__ == "__main__":
    unittest.main()
