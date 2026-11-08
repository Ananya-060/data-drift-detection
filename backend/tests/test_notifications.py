import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.notifications import notify_drift


class DriftNotificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = SimpleNamespace(
            id=12,
            source="lendingclub_2015-Q4",
            drifted_feature_count=2,
            feature_count=20,
            results={"a": {"severity": "severe"}, "b": {"severity": "moderate"}},
        )

    def test_does_not_send_when_drift_was_not_detected(self) -> None:
        self.run.drifted_feature_count = 0
        with patch("app.notifications._send_gmail") as send_gmail:
            delivery = notify_drift(self.run)

        self.assertEqual(delivery, {"gmail": "not_triggered"})
        send_gmail.assert_not_called()

    def test_reports_gmail_as_unconfigured(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GMAIL_SMTP_USERNAME": "",
                "GMAIL_APP_PASSWORD": "",
                "GMAIL_TO": "",
            },
        ):
            self.assertEqual(notify_drift(self.run), {"gmail": "not_configured"})

    def test_sends_gmail_alert(self) -> None:
        environment = {
            "GMAIL_SMTP_USERNAME": "sender@example.com",
            "GMAIL_APP_PASSWORD": "test-password",
            "GMAIL_TO": "recipient@example.com",
        }
        with patch.dict(os.environ, environment), patch(
            "app.notifications._send_gmail"
        ) as send_gmail:
            delivery = notify_drift(self.run)

        self.assertEqual(delivery, {"gmail": "sent"})
        send_gmail.assert_called_once()


if __name__ == "__main__":
    unittest.main()
