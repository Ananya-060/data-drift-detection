import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Any


logger = logging.getLogger(__name__)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def _alert_message(run: Any) -> str:
    severe = sum(
        result.get("severity") == "severe"
        for result in run.results.values()
    )
    return (
        f"Data drift detected for {run.source}: "
        f"{run.drifted_feature_count} of {run.feature_count} features crossed "
        f"the drift threshold ({severe} severe). Run ID: {run.id}."
    )


def _send_gmail(username: str, app_password: str, recipient: str, message: str) -> None:
    email = EmailMessage()
    email["Subject"] = "Data drift alert"
    email["From"] = username
    email["To"] = recipient
    email.set_content(message)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
        server.starttls()
        server.login(username, app_password)
        server.send_message(email)


def notify_drift(run: Any) -> dict[str, str]:
    if not run.drifted_feature_count:
        return {"gmail": "not_triggered"}

    message = _alert_message(run)
    username = os.getenv("GMAIL_SMTP_USERNAME", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    recipient = os.getenv("GMAIL_TO", "").strip()
    if username and app_password and recipient:
        try:
            _send_gmail(username, app_password, recipient, message)
            return {"gmail": "sent"}
        except (OSError, smtplib.SMTPException, ValueError) as exc:
            logger.warning("Gmail drift notification failed (%s)", type(exc).__name__)
            return {"gmail": "failed"}
    return {"gmail": "not_configured"}
