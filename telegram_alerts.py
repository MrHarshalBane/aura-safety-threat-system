import os
import requests
import threading
from typing import Tuple, Optional

def safe_log(msg: str):
    """Safely log debug and alert messages."""
    print(msg, flush=True)

def get_telegram_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve Telegram Bot Token and Chat ID from environment variables."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    return (token if token else None, chat_id if chat_id else None)

def send_telegram_message_sync(message: str, bot_token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """Sends a synchronous Telegram markdown-formatted message."""
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    target_chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    
    if not token or not target_chat:
        safe_log("[TELEGRAM] Warning: Telegram Bot Token or Chat ID not configured.")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=8)
        if response.status_code == 200:
            safe_log(f"[TELEGRAM] Alert successfully dispatched to chat {target_chat}.")
            return True
        else:
            safe_log(f"[TELEGRAM ERROR] API returned status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        safe_log(f"[TELEGRAM ERROR] Request failed: {e}")
        return False

def dispatch_telegram_alert_async(
    threat_level: str,
    person_count: int,
    trigger_reason: str,
    dispatch_id: Optional[str] = None,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    is_sos: bool = False
) -> None:
    """Asynchronously dispatches a formatted emergency alert to Telegram."""
    icon = "🚨" if is_sos or threat_level == "ALERT" else "⚠️"
    headline = "CRITICAL SOS DISTRESS DETECTED" if is_sos else f"AURA SECURITY ALERT: {threat_level}"
    
    msg_lines = [
        f"{icon} *{headline}*",
        f"• *Threat Level:* `{threat_level}`",
        f"• *Persons In Zone:* `{person_count}`",
        f"• *Trigger:* {trigger_reason}",
    ]
    if dispatch_id:
        msg_lines.append(f"• *Dispatch ID:* `{dispatch_id}`")
    msg_lines.append(f"• *Timestamp:* `{requests.utils.default_user_agent()}`")

    formatted_msg = "\n".join(msg_lines)

    thread = threading.Thread(
        target=send_telegram_message_sync,
        args=(formatted_msg, bot_token, chat_id),
        daemon=True
    )
    thread.start()
