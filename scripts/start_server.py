#!/usr/bin/env python3
"""Check the bundled release, then start Streamlit on Railway's PORT."""
import os
import sys
from pathlib import Path
from check_release import check_release

root = Path(__file__).resolve().parent.parent
try:
    port = int(os.environ.get("PORT", "8501"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")
    check_release(root)
except Exception as exc:
    sys.exit("Startup check failed: " + str(exc))
os.chdir(root)
os.environ["FOOTBALL_PROJECT_ROOT"] = str(root)
os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py",
                        "--server.address=0.0.0.0", "--server.port=" + str(port),
                        "--server.headless=true", "--browser.gatherUsageStats=false"])
