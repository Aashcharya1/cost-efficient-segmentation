"""Run frozen checkpoint evaluations and rebuild the report in a single process chain.

This orchestration is outside inference; each timed predictor uses no subprocesses.
"""

import subprocess, sys

for split in ("val", "test", "captured"):
    subprocess.run([sys.executable, "-m", "src.evaluate", "--split", split], check=True)
subprocess.run([sys.executable, "build_report.py"], check=True)
subprocess.run([sys.executable, "verify_submission.py"], check=True)
subprocess.run([sys.executable, "write_readme.py"], check=True)
subprocess.run([sys.executable, "package_submission.py"], check=True)
