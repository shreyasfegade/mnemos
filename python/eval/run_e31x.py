"""E31x: the mem0 oracle under two more local models, one probe per fact.

    cd python && ..\\.venv\\Scripts\\python -u -m eval.run_e31x

Starts Ollama if needed, pulls ``mistral:7b-instruct`` and ``gemma2:9b`` (each fits the 8 GB GPU
alone), runs ``eval.exp2.e31_amplify --k 1`` with each in the framework venv, and re-aggregates.
The predictions these runs test are registered in ``eval/exp2/verify_r4.py`` (E31x).
"""

from __future__ import annotations

import subprocess
import sys

from eval import overnight as o

MODELS = ("mistral:7b-instruct", "gemma2:9b")


def main() -> None:
    o.MODELS = [*o.MODELS, *MODELS]
    o.start_ollama(sys.stdout)
    for m in MODELS:
        print(f"== {m}", flush=True)
        r = subprocess.run(
            [
                str(o.VENV_FW),
                "-u",
                "-m",
                "eval.exp2.e31_amplify",
                "--model",
                m,
                "--k",
                "1",
                "--probes",
                "60",
                "--seed-size",
                "60",
            ],
            cwd=o.PY,
        )
        print(f"{m}: exit {r.returncode}", flush=True)
    subprocess.run([str(o.VENV_FW), "-m", "eval.exp2.e31_amplify", "--aggregate"], cwd=o.PY)


if __name__ == "__main__":
    main()
