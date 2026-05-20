"""
Sterling Workspace — Project Scaffolding
=========================================
Creates and manages projects in a configurable workspace directory.
Sterling's own files are never touched — everything goes under workspace.path.

Supported languages:
  Python, C, C++, JavaScript/Node, Rust, Go, Bash

Project layout examples:

  Python:
    my_project/
      main.py
      requirements.txt
      venv/              ← created via python -m venv
      README.md

  C / C++:
    my_project/
      main.c (or .cpp)
      Makefile
      README.md

  JavaScript:
    my_project/
      index.js
      package.json
      README.md

  Rust:
    my_project/
      src/main.rs
      Cargo.toml
      README.md

  Go:
    my_project/
      main.go
      go.mod
      README.md

  Bash:
    my_project/
      main.sh
      README.md
"""

import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("sterling.workspace")

# ─────────────────────────────────────────────────────────────────────────────
# Language metadata
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGE_ALIASES: dict[str, str] = {
    "python":       "python",
    "py":           "python",
    "c++":          "cpp",
    "cpp":          "cpp",
    "c plus plus":  "cpp",
    "c":            "c",
    "javascript":   "javascript",
    "js":           "javascript",
    "node":         "javascript",
    "nodejs":       "javascript",
    "rust":         "rust",
    "go":           "go",
    "golang":       "go",
    "bash":         "bash",
    "shell":        "bash",
    "sh":           "bash",
}

ENTRY_FILE: dict[str, str] = {
    "python":     "main.py",
    "cpp":        "main.cpp",
    "c":          "main.c",
    "javascript": "index.js",
    "rust":       "src/main.rs",
    "go":         "main.go",
    "bash":       "main.sh",
}

# Minimal boilerplate written when no description is given
BLANK_TEMPLATES: dict[str, str] = {
    "python": (
        "def main():\n"
        "    pass\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n"
    ),
    "cpp": (
        "#include <iostream>\n\n"
        "int main() {\n"
        "    return 0;\n"
        "}\n"
    ),
    "c": (
        "#include <stdio.h>\n\n"
        "int main(void) {\n"
        "    return 0;\n"
        "}\n"
    ),
    "javascript": (
        "'use strict';\n\n"
        "function main() {\n"
        "}\n\n"
        "main();\n"
    ),
    "rust": (
        "fn main() {\n"
        "}\n"
    ),
    "go": (
        "package main\n\n"
        "func main() {\n"
        "}\n"
    ),
    "bash": (
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Workspace
# ─────────────────────────────────────────────────────────────────────────────

class Workspace:
    """
    Manages the Sterling workspace directory.
    Creates project scaffolding and writes generated code files.
    Never touches anything outside workspace_path.
    """

    def __init__(self, workspace_path: str):
        self._root = Path(workspace_path).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Workspace root: {self._root}")

    @property
    def root(self) -> Path:
        return self._root

    def project_path(self, name: str) -> Path:
        return self._root / self._safe_name(name)

    # ─────────────────────────────────────────────────────────────────────────
    # Project creation
    # ─────────────────────────────────────────────────────────────────────────

    def create_project(
        self,
        name:        str,
        language:    str,
        code:        Optional[str] = None,
        description: str = "",
    ) -> dict:
        """
        Scaffold a new project directory for the given language.

        Args:
            name:        Project name. Converted to a safe directory name.
            language:    Canonical language key (e.g. "python", "cpp").
            code:        Generated code string. If None, uses blank template.
            description: Human-readable description — written into README.

        Returns:
            Dict with keys: path (str), entry_file (str), language (str), created (bool)
        """
        safe     = self._safe_name(name)
        project  = self._root / safe
        lang     = language.lower()

        already_exists = project.exists()
        project.mkdir(parents=True, exist_ok=True)

        entry     = ENTRY_FILE.get(lang, "main.txt")
        content   = code if code else BLANK_TEMPLATES.get(lang, "")

        # Some languages need subdirectories for the entry file
        entry_path = project / entry
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(content, encoding="utf-8")

        # Language-specific extras
        self._write_extras(project, lang, name, description)

        # Python: set up venv
        if lang == "python":
            self._create_venv(project)

        logger.info(f"Project '{safe}' created at {project} (language: {lang})")

        return {
            "path":       str(project),
            "entry_file": entry,
            "language":   lang,
            "created":    not already_exists,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Language-specific extras
    # ─────────────────────────────────────────────────────────────────────────

    def _write_extras(self, project: Path, lang: str, name: str, description: str):
        readme = (
            f"# {name}\n\n"
            f"{description}\n\n" if description else f"# {name}\n\n"
            f"Created by Sterling — {datetime.now().strftime('%Y-%m-%d')}\n"
        )
        (project / "README.md").write_text(readme, encoding="utf-8")

        if lang == "python":
            req = project / "requirements.txt"
            if not req.exists():
                req.write_text("# Add dependencies here\n", encoding="utf-8")

        elif lang in ("c", "cpp"):
            makefile = project / "Makefile"
            if not makefile.exists():
                src  = "main.cpp" if lang == "cpp" else "main.c"
                comp = "g++" if lang == "cpp" else "gcc"
                makefile.write_text(
                    f"CC = {comp}\n"
                    f"SRC = {src}\n"
                    f"OUT = main\n\n"
                    f"all:\n\t$(CC) -o $(OUT) $(SRC)\n\n"
                    f"clean:\n\trm -f $(OUT)\n",
                    encoding="utf-8",
                )

        elif lang == "javascript":
            pkg = project / "package.json"
            if not pkg.exists():
                pkg.write_text(
                    f'{{\n  "name": "{self._safe_name(name)}",\n'
                    f'  "version": "1.0.0",\n'
                    f'  "main": "index.js"\n}}\n',
                    encoding="utf-8",
                )

        elif lang == "rust":
            cargo = project / "Cargo.toml"
            if not cargo.exists():
                cargo.write_text(
                    f'[package]\nname = "{self._safe_name(name)}"\n'
                    f'version = "0.1.0"\nedition = "2021"\n',
                    encoding="utf-8",
                )

        elif lang == "go":
            mod = project / "go.mod"
            if not mod.exists():
                mod.write_text(
                    f"module {self._safe_name(name)}\n\ngo 1.21\n",
                    encoding="utf-8",
                )

    def _create_venv(self, project: Path):
        """Create a Python venv inside the project directory."""
        venv_path = project / "venv"
        if venv_path.exists():
            logger.debug(f"venv already exists at {venv_path}")
            return
        try:
            logger.info(f"Creating venv at {venv_path}...")
            subprocess.run(
                ["python3", "-m", "venv", str(venv_path)],
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.info("venv created.")
        except subprocess.TimeoutExpired:
            logger.warning("venv creation timed out.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"venv creation failed: {e.stderr.decode()}")
        except Exception as e:
            logger.warning(f"venv creation error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_name(name: str) -> str:
        """Convert a project name to a safe directory name."""
        import re
        safe = re.sub(r"[^\w\s-]", "", name.lower())
        safe = re.sub(r"[\s-]+", "_", safe).strip("_")
        return safe or "project"
