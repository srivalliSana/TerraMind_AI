"""Test-suite configuration: ignore any developer-local overrides so tests are deterministic."""
import os

os.environ["TERRAMIND_IGNORE_LOCAL_CONFIG"] = "1"
