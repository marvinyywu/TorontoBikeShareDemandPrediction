from unittest.mock import patch

import features
import pytest


@pytest.fixture(autouse=True)
def _no_io(tmp_path):
    """Redirect all features.py file-writes to a temp directory."""
    with (
        patch("features._PROCESSED_DIR", tmp_path),
        patch("features._FEATURES_PATH", tmp_path / "featured.parquet"),
    ):
        yield
