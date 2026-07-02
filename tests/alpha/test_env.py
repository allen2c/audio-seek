import os


def test_env() -> None:
    assert os.getenv("ENVIRONMENT") == "test"
    assert os.getenv("PYTEST_IS_RUNNING") == "true"
