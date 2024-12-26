import warnings
import pytest

@pytest.fixture(autouse=True)
def ignore_protobuf_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*PyType_Spec.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*custom tp_new.*",
        category=DeprecationWarning,
    )