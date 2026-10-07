import pytest
from config import parse_port
def test_valid_port_with_spaces():
    assert parse_port(" 8080 ") == 8080 and parse_port("1") == 1 and parse_port("65535") == 65535
@pytest.mark.parametrize("value", ["0", "65536", "-1", "abc"])
def test_invalid_port_raises(value):
    with pytest.raises(ValueError):
        parse_port(value)
