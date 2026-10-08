from calc import safe_divide
def test_zero_divisor_returns_zero():
    assert safe_divide(5, 0) == 0.0
def test_normal_division_unchanged():
    assert safe_divide(6, 3) == 2.0 and safe_divide(-9, 2) == -4.5
