
"""
Lightweight unit handling that prefers `pint` if available.
"""
try:
    import pint
    _ureg = pint.UnitRegistry()
    Q_ = _ureg.Quantity
    def convert(value, from_unit, to_unit):
        return (value * _ureg(from_unit)).to(to_unit).magnitude
    def quantity(value, unit):
        return Q_(value, unit)
    HAVE_PINT = True
except Exception:
    HAVE_PINT = False
    class _SimpleQuantity:
        def __init__(self, value, unit):
            self.value = float(value)
            self.unit = unit
        def __repr__(self):
            return f"{self.value} {self.unit}"
    def convert(value, from_unit, to_unit):
        # Minimal, same-unit no-op; extend as needed.
        if from_unit != to_unit:
            raise RuntimeError("Unit conversion requires `pint` for non-trivial cases.")
        return float(value)
    def quantity(value, unit):
        return _SimpleQuantity(value, unit)
