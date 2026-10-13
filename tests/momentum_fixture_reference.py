"""Independent Decimal reference for one frozen unchanged engineering case."""

from decimal import Decimal, localcontext

from liq.data.alpaca_momentum_numerical_files import digest, write_once
from liq.data.alpaca_momentum_numerical_runtime import ReferenceBinding
from test_momentum_daily_fixture import EVIDENCE, scope_case


def main() -> tuple[str, tuple[int, ...]]:
    with scope_case("reference") as (binding, fixture):
        indexed = dict(zip(fixture.calendar.expected_labels, fixture.closes, strict=True))
        endpoints = fixture.calendar.monthly_endpoints
        with localcontext() as context:
            context.prec = 50
            value = indexed[endpoints[-1]] / indexed[endpoints[0]] - Decimal(1)
        reference = ReferenceBinding(contract_sha256=binding.contract_sha256, value=str(value))
        raw = (reference.model_dump_json(indent=2) + "\n").encode()
        binding.check()
    identity = write_once(EVIDENCE, "reference.json", raw)
    return digest(raw), identity


if __name__ == "__main__":
    main()
