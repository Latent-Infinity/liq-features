"""One frozen Decimal comparison on the unchanged, guarded two-series case."""

import hashlib
import json
import sys
from decimal import Decimal

from test_daily_breadth_fixture import EVIDENCE, read_case


class BreadthReferenceRefused(ValueError):
    """The invocation does not identify the frozen finite-case contract."""


def main() -> None:
    contract_raw = (EVIDENCE / "contract.json").read_bytes()
    contract_hash = hashlib.sha256(contract_raw).hexdigest()
    if len(sys.argv) != 2 or contract_hash != sys.argv[1]:
        raise BreadthReferenceRefused("exact frozen breadth contract digest required")
    contract = json.loads(contract_raw)
    pair, labels = read_case(contract_raw)
    predicates = {
        fixture.request.symbol: fixture.closes[-1] < max(fixture.closes[-60:]) for fixture in pair
    }
    percentage = Decimal(sum(predicates.values())) * Decimal(100) / Decimal(2)
    receipt = {
        "contract_sha256": contract_hash,
        "predicates": predicates,
        "percentage": str(percentage),
        "enabled": percentage >= Decimal(90),
        "source_hashes": {fixture.request.symbol: fixture.source_sha256 for fixture in pair},
        "original_available_at": {
            fixture.request.symbol: fixture.available_at.isoformat() for fixture in pair
        },
        "population_available_at": contract["population_available_at"],
        "evaluation_cutoff": contract["evaluation_cutoff"],
        "decoded_observations": sum(len(fixture.closes) for fixture in pair),
        "arithmetic_observations": 120,
        "original_labels": [label.isoformat() for label in labels],
        "scope": "DEV_SMOKE_ONLY; exact captured KHC/SPY pair, not S&P membership",
        "signal_trials": 0,
        "economic_looks": 0,
        "new_source_requests": 0,
    }
    with (EVIDENCE / "reference.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
