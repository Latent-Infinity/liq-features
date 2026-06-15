# Phase 0 — liq-features decisions

## Decisions

- **Legacy `volatility.py` preserved as `volatility/_legacy.py` and
  re-exported.** The existing 35-line module (`yang_zhang` /
  `garman_klass` returning a scalar) collides with the new
  `volatility/` package. Rather than delete it in Phase 0, I converted
  the file into the package (via `git mv` so the rename is detected)
  and re-export the two functions from the package's `__init__.py`.
  Existing callers — `tests/test_volatility.py` and any downstream code
  — keep working unchanged. The legacy functions are retired in Phase 5
  per research plan §13 once the ATR-bridge writeup justifies the
  migration with numbers.
- **`VolEstimate` typed against `pd.Series` via `TYPE_CHECKING` import.**
  The frozen dataclass fields type to `pd.Series` per research plan
  §3.6, but `pandas` is a Phase-1 dependency (research plan §3.6 +
  impl plan §1.5); importing it eagerly in Phase 0 would force the dep
  before it's needed. Using `if TYPE_CHECKING:` for the `pd` import
  satisfies the type annotations without the runtime cost.
- **xfail-strict tests split into two groups.** The user-updated Phase 0
  spec asked for both (a) scaffold-state assertions and (b) future
  contract pinning. The test file separates them into
  `TestPhase0Scaffold` (today's `NotImplementedError` raise is one
  test that always passes; the future "returns a `VolEstimate`" is
  xfail-strict) and `TestPhase1ContractFuture` (three xfail-strict
  tests pinning return shape, spec validation, and PIT enforcement).
  Phase 1 turns the four xfails green by satisfying the contract.
- **`estimate_variance(bars: object, spec, *, asof=None)` typing.**
  Phase 0 ships `bars: object` so the signature is stable but doesn't
  yet require a `BarFrame` type that's not introduced until Phase 1.
  Phase 1 will narrow this to the proper bar-frame protocol.
- **No `__init__.py` in `src/liq/` (namespace package).** Confirmed
  per `quant/CLAUDE.md` guardrail — the new `volatility/` package
  lives under the namespace tree and has its own `__init__.py`, but no
  `__init__.py` is added at any level above `liq.features`.

## Deviations

None.
