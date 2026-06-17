# volbench — one-command reproduction and developer tasks.
# Scripts run with the package on the path via `python -m`-style PYTHONPATH.

PY := PYTHONPATH=src python3
SCRIPTS := scripts

.PHONY: help install dev test lint typecheck \
        validate benchmark garch figures economic multivariate harfamily regime \
        reproduce clean

help:
	@echo "Targets:"
	@echo "  install      pip install -e ."
	@echo "  dev          pip install -e .[dev]"
	@echo "  test         run the pytest suite"
	@echo "  lint         ruff check"
	@echo "  typecheck    mypy src/volbench"
	@echo "  validate     estimator validation vs simulation -> results/validation.json"
	@echo "  benchmark    Track 1 RV benchmark           -> results/summary.json + tables"
	@echo "  garch        Track 2 GARCH on returns       -> results/garch.json"
	@echo "  harfamily    HAR family (jumps/semivar)     -> results/har_family.json"
	@echo "  multivariate cross-index spillover HAR      -> results/multivariate.json"
	@echo "  ml           rigorous ML vs HAR comparison   -> results/ml.json"
	@echo "  economic     economic-value evaluation      -> results/economic.json"
	@echo "  vrp          variance-risk-premium edge      -> results/vrp.json"
	@echo "  strategy     vol-targeting + regime overlay  -> results/strategy.json"
	@echo "  crypto       Track 3: BTC/ETH/BNB/SOL        -> results/crypto.json"
	@echo "  regime       calm vs crisis subsamples      -> results/regime.json"
	@echo "  figures      publication figures            -> results/figures/"
	@echo "  reproduce    run the full pipeline end to end"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check src scripts tests

typecheck:
	mypy src/volbench || true

validate:
	$(PY) $(SCRIPTS)/validate_estimators.py

benchmark:
	$(PY) $(SCRIPTS)/run_benchmark.py

garch:
	$(PY) $(SCRIPTS)/run_garch.py

harfamily:
	$(PY) $(SCRIPTS)/run_har_family.py

multivariate:
	$(PY) $(SCRIPTS)/run_multivariate.py

ml:
	$(PY) $(SCRIPTS)/run_ml.py

economic:
	$(PY) $(SCRIPTS)/run_economic.py

vrp:
	$(PY) $(SCRIPTS)/run_vrp.py

strategy:
	$(PY) $(SCRIPTS)/run_strategy.py

crypto:
	$(PY) $(SCRIPTS)/build_crypto.py
	$(PY) $(SCRIPTS)/run_crypto.py

regime:
	$(PY) $(SCRIPTS)/run_regime.py

figures:
	$(PY) $(SCRIPTS)/make_figures.py

reproduce: validate benchmark garch harfamily multivariate ml economic vrp strategy crypto regime figures test
	@echo "\nFull reproduction complete. See results/ and run the report build in report/."

clean:
	rm -rf results/tables/*.csv results/figures/*.png results/*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
