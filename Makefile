.PHONY: install test smoke small full scaling scaling-quick all clean

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=src python -m pytest tests/ -v

# Offline sanity check. No network, runs in well under a minute on CPU.
smoke:
	python scripts/run_ablations.py --synthetic --epochs 4 --seeds 2 \
		--patience 0 --d-model 32 --d-ff 64 --out results/smoke

# Roughly the size of the original coursework notebook, for comparison.
small:
	python scripts/run_ablations.py --n-train 1000 --n-val 500 --n-test 500 \
		--epochs 20 --seeds 3 --out results/small

# The headline run. Intended for a GPU.
full:
	python scripts/run_ablations.py --n-train 20000 --n-val 5000 --n-test 25000 \
		--epochs 30 --seeds 3 --batch-size 64 --out results/full

# Data-scaling sweep. The headline chart: where does the Transformer overtake
# tf-idf? Intended for a GPU.
scaling:
	python scripts/run_scaling.py --sizes 500 1000 2000 5000 10000 20000 \
		--epochs 30 --seeds 3 --batch-size 64 --out results/scaling

# A faster version of the sweep, fewer sizes and one seed, for a first look.
scaling-quick:
	python scripts/run_scaling.py --sizes 500 2000 10000 \
		--epochs 20 --seeds 1 --batch-size 64 --out results/scaling-quick

# Everything, in the order the README presents it.
all: test full scaling

clean:
	rm -rf __pycache__ .pytest_cache src/**/__pycache__ tests/__pycache__
