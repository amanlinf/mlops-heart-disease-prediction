.PHONY: install-dependencies data eda train test lint serve smoke docker monitor

install-dependencies:
	python -m pip install -r requirements-dev.txt
