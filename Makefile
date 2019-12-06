# Useful makefile commands for hgmca

.PHONY: docs

docs:
	if [ -d "docs/generated" ]; then rm docs/generated/*; fi
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	open docs/_build/html/index.html
