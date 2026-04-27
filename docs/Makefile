SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = _build

.PHONY: html clean

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	python -c "import shutil; shutil.rmtree('$(BUILDDIR)', ignore_errors=True)"
