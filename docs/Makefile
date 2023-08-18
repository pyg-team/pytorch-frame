SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pytorch-frame
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0)
