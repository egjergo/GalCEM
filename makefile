exportcondaenv: 
	conda env export --no-builds | grep -v "^prefix: " > environment.yml

sphinxdocs:
	pandoc --mathjax README.md -o docs/README.rst
	sphinx-build docs/ docs/_build