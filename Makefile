all:

clean:
	rm -rf build/ dist/ *.egg-info

release: clean
	python setup.py sdist bdist_wheel

deploy:
	twine upload dist/*
