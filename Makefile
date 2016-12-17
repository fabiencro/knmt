all:

clean:
	rm -rf build/ dist/

release: clean
	python setup.py sdist bdist_wheel

deploy:
	twine upload dist/*
