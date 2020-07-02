#!/bin/sh
flake8 .

coverage erase
coverage run -m pytest
coverage report -m
coverage html -d coverage_report

rm -fr dist
python setup.py clean
python setup.py sdist bdist_wheel > /dev/null
