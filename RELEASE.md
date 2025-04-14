Pypi
====

Preparation:
* increment version in `setup.py`
* add new changelog section in `CHANGES.rst`
* commit/push all changes

Commands for releasing on pypi.org (requires twine >= 1.8.0):

```
find -name "*~" -delete
rm dist/*
python3 setup.py clean
python3 setup.py sdist
twine upload dist/*
```


Github
======

Steps:
* start new release (version: `vX.Y.Z`)
* enter release notes, i.e., significant changes since last release
* upload `wai_ma-X.Y.Z.tar.gz` previously generated with `setup.py`
* publish

