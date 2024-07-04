echo "run ut"
bash run_ut.sh
echo "build package"
python setup.py sdist bdist_wheel
echo "upload package"
twine upload dist/*
echo "clean temp data"
rm -rf *.egg-info build dist
echo "done"