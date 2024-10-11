echo "run ut"
bash run_ut.sh
echo "build package"
python -m build
echo "upload package"
twine check dist/*
twine upload dist/*
echo "clean temp data"
rm -rf *.egg-info build dist
echo "done"