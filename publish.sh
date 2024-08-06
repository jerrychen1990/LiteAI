version=${1:-''}

echo "run ut"
bash run_ut.sh
echo "build package"
if [ -z "$version" ]; then
    python setup.py sdist bdist_wheel
else
    echo publish with version $version
    python setup.py sdist bdist_wheel $version
fi
echo "upload package"
twine upload dist/*
echo "clean temp data"
rm -rf *.egg-info build dist
echo "done"