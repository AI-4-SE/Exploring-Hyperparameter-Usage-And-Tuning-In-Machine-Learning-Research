# ----------------------------------------------------------------------------
# Prepare file system
# ----------------------------------------------------------------------------
# We create the folder structure the node where the job runs to save some
# network bandwidth.
EVALUATION=out

LOCALPATH=/tmp/$USER/sklearn/$1

# Clean up leftovers from previous failed runs
# rm -rf $LOCALPATH/

# Create results folder
# mkdir -p $LOCALPATH

# ----------------------------------------------------------------------------
# Prepare environment
# ----------------------------------------------------------------------------

# Create and source virtual environment
python3.10 -m venv $LOCALPATH/venv
source $LOCALPATH/venv/bin/activate


# Install dependencies
wheel="$(find $2 -type f -iname "*.whl")"

pip install $wheel
pip install gitpython joblib
pip install GitPython

cd "$LOCALPATH"

rm -rf "$EVALUATION"

# Get evaluation script
cp $2/evaluation.py .
cp $2/sample_set_urls.json .
# ----------------------------------------------------------------------------
# Run experiment
# ----------------------------------------------------------------------------

python3.10 evaluation.py $1

# ----------------------------------------------------------------------------
# Copy results
# ----------------------------------------------------------------------------

cp -r $LOCALPATH/out/* $2/

# ----------------------------------------------------------------------------
# Clean up
# ------------

deactivate

rm -rf "$LOCALPATH"
