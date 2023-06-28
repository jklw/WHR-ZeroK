#!/bin/zsh
set -eu

source env/bin/activate

#jupyter nbconvert posting1.ipynb --to html --output docs/posting1.html --config jupyter_nbconvert_config.py --template dunno_template --theme dark --no-input
# jupyter nbconvert posting1.ipynb --to html --output docs/posting1.html --config jupyter_nbconvert_config.py --template classic --theme dark

quarto render posting1.ipynb

mkdir -p docs
mv posting1.html docs/
rm -rf docs/posting1_files
mv posting1_files docs/