#!/bin/zsh
set -eu

source env/bin/activate

#jupyter nbconvert posting1.ipynb --to html --output docs/posting1.html --config jupyter_nbconvert_config.py --template dunno_template --theme dark --no-input
# jupyter nbconvert posting1.ipynb --to html --output docs/posting1.html --config jupyter_nbconvert_config.py --template classic --theme dark

p='posting1_v2'

quarto render ${p}.ipynb

mkdir -p docs
mv ${p}.html docs/
rm -rf docs/${p}_files
mv ${p}_files docs/