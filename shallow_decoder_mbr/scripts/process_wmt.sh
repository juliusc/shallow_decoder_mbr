pip install mtdata==0.4.2
wget https://www.statmt.org/wmt24/mtdata/mtdata.recipes.wmt24-constrained.yml
mtdata cache -j 8 -ri "wmt24-eng-ces"
for id in wmt24-eng-ces; do
  mtdata get-recipe -i $id -o $id --compress --no-merge -j 16
done
