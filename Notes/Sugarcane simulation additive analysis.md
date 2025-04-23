```bash
composer="./ML_composer/"
width=256
depth=2
ploidy=2
locallyConnect=16
locallyblock=10
embedding=16
heads=1
round=1
epoch=30
loss="mse"
act="leaky_relu"
AB=1
path="D:/OneDrive - The University of Queensland/PhD/data/sugarcane_sim_recombination/"
traits=("Norm100" "Norm60" "Norm30" "Norm10" "Norm05")
#path="/scratch/qaafi/uqcche32/sugarcane_sim"
geno="${path}/data/sugarcane_sim_diploid_12"
pheno="${path}/data/sugarcane_sim_diploid.add.phen2"
index="${path}/data/sugarcane_sim_diploid.index"
model="LCL Attention"
target="D:/OneDrive - The University of Queensland/PhD/HPC_Results/sugarcane_sim/LocalTestADD/"
trait=1
traitIdx=$(($trait+1))
python $composer/GS_composer.py --build --analysis --geno "$geno" --pheno "$pheno" --mpheno $traitIdx --index "$index" --trait ${traits[trait]} \
        --width $width --depth $depth \
        --model "$model" \
        -o "$target" --quiet 2 --plot --epoch $epoch \
        --locallyConnect $locallyConnect \
        --embedding $embedding \
        --AttentionBlock $AB \
        --activation $act \
        --num-heads $heads \
        --locallyBlock $locallyblock \
        --batch 24 --lr 0.01 --loss $loss --round $round
```

