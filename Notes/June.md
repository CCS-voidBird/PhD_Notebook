```bash
#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=40G
#SBATCH --time=12:00:00
#SBATCH -o simulate_gblup_qc.out
#SBATCH -e simulate_gblup_qc.error
#SBATCH --job-name=gblup_simulate
#SBATCH --partition=general

source ~/.bash_profile
module load r/4.2.1-foss-2021a
storage
cd sugarcane_sim/analysis_recombine/maf_qc/
mkdir -p gblup/A/mtg2/
mkdir -p gblup/A/gcta/
path=`pwd`

####
for idx in {1..5}
do
echo $idx
pfolder="$path/data/sugarcane_sim_diploid_V${idx}.phen"
traits=("Rank100" "Rank60" "Rank30" "Rank10" "Rank05" "Mix10030" "Mix10010" "Mix3010")
for trait in {0..7}
do
echo ${traits[trait]}
for keep in 12
do
echo "Keep rate: $keep"

cp data/sugarcane_sim_diploid_${keep}.ped sugarcane_sim_diploid.ped
cp data/sugarcane_sim_diploid_${keep}.fam sugarcane_sim_diploid.fam
cp data/sugarcane_sim_diploid_${keep}.map sugarcane_sim_diploid.map
Rscript convert_plink.R

~/plink --file sugarcane_sim_diploid --allow-no-sex --make-bed --freq --extract data/sugarcane_sim_allele_qc.include --out sugarcane_sim_diploid_allele_qc
~/plink --file sugarcane_sim_diploid --allow-no-sex --make-bed --freq --extract data/sugarcane_sim_12_qc.include --out sugarcane_sim_diploid_12_qc

~/gcta_v1.94/gcta --bfile sugarcane_sim_diploid_allele_qc --make-grm --out ./mtg2_sugarcane_sim_diploid_allele_qc.a
~/gcta_v1.94/gcta --bfile sugarcane_sim_diploid_allele_qc --make-grm-d  --out ./mtg2_sugarcane_sim_diploid_allele_qc

~/gcta_v1.94/gcta --bfile sugarcane_sim_diploid_12_qc --make-grm --out ./mtg2_sugarcane_sim_diploid_12_qc.a
~/gcta_v1.94/gcta --bfile sugarcane_sim_diploid_12_qc --make-grm-d  --out ./mtg2_sugarcane_sim_diploid_12_qc
#Mtg2
traitIdx=$(($trait+3))
cat $pfolder | awk -v trait="$traitIdx" '{print $1,$2,$trait > "./mtg2.a.phen"}' 
#cp $path/grm/sugarcane_sim_diploid_${keep}.a.grm.bin ./mtg2_sugarcane_sim_diploid.a.grm.bin
#cp $path/grm/sugarcane_sim_diploid_${keep}.d.grm.bin ./mtg2_sugarcane_sim_diploid.d.grm.bin
~/mtg2 -p sugarcane_sim_diploid.fam -pheno ./mtg2.a.phen -mod 1 -bg ./mtg2_sugarcane_sim_diploid_allele_qc.a.grm.bin \
        -out ./gblup/A/mtg2/${traits[trait]}_keep${keep}_a_${idx}_allele_qc.txt -bvr ./gblup/A/mtg2/${traits[trait]}_keep${keep}_a_${idx}_allele_qc.bv

~/mtg2 -p sugarcane_sim_diploid.fam -pheno ./mtg2.a.phen -mod 1 -bg ./mtg2_sugarcane_sim_diploid_12_qc.a.grm.bin \
        -out ./gblup/A/mtg2/${traits[trait]}_keep${keep}_a_${idx}_12_qc.txt -bvr ./gblup/A/mtg2/${traits[trait]}_keep${keep}_a_${idx}_12_qc.bv


echo mtg2_sugarcane_sim_diploid_allele_qc.a.grm.bin > mtg2_adgrm_allele_qc.a.txt
echo mtg2_sugarcane_sim_diploid_allele_qc.d.grm.bin >> mtg2_adgrm_allele_qc.a.txt

~/mtg2 -p sugarcane_sim_diploid.fam -pheno ./mtg2.a.phen -mod 1 -mbg mtg2_adgrm_allele_qc.a.txt \
        -out ./gblup/A/mtg2/${traits[trait]}_keep${keep}_ad_${idx}_allele_qc.txt -bvr ./gblup/A/mtg2/${traits[trait]}_keep${keep}_ad_${idx}_allele_qc.bv

echo mtg2_sugarcane_sim_diploid_12_qc.a.grm.bin > mtg2_adgrm_12_qc.a.txt
echo mtg2_sugarcane_sim_diploid_12_qc.d.grm.bin >> mtg2_adgrm_12_qc.a.txt

~/mtg2 -p sugarcane_sim_diploid.fam -pheno ./mtg2.a.phen -mod 1 -mbg mtg2_adgrm_12_qc.a.txt \
        -out ./gblup/A/mtg2/${traits[trait]}_keep${keep}_ad_${idx}_12_qc.txt -bvr ./gblup/A/mtg2/${traits[trait]}_keep${keep}_ad_${idx}_12_qc.bv

##Gcta
#echo ./mtg2_sugarcane_sim_diploid.a > gcta_adgrm.txt
#echo ./mtg2_sugarcane_sim_diploid.d >> gcta_adgrm.txt
#~/gcta_v1.94/gcta --reml --reml-pred-rand --grm $path/grm/sugarcane_sim_diploid_${keep}.a  --pheno ./mtg2.a.phen --mpheno 1 \
#                    --out $path/gblup/A/gcta/${traits[trait]}_keep${keep}_a_${idx} --reml-est-fix --threads 10
#~/gcta_v1.94/gcta --reml --reml-pred-rand --mgrm gcta_adgrm.txt --pheno ./mtg2.a.phen --mpheno 1 \
#                    --out $path/gblup/A/gcta/${traits[trait]}_keep${keep}_ad_${idx} --reml-est-fix --threads 10
#
#~/gcta_v1.94/gcta  --bfile sugarcane_sim_diploid --blup-snp $path/gblup/A/gcta/${traits[trait]}_keep${keep}_a_${idx}.indi.blp  \
#                    --out $path/gblup/A/gcta/${traits[trait]}_keep${keep}_a_${idx} --threads 10
#~/gcta_v1.94/gcta  --bfile sugarcane_sim_diploid --blup-snp $path/gblup/A/gcta/${traits[trait]}_keep${keep}_ad_${idx}.indi.blp  \
#                    --out $path/gblup/A/gcta/${traits[trait]}_keep${keep}_ad_${idx} --threads 10
done
done
done


Rscript ./gblup/A/validate.R


```



Script for DL qc analysis

```bash
composer="./ML_composer/"
mkdir -p "D:/OneDrive - The University of Queensland/PhD/HPC_Results/sugarcane_sim/maf_qc/"
mkdir -p "D:/OneDrive - The University of Queensland/PhD/HPC_Results/sugarcane_sim/maf_qc/A/allele_qc/"
mkdir -p "D:/OneDrive - The University of Queensland/PhD/HPC_Results/sugarcane_sim/maf_qc/AD/allele_qc/"
mkdir -p "D:/OneDrive - The University of Queensland/PhD/HPC_Results/sugarcane_sim/maf_qc/ADD/allele_qc/"
mkdir -p "D:/OneDrive - The University of Queensland/PhD/HPC_Results/sugarcane_sim/maf_qc/ex/allele_qc/"
width=256
depth=2
ploidy=12
maf=0.02
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
traits=("Rank100" "Rank60" "Rank30" "Rank10" "Rank05" "Mix10030" "Mix10010" "Mix3010")
#path="/scratch/qaafi/uqcche32/sugarcane_sim"
geno="${path}/data/sugarcane_sim_allele"
pheno="${path}/data/sugarcane_sim_diploid.phen.ml"
index="${path}/data/sugarcane_sim_diploid.index"
model="MultiLevel Attention"
target="D:/OneDrive - The University of Queensland/PhD/HPC_Results/sugarcane_sim/maf_qc/A/allele_qc/Attention/"
for trait in {0..7}
do
traitIdx=$(($trait+1))
python $composer/GS_composer.py --build --ped "$geno" --pheno "$pheno" --mpheno $traitIdx --index "$index" --trait ${traits[trait]} \
		--maf $maf --ploidy $ploidy \
        --width $width --depth $depth \
        --model "MultiLevel Attention" \
        -o "$target" --quiet 2 --plot --epoch $epoch \
        --locallyConnect $locallyConnect \
        --embedding $embedding \
        --AttentionBlock $AB \
        --activation $act \
        --num-heads $heads \
        --locallyBlock $locallyblock \
        --batch 24 --lr 0.01 --loss $loss --round $round
       
done
```

