```bash
composer="./ML_composer/"
traits="LogNFNB"

config="D:/OneDrive - The University of Queensland\PhD\data\Barley phase 2\GS_Composer\Barley_NFNB.ini"
python $composer/GS_composer.py --config "$config" --mpheno 2 --trait "$traits" --vindex 2

config="D:/OneDrive - The University of Queensland\PhD\data\Barley phase 2\GS_Composer\Barley_NFNB_LCLAttention.ini"
python $composer/GS_composer.py --config "$config" --mpheno 1 --trait "$traits" --vindex 2

config="D:/OneDrive - The University of Queensland\PhD\data\Barley phase 2\GS_Composer\Barley_NFNB_log.ini"
python $composer/GS_composer.py --config "$config" --mpheno 2 --trait "$traits" --vindex 2 --model "MLP"
```

**The attention CNN with 0.0001 lr, 8 AB and 8head returned 0.13 accuracy.



To do modelling list

1. Log transformation
2. Positional encoding or knowledge based encoding
3. Auto Correlation Attention
4. After-attention conv



model_test code:

```bash
composer="d/GS_Composer_Torch"
traits="test"

config="D:\OneDrive - The University of Queensland\PhD\data\test_dataset\test.ini"
python $composer/GS_composer.py --config "$config" --mpheno 1 --trait "$traits"
```

