```bash
composer="D:/GS_Composer_Torch/"
traits="test"

config="D:\OneDrive - The University of Queensland\PhD\data\test_dataset\test.ini"
python $composer/main.py --build --config "./config.ini" --mpheno 1 --trait "$traits" --epochs 10 --model "Transformer"
```

