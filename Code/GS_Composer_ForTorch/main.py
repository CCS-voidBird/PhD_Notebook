import os
import shutil
from Structure.Composer_main import ML_composer
from Structure.Composer_ArgsPaeser import get_args    


if __name__ == "__main__":

    args, parser = get_args()
    
    """
    Create folders from given output path
    """
    if args.output[0] == "/":
        locat = '/' + args.output.strip('/') + '/'
    else:
        locat = args.output.strip('/') + '/'
    if not os.path.exists(locat):
        os.mkdir(locat)

    ##write args to a txt file in the locat
    with open(locat + 'args.txt', 'w') as f:
        f.write(str(args))

    ##paste the config file to the locat
    if args.config:
        shutil.copy(args.config, locat + 'config.ini')

    
    if args.build is True:
        composer = ML_composer(args=args)

        index_ref = composer.prepare_cross_validate()
        i = 1
        for train_idx,valid_idx in index_ref:
            print("Cross-validate: {},{}".format(i,valid_idx[0]))
            composer.data_provider.prepare_training(train_idx,valid_idx)
            composer.compose(train_idx,valid_idx,valid_idx[0])
            i+=1

    elif args.analysis is True and args.load is not None and args.build is False:
        print("Start analysis model...")
        composer = ML_composer(args=args)
        maf_info = composer._info["MAF"]
        #investigate_model(
        #                model_path=args.load,ploidy=args.ploidy,marker_maf = np.array(maf_info),args=args)