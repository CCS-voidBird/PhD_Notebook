import argparse
from Structure.ModelProcessor import MODELS, loss_fn
from utils.configGenerator import generate_config_from_parser
import configparser
import os

def get_args():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group(title='General')
    general.add_argument('--make-config',help="Make a config template.", action='store_true')
    general.set_defaults(make_config=False)

    general.add_argument('--config', type=str, help="Config file path.", default=None)
    general.add_argument('--geno', type=str, help="PED-like genotype file name")
    general.add_argument('-pheno', '--pheno', type=str, help="Phenotype file.")
    general.add_argument('-mpheno', '--mpheno', type=int, help="Phenotype columns, start with 1 (FID, IID, 1st Phenotype). If not specified, multi-trait would be enabled.", default=0)
    general.add_argument('-index', '--index', type=str, help="index file", default = None)
    general.add_argument('-vindex', '--vindex', type=int, help="index for validate, 0: no vaidation", default = None)
    #general.add_argument('-include', '--include', type=str, help="Specify a list of SNPs to be included in the analysis.", default = None)
    #general.add_argument('-exclude', '--exclude', type=str, help="Specify a list of SNPs to be excluded in the analysis.", default = None)
    general.add_argument('-annotation', '--annotation', type=str, help="annotation file,1st row as colname", default=None)
    general.add_argument('-o', '--output', type=str, help="Input output dir.",default="./Composed")
    general.add_argument('--trait', type=str, nargs='+', help="give trait a name or name list.", default=None)
    general.add_argument('-maf', '--maf', type=float, help="Filter shreshold for marker MAF, default is 0", default=0.0)
    general.add_argument('-ploidy', '--ploidy', type=int, help="Ploidy for marker MAF, default is 2", default=2)

    task_opts = parser.add_argument_group(title='Task Options')
    task_opts.add_argument('-build', "--build", help="Modelling process.", dest='build', action='store_true')
    parser.set_defaults(build=False)
    task_opts.add_argument('-analysis', '--analysis', help="Analysis process.", dest='analysis', action='store_true')
    parser.set_defaults(analysis=False)
    task_opts.add_argument('--analysis-format', type=int, help="1: SNP on columns, 2: SNP on rows.", default=1)

    task_opts.add_argument('-predict', '--predict', help="Predict process, to predict all the phenotypes as a seperated file, currently not avaliable.", dest='predict', action='store_true')
    parser.set_defaults(predict=False)

    build_args = parser.add_argument_group(title='Model Options')
    build_args.add_argument('--model', type=str, help="Select training model from {}.".format(", ".join(MODELS.keys())))

    ### Neural model default attributes##
    build_args.add_argument('--width', type=int, help="FC layer width (units).", default=8)
    build_args.add_argument('--depth', type=int, help="FC layer depth.", default=4)
    build_args.add_argument('--load', type=str, help="load model from file.", default=None)
    build_args.add_argument('--data-type', type=str, help="Trait type (numerous, ordinal, binary)", default="numerous")
    build_args.add_argument('-r', '--round', type=int, help="training round.", default=1)
    build_args.add_argument('-lr', '--lr', type=float, help="Learning rate.", default=0.0001)
    build_args.add_argument('-epo', '--epoch', type=int, help="training epoch.", default=50)
    build_args.add_argument('-batch', '--batch', type=int, help="batch size.", default=16)
    build_args.add_argument('-numDecay', '--numDecay', type=int, help="Number of samples to apply lr decay.", default=6000)
    build_args.add_argument('--rank', type=bool, help="If the trait is a ranked value, will use a standard value instead.", default=False)
    build_args.add_argument('-quiet', '--quiet', type=int, help="silent mode, 0: quiet, 1: normal, 2: verbose", default=2)
    build_args.add_argument('--method', type=str, help="Methods used for final prediction layer (Multiply or Harmonic).", default="Multiply")


    build_args.add_argument('-plot', '--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)

    build_args.add_argument('-mafm', '--mafm', help="Enable minor allele frequency multiplier, it will adjust genotype alleles with its MAF.", dest='mafm', action='store_true')
    parser.set_defaults(mafm=False)

    build_args.add_argument('-save', '--save', dest='save', action='store_true', help="save model True/False")
    parser.set_defaults(save=False)

    build_args.add_argument('--use-mean', dest='mean', action='store_true')
    parser.set_defaults(mean=False)

    parser.add_argument('--features', type=str, default='ST',
                        help='forecasting task, options:[ST, MT]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate',)

    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, help="loss founction from {}.".format(", ".join(loss_fn.keys())), default="mse")
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    if args.make_config is True:
        generate_config_from_parser(parser, "config.yaml", format="yaml")
        generate_config_from_parser(parser, "config.json", format="json")
        generate_config_from_parser(parser, "config.ini", format="ini")
        print("Config file generated, please check the current directory.")
        exit(0)

    if args.config:
        config_path = os.path.abspath(args.config)
        config = configparser.ConfigParser()
    ## Let configparser can read captial letter
        config.optionxform = lambda option: option

        config.read(config_path)
    ## replace the default value with the config file
        print("Config file path: ", config_path)
        print("Reading values with config file...")
    if args.config:
        for mothers_key in config:
            #print(mothers_key)
            print()
            print("Now reading {}...".format(mothers_key))
            #print(config[mothers_key].keys)
            for key in config[mothers_key]:
                if key in args and getattr(args,key) is parser.get_default(key) and config.get(mothers_key,key):
                    #print("Set {} value as {} ...............(from config file)".format(key,config.get(mothers_key,key)))
                    #print("Set {} value as {} ...............(from config file)".format(key,getattr(args,key), config.get(mothers_key,key) ))
                    if type(getattr(args,key)) == bool:
                        setattr(args, key, config.getboolean(mothers_key,key))
                    elif type(getattr(args,key)) == int:
                        setattr(args, key, config.getint(mothers_key,key))
                    elif type(getattr(args,key)) == float:
                        setattr(args, key, config.getfloat(mothers_key,key))
                    else:
                        setattr(args, key, config.get(mothers_key,key))
                elif key not in args:
                    print("Key {} not in args".format(key))
                elif getattr(args,key) is not parser.get_default(key):
                    print("Replace config parameter: {} to {}...............(from command)".format(key, getattr(args,key) ))
                    #print("Key {} is passed by command parameter".format(key))

    return args, parser