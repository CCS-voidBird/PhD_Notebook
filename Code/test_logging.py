import argparse
import logging
from Functions import get_args

args = get_args()
# write all args from variable args to file "Test_logging.txt"
with open("Test_logging.txt", "w") as f:
    for key, value in args.__dict__.items():
        f.write(f"{key} = {value}\n")



