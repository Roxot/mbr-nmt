import argparse

import mbr_nmt.translate as translate

def main():
    parser = argparse.ArgumentParser(description="mbr-nmt: minimum Bayes-risk decoding for neural machine translation")
    subparsers = parser.add_subparsers(dest="command")
    translate.create_parser(subparsers)
    args = parser.parse_args()

    if args.command == "translate":
        translate.translate(args)       
