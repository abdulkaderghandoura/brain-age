import argparse
parser = argparse.ArgumentParser(description="Toy Example")
parser.add_argument('--flag', nargs='+', action='append', required=True)
args = parser.parse_args()
print(args.flag)