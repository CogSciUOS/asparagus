import argparse
import pandas as pd
from pathlib import Path


def combine(folder, outfile):
    """
    Combines the csv files into one.

    Args:
        folder(str): Folder to read the csv files from
        outfile(str): File to combine the csv files
    """
    
    header_written = False
    with Path(outfile).open('w') as outf:
        for infile in Path(folder).iterdir():
            if not infile.is_file():
                continue
            lines = infile.read_text().splitlines()
            if header_written:
                lines = lines[1:]
            else:
                lines[0] = lines[0].replace(",,", "")
                header_written = True
            outf.write('\n'.join(lines) + '\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder to find files to combine in')
    parser.add_argument('outfile', nargs='?', default='labels.csv', help='output file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    combine(args.folder, args.outfile)
