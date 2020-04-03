import argparse
from pathlib import Path


def combine(folders, outfile):
    """
    Combines the csv files into one.

    Args:
        folders(list): Folders to read the csv files from
        outfile(str): File to combine the csv files
    """
    header_written = False
    with Path(outfile).open('w') as outf:
        for folder in folders:
            for infile in Path(folder).iterdir():
                if not infile.is_file():
                    continue
                lines = infile.read_text().splitlines()
                if header_written:
                    lines = lines[1:]
                else:
                    header_written = True
                outf.write('\n'.join(lines) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    # give one or many folder names in which the label files lie
    parser.add_argument('folder', nargs='+',
                        help='folders to find files to combine in')
    # optional: give outfile name
    parser.add_argument('--outfile', nargs='?',
                        default='labels.csv', help='output file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    combine(args.folder, args.outfile)
