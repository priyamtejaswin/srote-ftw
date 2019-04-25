#!/usr/bin/env python
"""
created at: 22/04/19 11:54 PM
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Take a directory of videos, dump the frames.
"""

import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Take a directory of videos, dump the frames.")
parser.add_argument('videos', type=str, help="Required arg; directory of video data files.")
parser.add_argument('--outdir', type=str, help="Optional arg; directory of extracted frames.", default="frames")

args = parser.parse_args()

assert os.path.isdir(args.videos), "Directory - %s - does not exist."%args.videos
assert not os.path.isdir(args.outdir), "Directory - %s - already exists. Delete or specify new."%args.outdir

files = os.listdir(args.videos)
os.makedirs(args.outdir)

for fname in tqdm(files):
    vdir = os.path.join(args.outdir, fname) ## Folder for that video's frames.
    os.makedirs(vdir)
    fpath = os.path.join(args.videos, fname) ## Vid file source.
    wpath = os.path.join(vdir, fname)
    cmd = "ffmpeg -i {} {}___%d.png".format(fpath, wpath)
    os.system(cmd)

print "Done. Extracted frames from %d files."%len(files)
