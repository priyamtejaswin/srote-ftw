#!/usr/bin/env python
"""
created at: 23/04/19 1:20 AM
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Utility functions. Data, io, processing, etc.
"""

import os
import sys


VFDEL = '___' ## VideoName___FrameId delimiter.
NFRAMES = 2 ## Number of frames considered per sample.


def _fskey(f):
    vname, fid = f.rstrip('.png').split(VFDEL)
    fid = int(fid)
    return vname, fid

def load_fnames(fdir):
    all_frames = []

    vdirs = [os.path.join(fdir, d) for d in os.listdir(fdir)]
    vdirs = [d for d in vdirs if os.path.isdir(d)]
    print "Found %d vid files."%len(vdirs)

    for dpath in vdirs:
        print "In %s"%dpath
        frames = [f for f in os.listdir(dpath) if f.endswith('.png')]
        print "\tFound %d frames."%len(frames)

        frames = sorted(frames, key=lambda x:_fskey(x))
        frames = [os.path.join(dpath, f) for f in frames]

        ## Group sorted frames by NFRAMES.
        motion_frames = []
        for i in range(len(frames) - NFRAMES + 1):
            motion_frames.append(tuple(frames[i : i+NFRAMES]))

        ## Add to master list.
        all_frames.extend(motion_frames)

    print "Found %d frame groups of length %d."%(len(all_frames), NFRAMES)
    return all_frames


if __name__ == '__main__':
    dirpath = sys.argv[1]
    load_fnames(dirpath)