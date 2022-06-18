import os
import sys
from deepstream_io.deepstream_io import start



args = ['file:///home/hduser/Documents/Nhat/data/17.mp4',
        'file:///home/hduser/Documents/Nhat/data/16.mp4',
        'file:///home/hduser/Documents/Nhat/data/7.mp4',
        'file:///home/hduser/Documents/Nhat/data/2.mp4',
    ]

start(args)