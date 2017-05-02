#!/bin/bash
python nmt_dynet.py ../data/train.src ../data/train.tgt ../data/dev.src ../data/dev.tgt ../data/final_nmt_dynet --dynet-mem 5000
