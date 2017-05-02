#!/bin/bash
python nmt_dynet_attention.py ../data/train.src ../data/train.tgt ../data/dev.src ../data/dev.tgt ../data/final_nmt_dynet_attention --dynet-mem 5000
