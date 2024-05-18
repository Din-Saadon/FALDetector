#!/bin/bash
python main.py --inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder \
--inference_dataset_root ./demo \
--resume ../FlowNet2_checkpoint.pth.tar
