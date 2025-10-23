
## Format Data as COCO-like format

```bash
python data/format.py
```

This generates the training set.  
If you need a validation set, just split the training data as the comments suggest.

## Train Model
This script provides an example of how to train PointLLM.  
We use [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) for training.

```bash
cd pointllm
deepspeed train.py \
    --unfreeze_language_model \
    --bs 8 --epoch 50 \
    --grad_accum_steps 4 \
    --do_standard 
```

## Evaluate Model
```bash
cd pointllm
deepspeed evaluate.py \
    --bs 8 \
    --do_standard 
```
