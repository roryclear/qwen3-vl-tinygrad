# qwen3-vl-tinygrad

## Setup:
```
pip install -r requirements.txt
```

## Inference on single image:
```
python qwen3vl.py --model={2B|4B} --image={path to an image}
```

### for faster inference use tinygrad's BEAM search:
```
BEAM=2 python qwen3vl.py --model={2B|4B} --image={path to an image}
```
