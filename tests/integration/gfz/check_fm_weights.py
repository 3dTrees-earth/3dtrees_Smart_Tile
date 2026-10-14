import argparse
import hashlib
from pathlib import Path
import json
import sys
sys.path.insert(0, "/workspace")
import torch
from mmengine.config import Config
from mmdet3d.registry import MODELS
import oneformer3d
from mmdet3d.utils import register_all_modules
register_all_modules(init_default_scope=True)

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--checkpoint', required=True)
args = parser.parse_args()
config = Config.fromfile(args.config)
model = MODELS.build(config.model)
checkpoint = torch.load(args.checkpoint, map_location='cpu')
state = checkpoint.get('state_dict', checkpoint)
model.load_state_dict(state, strict=True)
print(json.dumps({'strict_checkpoint_load':'passed','parameters':len(state), 'config':args.config, 'checkpoint':args.checkpoint, 'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()}))
