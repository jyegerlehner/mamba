# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# from mamba_ssm.modules.control import control_callback
from einops import rearrange

import pickle 
import pathlib
import copy

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--ssm_state_file", type=str, default=None) 
parser.add_argument('--steer', type=bool, default=False)
args = parser.parse_args()

repeats = 0
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Save the state machines' states to a file.
def prompt_ssm_state_callback(states):
    print('===================')
    print('prompt state callback:')
    for i, state in enumerate(states):
            print('-----------')
            print('layer_index:{0}'.format(i))
            print('state shape:{0}'.format(state.shape))
            # print(state)

    state_file = args.ssm_state_file
    steer = args.steer
    # We are either steering to previously-saved target states, or
    # we are saving the target states "harvested" from a prompt.
    # Don't save the target states if we are steering.
    if not steer:
        if state_file is not None:
            # if state_file is not None:
            directory = pathlib.Path('/media/jim/mass1/mamba')
            filename =   "{0}.pickle".format(state_file)

            outfile = directory / filename
            print('saving states to:{0}'.format(outfile))
            with open(outfile, 'wb') as file:
                pickle.dump(states, file)

def control_callback(dA, dB, C, self.D, ssm_state, layer_idx):
    return None

# def compute_control_effort(ssm_state, )                

torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
max_length = input_ids.shape[1] + args.genlen

if args.steer:
    cc = control_callback
else:
    cc = None

fn = lambda: model.generate(
    input_ids=input_ids,
    max_length=max_length,
    # cg=True,
    cg=False,
    return_dict_in_generate=True,
    output_scores=True,
    prompt_ssm_state_callback=prompt_ssm_state_callback,
    control_callback=cc,
    enable_timing=False,
    temperature=args.temperature,
    top_k=args.topk,
    top_p=args.topp,
    )

out = fn()
if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
if repeats > 0:
    print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")

