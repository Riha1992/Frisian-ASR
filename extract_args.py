# extract training arguments 
import torch
checkpoint="checkpoint-1500/"
repo_name="wav2vec2_fy_common_voice_33/" 
base_dir = "/data/p312702/from_wietse/"

args_dir=base_dir+repo_name+checkpoint+"training_args.bin"


training_args=torch.load(args_dir)

with open(base_dir+repo_name+checkpoint+"args.txt",'w') as f:
    f.write(training_args)
    
    
    