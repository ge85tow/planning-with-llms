from safetensors.torch import load_file, save_file
from collections import OrderedDict
import re
import os

n=3
it=19
base_dir=f'/srv/chawak/planning-with-llms/results/SFT/{n}_blocks/'
input_path = base_dir+f'iteration_{it}/adapter_model.safetensors'
output_path = base_dir+f'changed_iteration_{it}/adapter_model.safetensors'

# Load the original safetensors
state_dict = load_file(input_path)
fixed_state_dict = OrderedDict()

for k in list(state_dict.keys())[:5]:
    print(f"-----------Original key: {k}---------")

for key, tensor in state_dict.items():
    # Step 1: Remove repeated base_model.model prefixes
    parts = key.split(".")
    while parts[:2] == ["base_model", "model"]:
        parts = parts[2:]
    key = "base_model.model." + ".".join(parts)

    # Step 2: Insert 'default' into LoRA adapter keys if missing
    key = re.sub(r'(lora_[AB])\.(weight|bias)$', r'\1.default.\2', key)

    fixed_state_dict[key] = tensor

for k in list(fixed_state_dict.keys())[:5]:
    print(f"-----------Fixed key: {k}---------")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# Save the corrected file
save_file(fixed_state_dict, output_path)
print(f"✅ Fixed adapter keys saved to: {output_path}")
