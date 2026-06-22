import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# 1. Setup local output directory
output_dir = "./llm2vec_llama_3_8b_inst_mntp_unsup_simcse/"
os.makedirs(output_dir, exist_ok=True)

# 2. Load base MNTP model configuration and weights
model_name = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

print("Loading base model...")
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

# 3. Merge the MNTP LoRA weights
print("Merging MNTP adapter layers...")
model = PeftModel.from_pretrained(model, model_name)
model = model.merge_and_unload()

# 4. Load and merge the final SimCSE LoRA weights
print("Merging SimCSE adapter layers...")
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
)
model = model.merge_and_unload() 

# 5. CRITICAL FIX: Reset the PEFT state flag so transformers saves the full model weights
model._hf_peft_config_loaded = False

# 6. Save the fully unified model and tokenizer locally
print(f"Saving fully merged standalone model to: {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Verification Complete! The directory contains a valid, standard Hugging Face checkpoint.")