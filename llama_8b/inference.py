## Imports
from datasets import load_dataset
import torch
from unsloth import FastLanguageModel
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

def initialize_model(max_seq_length):
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    return model, tokenizer

def initialize_dataset():
    ds = load_dataset("cais/mmlu", "all")

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Select the choice from the given 'Choices'.
    It's Very Important to have an output containing THE CHOICE !!!!
    ### Instruction:
    {}

    ### Choices:
    {}

    ### Response:
    {}"""

    return ds, alpaca_prompt

def evaluate_model(model, tokenizer, dataset, alpaca_prompt):
    list_response = []
    for i in range(len(dataset)):
        question = dataset[i]["question"]
        choices = dataset[i]["choices"]
        input_ids = tokenizer([
                alpaca_prompt.format(
                    question, # instruction
                    choices,
                    "", # output - leave this blank for generation!
                )
            ], return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=128,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
        # Decode and print the output
        output_text = tokenizer.batch_decode(output_ids)[0].strip()
        list_response.append(output_text)
    return list_response

def create_json(list_response):
    fichier_json = './response_dict.json'
    with open(fichier_json, 'w', encoding='utf-8') as f:
        json.dump(list_response, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    max_seq_length = 2048
    model, tokenizer = initialize_model(max_seq_length)

    list_subject = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
       'clinical_knowledge', 'college_biology', 'college_chemistry',
       'college_computer_science', 'college_mathematics',
       'college_medicine', 'college_physics', 'computer_security',
       'conceptual_physics', 'econometrics', 'electrical_engineering',
       'elementary_mathematics', 'formal_logic', 'global_facts',
       'high_school_biology', 'high_school_chemistry',
       'high_school_computer_science', 'high_school_european_history',
       'high_school_geography', 'high_school_government_and_politics',
       'high_school_macroeconomics', 'high_school_mathematics',
       'high_school_microeconomics', 'high_school_physics',
       'high_school_psychology', 'high_school_statistics',
       'high_school_us_history', 'high_school_world_history',
       'human_aging', 'human_sexuality', 'international_law',
       'jurisprudence', 'logical_fallacies', 'machine_learning',
       'management', 'marketing', 'medical_genetics', 'miscellaneous',
       'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy',
       'prehistory', 'professional_accounting', 'professional_law',
       'professional_medicine', 'professional_psychology',
       'public_relations', 'security_studies', 'sociology',
       'us_foreign_policy', 'virology', 'world_religions']
    FastLanguageModel.for_inference(model)
    list_response = {}
    dataset, alcapa_prt = initialize_dataset()
    for subject in list_subject:
        dataset_filtered = dataset.filter(lambda x: x['subject'] == subject)
        print('Subject : ', subject, '-'*5)
        response = evaluate_model(model, tokenizer, dataset_filtered['validation'], alcapa_prt)
        list_response[subject] = response
    
    create_json(list_response)