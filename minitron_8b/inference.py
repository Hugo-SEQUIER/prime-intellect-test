## Imports
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from peft import LoraConfig
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

def initialize_model():
    model_path = "nvidia/Minitron-8B-Base"
    dtype=torch.bfloat16
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=load_in_4bit))
    
    peft_config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        init_lora_weights=False,
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.add_adapter(peft_config, adapter_name="adapter_1")
    tokenizer.pad_token = tokenizer.eos_token
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
        output_ids = model.generate(**input_ids, max_length=512, max_new_tokens=128, num_return_sequences=1)
        # Decode and print the output
        output_text = tokenizer.batch_decode(output_ids)[0].strip()
        list_response.append(output_text)
    return list_response

def create_json(list_response):
    fichier_json = './response_dict.json'
    with open(fichier_json, 'w', encoding='utf-8') as f:
        json.dump(list_response, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    model, tokenizer = initialize_model()

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

    list_response = {}
    dataset, alcapa_prt = initialize_dataset()
    for subject in list_subject:
        dataset = dataset.filter(lambda x: x['subject'] == subject)
        print('Subject : ', subject, '-'*5)
        response = evaluate_model(model, tokenizer, dataset['validation'], alcapa_prt)
        list_response[subject] = response
    
    create_json(list_response)