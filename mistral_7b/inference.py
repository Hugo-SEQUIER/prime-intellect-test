## Imports
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.empty_cache()

def initialize_model():
    model_path = "mistral_model" # Depends of the directory storage
    device='cuda'
    dtype=torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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
    correct = 0
    total = len(dataset)
    for i in range(total):
        question = dataset[i]["question"]
        choices = dataset[i]["choices"]
        input_ids = tokenizer([
                alpaca_prompt.format(
                    question, # instruction
                    choices,
                    "", # output - leave this blank for generation!
                )
            ], return_tensors="pt").to(model.device)
        output_ids = model.generate(**input_ids, max_length=2048, num_return_sequences=1)
        # Decode and print the output
        output_text = tokenizer.batch_decode(output_ids)[0].strip()
        output_text = output_text.split("Response:\n")[1].split("<|")[0].replace("'","").replace('.','').split(' ')[-1] # Ensure output is cleaned up of any extra whitespace
        if output_text == int(dataset[i]["answer"]) or output_text == dataset[i]['choices'][int(dataset[i]['answer'])]:
          correct += 1 
                
    accuracy = correct / total
    
    return accuracy

if __name__ == "__main__":
    model, tokenizer = initialize_model()
    dataset, alcapa_prt = initialize_dataset()

    accuracy_list = []
    for i in range(5):
        accuracy = evaluate_model(model, tokenizer, dataset['test'], alcapa_prt)
        accuracy_list.append(accuracy)
        print('Round : ', i, ' - Accuracy : ', accuracy)
    print(sum(accuracy_list) / len(accuracy_list))