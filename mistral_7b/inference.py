from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, load_metric
import torch
import nltk
nltk.download('wordnet')
torch.cuda.empty_cache()

def initialize(dataset_path, model_name):

    # Load the Alpaca dataset
    dataset = load_dataset(dataset_path, "all")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer, dataset

def initialize_metrics():

    # Load the evaluation metric (e.g., perplexity, BLEU)
    bleu_metric = load_metric("bleu")
    rouge_metric = load_metric("rouge")

    return bleu_metric, rouge_metric

def test(model_name, tokenizer, tokenized_datasets, bleu, rouge):
    # Load the fine-tuned model
    device='cuda'
    dtype=torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
    
    # Define the training arguments (if needed)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        no_cuda=False,
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=None,  # We'll compute metrics manually below
    )
    
    # Perform evaluation
    eval_results = trainer.evaluate()
    
    # Calculate additional metrics
    predictions = trainer.predict(tokenized_datasets["validation"]).predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Assuming the reference text is in the 'text' column of the validation set
    references = [example['text'] for example in tokenized_datasets["validation"]]
    
    # Calculate BLEU and ROUGE scores
    bleu_score = bleu.compute(predictions=decoded_preds, references=references)
    rouge_score = rouge.compute(predictions=decoded_preds, references=references)
    
    # Store the results
    evaluation_results = {
        "eval_loss": eval_results["eval_loss"],
        "bleu_score": bleu_score,
        "rouge_score": rouge_score
    }

    return evaluation_results,

if __name__ == "__main__":

    model_name = "mistral_model"

    tokenizer, tokenized_dataset = initialize("cais/mmlu", model_name)
    bleu, rouge = initialize_metrics()
    evaluation_results = test(model_name, tokenizer, tokenized_dataset, bleu, rouge)
    # Print the summary of all models
    print('-'*10)
    print("\nSummary :")
    print('-'*10)
    for model_name, metrics in evaluation_results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    print('-'*10)
