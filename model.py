def train():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import get_peft_model, LoraConfig, TaskType
    from datasets import Dataset

    # Load model and tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check available device
    model_name = "Qwen/Qwen2-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)  # Move model to device

    tokenizer.pad_token = tokenizer.eos_token 

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[f"layers.{i}.self_attn.{mat}_proj" for i in range(28) for mat in ["k","v"]],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Load and preprocess data
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text_data = f.read()

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)

    # Create a dataset
    dataset = Dataset.from_dict({"text": [text_data]})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to device
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            print(loss.item())
            return (loss, outputs) if return_outputs else loss

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./shakespeare_model",
        num_train_epochs=100,  # Increase the number of epochs
        per_device_train_batch_size=1,  # Adjust the batch size
        save_steps=500,  # Save the model more frequently
        save_total_limit=3,
        learning_rate=1e-3,  # Adjust the learning rate
        logging_dir='./logs',
        logging_steps=10,
    )

    # Create Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train the model
    trainer.train()

    model_path = "./shakespeare_model"
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    model.config.save_pretrained(model_path)

    # Generate text
    input_text = "To be or not to be,"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device) 
    output = model.generate(input_ids, 
                            max_new_tokens=20,
                            num_return_sequences=1, 
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=torch.ones_like(input_ids)
                            )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

def eval():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Assuming you're using a model that can be loaded with AutoModelForCausalLM
    # and AutoTokenizer for simplicity
    model_path = "./shakespeare_model"

    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Ensure the model is in evaluation mode
    model.eval()

    # Assuming you're using a device like 'cuda' or 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare the input text
    input_text = "I doth die"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate text
    output = model.generate(input_ids, 
                            max_new_tokens=80,
                            num_return_sequences=1, 
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=torch.ones_like(input_ids)
                            )

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    eval()
