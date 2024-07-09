from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset, DataLoader

# Specify the model name
model_name = "Qwen/Qwen2-1.5B"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

####################################################################################################

# Read and tokenize data from shakespeare.txt
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

# Tokenize the text data
tokenized_data = tokenizer(text_data, truncation=True, padding=True, return_tensors="pt")


####################################################################################################

# Define dataset class
class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

# Create dataset and dataloader
dataset = MyDataset(tokenized_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 3)

####################################################################################################

# Fine-tune the model with PEFT
num_epochs = 3  # Example: 3 epochs
progress_bar = tqdm(range(num_epochs), desc="Training Epochs")
for epoch in progress_bar:
    for batch in dataloader:
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**batch)

        # Calculate loss (example: mean squared error)
        loss = torch.mean((outputs.last_hidden_state - batch["input_ids"]) ** 2)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

# After fine-tuning, apply PEFT
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["encoder.layer.*.attention.self.query", "encoder.layer.*.attention.self.key", "encoder.layer.*.attention.self.value"]
)

# Apply LoRA to the model
lora_model = get_peft_model(model, lora_config)

