import pandas as pd
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
import re
import torch

train_data = pd.read_csv("samsum-train.csv")
val_data = pd.read_csv("samsum-validation.csv")

# random sampling
train_data = train_data.sample(n=4000, random_state=42).reset_index(drop=True)
val_data = val_data.sample(n=500, random_state=42).reset_index(drop=True)

# Data pre-processing

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

train_data["dialogue"] = train_data["dialogue"].apply(clean_data)
train_data["summary"] = train_data["summary"].apply(clean_data)

val_data["dialogue"] = val_data["dialogue"].apply(clean_data)
val_data["summary"] = val_data["summary"].apply(clean_data)

# Tokenize
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# raw data => tokenized inputs for fine-tuning

def tokenize(data):
    inputs = tokenizer(data["dialogue"], padding="max_length", max_length=512, truncation=True)
    targets = tokenizer(data["summary"], padding="max_length", max_length=150, truncation=True)

    inputs["labels"] = targets["input_ids"] # token ids => add to input as labels
    return inputs

train_dataset = train_data.apply(tokenize, axis=1).tolist()
val_dataset = val_data.apply(tokenize, axis=1).tolist()

# Working with our Model

# NLP => generation task

model = T5ForConditionalGeneration.from_pretrained("t5-small")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device: ", device)
model.to(device)

# Training Arguments

training_args = TrainingArguments(
    output_dir = "./results",

    num_train_epochs=6,
    weight_decay=0.01,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    eval_strategy="epoch",
    save_strategy="epoch",

    warmup_steps=500
    # 0 => lr default
)

# train the model
trainer.train()

# model load => fine-tune => save the model

model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Test the core logic for summarization

def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary

test_dialogue = """ 
Reporter: In today's technology news, artificial intelligence continues to expand rapidly across industries, from healthcare to finance and education. Recent reports suggest that AI adoption has significantly increased over the past few years.

Reporter: Companies are investing heavily in machine learning systems to automate tasks, improve decision-making, and enhance customer experiences. However, this growth has also raised questions about job displacement and ethical concerns.

Expert: AI systems are becoming more capable due to advances in deep learning and access to large datasets. These models can now perform complex tasks such as language understanding, image recognition, and even code generation.

Expert: At the same time, there are valid concerns about bias in AI models, as they often reflect the data they are trained on. Ensuring fairness and transparency is becoming a key area of research.

Reporter: Governments and organizations are beginning to introduce regulations to guide the development and deployment of AI technologies. The goal is to balance innovation with accountability.

Expert: Another challenge is explainability. Many modern AI systems, especially deep neural networks, operate as “black boxes,” making it difficult to understand how decisions are made.

Reporter: Experts also highlight the importance of responsible AI development, including data privacy, security, and long-term societal impact.

Expert: Looking ahead, collaboration between researchers, policymakers, and industry leaders will be crucial to ensure that AI systems are developed and used in a safe and beneficial way.
"""

summary = summarize_dialogue(test_dialogue)

print("Summary: ", summary)