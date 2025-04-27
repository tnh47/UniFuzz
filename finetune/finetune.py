from datasets import Dataset, load_dataset
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer
)
from unsloth import FastLanguageModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
import torch
import json

def prepare_dataset(rag_db_path):
    dataset = []
    
    vector_store = FAISS.load_local(
        rag_db_path,
        OllamaEmbeddings(model="nomic-embed-text"),
        allow_dangerous_deserialization=True
    )
    
    for doc in vector_store.docstore._dict.values():
        entry = {
            "instruction": "Generate fuzzing inputs for detected vulnerabilities",
            "input": doc.page_content,
            "output": json.dumps(extract_vulnerability_patterns(doc.metadata))
        }
        dataset.append(entry)
    
    return Dataset.from_list(dataset)

def train():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="codellama/CodeLlama-7b-hf",
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    dataset = prepare_dataset("./RAG/vector_db")
    dataset = dataset.map(lambda x: {
        "text": f"""### Instruction: {x['instruction']}
### Input Contract: {x['input']}
### Expected Output: {x['output']}"""
    })

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        output_dir="./outputs",
        save_strategy="epoch"
    )

    trainer = FastLanguageModel.get_trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=4096,
        packing=True
    )

    trainer.train()
    model.save_pretrained("./finetune/finetuned_model")

if __name__ == "__main__":
    train()
