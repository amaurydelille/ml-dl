from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch
import torch.nn.functional as F
import time
from datasets import load_dataset # Import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TeacherModel:
    def __init__(self, model_name: str, device=None):
        logger.info(f"Initializing teacher model with {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device if device is not None else torch.device("cpu")
        self.model.to(self.device)
        logger.info(f"Teacher model moved to {self.device}")

    def forward(self, x):
        # Modified to accept tokenized inputs (dict) or raw text (str)
        if isinstance(x, str):
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        else:
            inputs = x # Assume already tokenized if not string
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs
    
class StudentModel:
    def __init__(self, model_name: str, device=None):
        logger.info(f"Initializing student model with {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device if device is not None else torch.device("cpu")
        self.model.to(self.device)
        logger.info(f"Student model moved to {self.device}")
        
    def forward(self, x):
        # Modified to accept tokenized inputs (dict) or raw text (str)
        if isinstance(x, str):
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        else:
            inputs = x # Assume already tokenized if not string
        outputs = self.model(**inputs)
        return outputs
    
class DistillationModel:
    def __init__(self, teacher_model: TeacherModel, student_model: StudentModel):
        self.teacher_model = teacher_model
        self.student_model = student_model
    
    def forward(self, student_inputs, teacher_inputs):
        logger.info("Forwarding input through distillation model...")
        teacher_outputs = self.teacher_model.forward(teacher_inputs)
        student_outputs = self.student_model.forward(student_inputs)
        return teacher_outputs, student_outputs

    def _distillation_loss(
        self, 
        teacher_outputs: torch.Tensor, 
        student_outputs: torch.Tensor,
        temperature: float = 1.0
    ):
        teacher_distribution = F.softmax(teacher_outputs / temperature, dim=-1)
        student_distribution = F.log_softmax(student_outputs / temperature, dim=-1)
        return F.kl_div(
            input=student_distribution, 
            target=teacher_distribution,
            reduction='batchmean'
        ) * (temperature ** 2)

class DistillationTrainer:
    def __init__(self, distillation_model: DistillationModel, teacher_model: TeacherModel, student_model: StudentModel, epochs: int, lr: float):
        self.distillation_model = distillation_model
        self.teacher_model = teacher_model
        self.student_model = student_model.model # Access the actual student model
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.lr)
    
    def run(self, train_dataloader):
        self.student_model.train() # Set student model to training mode
        # Ensure teacher is in eval mode during distillation
        self.distillation_model.teacher_model.model.eval()
        for epoch in range(self.epochs):
            total_loss = 0
            epoch_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")
            for batch_idx, batch in enumerate(epoch_progress):
                self.optimizer.zero_grad()
                
                # Move batch to appropriate device if using GPU
                student_inputs = {k: v.to(self.distillation_model.student_model.device) for k, v in batch['student_inputs'].items()}
                teacher_inputs = {k: v.to(self.distillation_model.teacher_model.device) for k, v in batch['teacher_inputs'].items()}

                teacher_outputs, student_outputs = self.distillation_model.forward(student_inputs, teacher_inputs)
                loss = self.distillation_model._distillation_loss(teacher_outputs.logits, student_outputs.logits)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                # Update progress bar with current loss
                epoch_progress.set_postfix({"loss": f"{loss.item():.4f}"})
                
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
                
def benchmark_model(model, dataloader, input_key: str):
    start_time = time.time()
    for batch in tqdm(dataloader, desc="Benchmarking", unit="batch"):
        inputs = {k: v.to(model.device) for k, v in batch[input_key].items()}
        _ = model.forward(inputs)
    end_time = time.time()
    return end_time - start_time

def get_model_size_mb(model):
    torch.save(model.model.state_dict(), "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024*1024)
    os.remove("temp_model.pt")
    return size_mb

def evaluate_model_accuracy(model, dataloader, input_key: str):
    model.model.eval() # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Accuracy", unit="batch"):
            inputs = {k: v.to(model.device) for k, v in batch[input_key].items()}
            labels = batch['labels'].to(model.device)
            outputs = model.forward(inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    return correct_predictions / total_predictions

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader

    # Set up device - use mps if available
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    # Initialize models with device
    teacher_model = TeacherModel("distilbert-base-uncased", device=device)
    student_model = StudentModel("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # Tokenizers
    teacher_tokenizer = teacher_model.tokenizer
    student_tokenizer = student_model.tokenizer

    # Custom collate function to tokenize for both teacher and student; avoids padding errors on raw text field
    def collate_fn(batch):
        texts = [example["text"] for example in batch]
        labels = torch.tensor([example["label"] for example in batch], dtype=torch.long)
        student_inputs = student_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        teacher_inputs = teacher_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {
            "student_inputs": student_inputs,
            "teacher_inputs": teacher_inputs,
            "labels": labels,
        }

    # Create dataloaders on raw dataset splits
    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=16, collate_fn=collate_fn)
    eval_dataloader = DataLoader(dataset["test"], batch_size=16, collate_fn=collate_fn)

    distillation_model = DistillationModel(teacher_model, student_model)
    trainer = DistillationTrainer(distillation_model, teacher_model, student_model, epochs=3, lr=5e-5) # Reduced epochs for quicker demonstration

    # Measure initial model sizes
    logger.info("\n--- Initial Model Sizes ---")
    teacher_size = get_model_size_mb(teacher_model)
    student_initial_size = get_model_size_mb(student_model)
    logger.info(f"Teacher Model Size: {teacher_size:.2f} MB")
    logger.info(f"Student Model Initial Size: {student_initial_size:.2f} MB")

    # Run distillation training
    logger.info("\n--- Starting Distillation Training ---")
    trainer.run(train_dataloader)
    logger.info("--- Distillation Training Finished ---")

    # Measure final student model size
    student_final_size = get_model_size_mb(student_model)
    logger.info(f"Student Model Final Size: {student_final_size:.2f} MB")

    # Benchmarking
    logger.info("\n--- Benchmarking Teacher Model ---")
    teacher_inference_time = benchmark_model(teacher_model, eval_dataloader, input_key="teacher_inputs") # Run on eval data
    logger.info(f"Teacher Model Inference Time: {teacher_inference_time:.4f} seconds")

    logger.info("\n--- Benchmarking Student Model ---")
    student_inference_time = benchmark_model(student_model, eval_dataloader, input_key="student_inputs")
    logger.info(f"Student Model Inference Time: {student_inference_time:.4f} seconds")

    # Evaluate accuracy
    logger.info("\n--- Evaluating Model Accuracy ---")
    teacher_accuracy = evaluate_model_accuracy(teacher_model, eval_dataloader, input_key="teacher_inputs")
    student_accuracy = evaluate_model_accuracy(student_model, eval_dataloader, input_key="student_inputs")
    logger.info(f"Teacher Model Accuracy: {teacher_accuracy:.4f}")
    logger.info(f"Student Model Accuracy (after distillation): {student_accuracy:.4f}")