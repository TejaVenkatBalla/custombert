import torch
import joblib
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.nn.functional as F

class MultiOutputBERT(nn.Module):
    def __init__(self, num_categories, num_subcategories):
        super(MultiOutputBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.category_classifier = nn.Linear(self.bert.config.hidden_size, num_categories)
        self.subcategory_classifier = nn.Linear(self.bert.config.hidden_size, num_subcategories)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        category_logits = self.category_classifier(pooled_output)
        subcategory_logits = self.subcategory_classifier(pooled_output)
        return category_logits, subcategory_logits


class CybercrimeClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_data = joblib.load(model_path)

        # Initialize and load the model structure
        num_categories = len(self.model_data["category_encoder"].classes_)
        num_subcategories = len(self.model_data["subcategory_encoder"].classes_)
        self.model = MultiOutputBERT(num_categories, num_subcategories)
        self.model.load_state_dict(self.model_data["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize tokenizer and encoders
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.category_encoder = self.model_data["category_encoder"]
        self.subcategory_encoder = self.model_data["subcategory_encoder"]

    def predict(self, text):
        # Preprocess the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            category_logits, subcategory_logits = self.model(input_ids, attention_mask)

        # Get predictions
        category_pred = torch.argmax(F.softmax(category_logits, dim=1), dim=1).item()
        subcategory_pred = torch.argmax(F.softmax(subcategory_logits, dim=1), dim=1).item()

        # Decode the predicted labels
        category = self.category_encoder.inverse_transform([category_pred])[0]
        subcategory = self.subcategory_encoder.inverse_transform([subcategory_pred])[0]

        # Calculate confidence
        category_confidence = F.softmax(category_logits, dim=1).max().item()
        subcategory_confidence = F.softmax(subcategory_logits, dim=1).max().item()

        return {
            "category": category,
            "category_confidence": category_confidence,
            "subcategory": subcategory,
            "subcategory_confidence": subcategory_confidence
        }

    @staticmethod
    def load_model(model_path):
        return CybercrimeClassifier(model_path)
