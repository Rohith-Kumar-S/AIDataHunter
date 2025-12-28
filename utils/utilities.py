
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
from captum.attr import LayerIntegratedGradients
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_text(text_list):
  encoded = tokenizer(
        text_list,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
  return encoded["input_ids"], encoded["attention_mask"]

def load_bert_ai_text_detector():
    class BERTModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        def forward(self, input_ids, attention_mask):
            out =  self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

    model = BERTModel()
    # Load best model
    states = torch.load("model.pth",
                        map_location="cuda")

    device = torch.device("cuda")

    model.load_state_dict(states["model"])
    model.to(device)
    return model

def visualize_attributions(model, input_ids, attention_mask, tokenizer, predicted_class=0):
    lig = LayerIntegratedGradients(model, model.bert_model.bert.embeddings)
    baseline_ids = torch.zeros_like(input_ids)

    attributions = lig.attribute(
        input_ids,
        baselines=baseline_ids,
        target=predicted_class,
        additional_forward_args=(attention_mask,)
    )
    # Aggregate to per-token scores
    token_scores = attributions.norm(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    
    # Filter out PAD and special tokens, normalize scores
    mask = [t not in ["[PAD]", "[CLS]", "[SEP]"] for t in tokens]
    filtered_tokens = [t for t, m in zip(tokens, mask) if m]
    filtered_scores = token_scores[mask]
    
    # Normalize to 0-1 range
    p5, p95 = np.percentile(filtered_scores, [5, 95])
    clipped = np.clip(filtered_scores, p5, p95)
    normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)
    
    # Build HTML with color-coded tokens
    html = '<div style="font-family: monospace; line-height: 1.8; padding: 10px;">'
    for token, score in zip(filtered_tokens, normalized):
        # Red intensity based on score
        r, g, b = 255, int(255 * (1 - score)), int(255 * (1 - score))
        color = f"rgb({r},{g},{b})"
        
        # Clean up subword tokens
        display_token = token.replace("##", "")
        html += f'<span style="background-color: {color}; color: black; padding: 2px 4px; margin: 1px; border-radius: 3px;">{display_token}</span> '
    
    html += '</div>'
    return html

def predict(text):
    device = torch.device("cuda")
    model = load_bert_ai_text_detector()
    input_ids, attention_mask = tokenize_text([text])
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        y_preds = model(input_ids, attention_mask)
        probs = torch.softmax(y_preds, dim=1)
        predictions = torch.argmax(probs, dim=1)
        res = {"confidence": {0: float(probs.cpu().numpy()[0][0]), 1: float(probs.cpu().numpy()[0][1])},
                    "prediction": predictions.cpu().numpy()[0]
        }
        
        return res, model, input_ids, attention_mask, tokenizer