
from transformers import pipeline
import torch.nn as nn
import torch

# # Load the pretrained text model
textClassifier = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=-1
)
# After model definition
model.fc = nn.Linear(model.fc.in_features, 2)
# Save the model
torch.save(model.state_dict(), 'my_multimodal_sentement_classifier_model.pth')