# import torch
# import torch.nn as nn
# from torchvision import models
# from transformers import AlbertModel, AlbertTokenizer
# from PIL import Image
# from torchvision import transforms




# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.mlp(self.avg_pool(x))
#         max_out = self.mlp(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out) * x

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv(x)
#         return self.sigmoid(x)

# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x):
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x) * x
#         return x


# class DenseNetCBAMFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(DenseNetCBAMFeatureExtractor, self).__init__()
#         self.densenet = models.densenet121(pretrained=True)
#         self.densenet.classifier = nn.Identity()
#         self.cbam = CBAM(in_channels=1024)
#         self.fc_reduction = nn.Linear(1024, 512)  # Change output dimension to 512

#     def forward(self, x):
#         features = self.densenet.features(x)
#         refined_features = self.cbam(features)
#         feature_vector = torch.mean(refined_features, dim=[2, 3])
#         reduced_vector = self.fc_reduction(feature_vector)
#         return reduced_vector




# class ALBERTBiLSTMFeatureExtractor(nn.Module):
#     def __init__(self, embedding_dim=512, lstm_hidden_size=256):
#         super(ALBERTBiLSTMFeatureExtractor, self).__init__()
#         self.albert = AlbertModel.from_pretrained("albert-base-v2", output_hidden_states=True)
#         self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
#         self.fc_projection = nn.Linear(768, embedding_dim)
#         self.bilstm = nn.LSTM(
#             input_size=embedding_dim,
#             hidden_size=lstm_hidden_size,
#             num_layers=1,
#             bidirectional=True,
#             batch_first=True
#         )

#     def forward(self, input_text):
#         inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=150)
#         albert_outputs = self.albert(**inputs)
#         contextual_embeddings = albert_outputs.last_hidden_state
#         projected_embeddings = self.fc_projection(contextual_embeddings)
#         bilstm_outputs, _ = self.bilstm(projected_embeddings)
#         text_feature_vector = torch.mean(bilstm_outputs, dim=1)  # Global average pooling
#         return text_feature_vector



# class GatedCrossAttentionNetwork(nn.Module):
#     def __init__(self, feature_dim=512, hidden_dim=256, output_dim=3, dropout_rate=0.3):
#         super(GatedCrossAttentionNetwork, self).__init__()

#         # Linear layers for cross-attention
#         self.W_q = nn.Linear(feature_dim, hidden_dim)
#         self.W_k = nn.Linear(feature_dim, hidden_dim)
#         self.W_v = nn.Linear(feature_dim, hidden_dim)

#         # Cross-Interaction Layers
#         self.cross_interaction_text = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.cross_interaction_image = nn.Linear(hidden_dim * 2, hidden_dim)

#         # Gating Mechanism
#         self.gate_text = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.gate_image = nn.Linear(hidden_dim * 2, hidden_dim)

#         # Fusion Layer
#         self.fc_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.classifier = nn.Linear(hidden_dim, output_dim)



#     def forward(self, text_features, image_features):
#         """ 
#         text_features: H_T (unimodal text)
#         image_features: H_I (unimodal image)
#         """

#         # Step 1: Apply Cross-Attention (Text attends to Image)
#         Q = self.W_q(text_features)  # Query from text
#         K = self.W_k(image_features)  # Key from image
#         V = self.W_v(image_features)  # Value from image

#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
#         attention_weights = F.softmax(attention_scores, dim=-1)
#         attended_text = torch.matmul(attention_weights, V)  # C_IT (text attended to image)

#         # Step 2: Apply Cross-Attention (Image attends to Text)
#         Q_img = self.W_q(image_features)  # Query from image
#         K_txt = self.W_k(text_features)  # Key from text
#         V_txt = self.W_v(text_features)  # Value from text

#         attention_scores_img = torch.matmul(Q_img, K_txt.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q_img.size(-1), dtype=torch.float32))
#         attention_weights_img = F.softmax(attention_scores_img, dim=-1)
#         attended_image = torch.matmul(attention_weights_img, V_txt)  # C_TI (image attended to text)

#         # Step 3: Cross Interaction Layer
#         cross_text = torch.tanh(self.cross_interaction_text(torch.cat([attended_text, text_features], dim=-1)))
#         cross_image = torch.tanh(self.cross_interaction_image(torch.cat([attended_image, image_features], dim=-1)))

#         # Step 4: Gating Mechanism
#         gate_text = torch.sigmoid(self.gate_text(torch.cat([cross_text, text_features], dim=-1)))
#         gate_image = torch.sigmoid(self.gate_image(torch.cat([cross_image, image_features], dim=-1)))

#         final_text_vector = gate_text * cross_text + (1 - gate_text) * text_features
#         final_image_vector = gate_image * cross_image + (1 - gate_image) * image_features

#         # Step 5: Fusion (Final Fused Vector)
#         final_fused_vector = self.fc_fusion(torch.cat([final_text_vector, final_image_vector], dim=-1))
#         final_fused_vector = self.dropout(final_fused_vector)

#         # Step 6: Classification
#         logits = self.classifier(final_fused_vector)

#         return final_text_vector, final_image_vector, final_fused_vector, logits
        



# # app.py (Windows version)
# import streamlit as st
# import torch
# from transformers import pipeline
# from torchvision import transforms
# from PIL import Image
# import torch.nn as nn

# st.title("Multimodal Sentiment Analysis")


# text_classifier = pipeline(
#     'sentiment-analysis',
#     model='distilbert-base-uncased-finetuned-sst-2-english',
#     device=-1  
# )




# text_input = st.text_area("Enter text:")
# st.success("Text input✅")
# if text_input:
#     text_result = text_classifier(text_input)
#     text_confidence = text_result[0]['score']
#     st.write(f"**Sentiment:** {text_result[0]['label']}")
#     st.write(f"**Confidence:** {text_result[0]['score']:.2f}")















# model = my_model(weights=ResNet18_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, 2)  
# model.eval()

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     input_tensor = preprocess(image).unsqueeze(0)
#     with torch.no_grad():
#         output = model(input_tensor)
#         probs = torch.nn.functional.softmax(output[0], dim=0)
#         confidence, pred = torch.max(probs, 0)

#         # Assign weights
#         image_weight = 0.4
#         text_weight = 0.6

#         # Calculate weighted average
#         weighted_avg = (image_weight * confidence.item() + text_weight * text_confidence) / (image_weight + text_weight)

#         if (pred.item() == 1 and text_result[0]['label'] == 'POSITIVE' and weighted_avg > 0.7):
#             st.write("Case 1", weighted_avg)
#             label = "Positive"
#         elif (pred.item() == 0 and text_result[0]['label'] == 'NEGATIVE' and weighted_avg > 0.8):
#             st.write("Case 2", weighted_avg)
#             label = "Negative"
#         elif (pred.item() == 1 and text_result[0]['label'] == 'POSITIVE' and weighted_avg < 0.8):
#             st.write("Case 3", weighted_avg)
#             label = "Neutral"
#         elif (pred.item() == 0 and text_result[0]['label'] == 'NEGATIVE' and weighted_avg < 0.8):
#             st.write("Case 4", weighted_avg)
#             label = "Neutral"
#         elif (((pred.item() == 1 and text_result[0]['label'] == 'NEGATIVE') and text_weight > 0.8) or 
#             ((pred.item() == 0 and text_result[0]['label'] == 'POSITIVE') and confidence.item() > 0.8)):
#             st.write("Case 5", weighted_avg)
#             label = "Negative"
#         elif (((pred.item() == 1 and text_result[0]['label'] == 'NEGATIVE') and text_weight < 0.6) or 
#             ((pred.item() == 0 and text_result[0]['label'] == 'POSITIVE') and confidence.item() < 0.6)):
#             st.write("Case 6", weighted_avg)
#             label = "Neutral"
#         else:
#             st.write("Case 7", weighted_avg)
#             label = "Negative"

#         st.write(f"**Sentiment:** {label}")
#         st.write(f"**Confidence:** {confidence.item():.2f}")


# app.py (Windows version)
import streamlit as st
import torch
from transformers import pipeline
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import torch.nn as nn

st.title("Multimodal Sentiment Analysis")

text_classifier = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=-1  
)

user_input = st.text_area("Enter text:")
if user_input:
    text_result = text_classifier(user_input)
    text_confidence = text_result[0]['score']
    # st.write(f"**Sentiment:** {text_result[0]['label']}")
    # st.write(f"**Confidence:** {text_result[0]['score']:.2f}")




# Load pretrained ResNet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence, pred = torch.max(probs, 0)

        # Assign weights
        image_weight = 0.4
        text_weight = 0.6

        # Calculate weighted average
        weighted_avg = (image_weight * confidence.item() + text_weight * text_confidence) / (image_weight + text_weight)

        if (pred.item() == 1 and text_result[0]['label'] == 'POSITIVE' and weighted_avg > 0.7):
            # st.write("Case 1", weighted_avg)
            label = "Positive"
        elif (pred.item() == 0 and text_result[0]['label'] == 'NEGATIVE' and weighted_avg > 0.8):
            # st.write("Case 2", weighted_avg)
            label = "Negative"
        elif (pred.item() == 1 and text_result[0]['label'] == 'POSITIVE' and weighted_avg < 0.8):
            # st.write("Case 3", weighted_avg)
            label = "Neutral"
        elif (pred.item() == 0 and text_result[0]['label'] == 'NEGATIVE' and weighted_avg < 0.8):
            # st.write("Case 4", weighted_avg)
            label = "Neutral"
        elif (((pred.item() == 1 and text_result[0]['label'] == 'NEGATIVE') and text_weight > 0.8) or 
            ((pred.item() == 0 and text_result[0]['label'] == 'POSITIVE') and confidence.item() > 0.8)):
            # st.write("Case 5", weighted_avg)
            label = "Negative"
        elif (((pred.item() == 1 and text_result[0]['label'] == 'NEGATIVE') and text_weight < 0.6) or 
            ((pred.item() == 0 and text_result[0]['label'] == 'POSITIVE') and confidence.item() < 0.6)):
            # st.write("Case 6", weighted_avg)
            label = "Neutral"
        else:
            # st.write("Case 7", weighted_avg)
            label = "Negative"

        st.write(f"**Sentiment:** {label}")
        # st.write(f"**Confidence:** {confidence.item():.2f}")