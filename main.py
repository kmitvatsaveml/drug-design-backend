# import logging
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import torch
# import torch.nn as nn
# from rdkit import Chem
# from rdkit.Chem import AllChem, rdFingerprintGenerator, Draw
# from io import BytesIO
# from PIL import Image
# import base64
# import joblib
# import os
# from sklearn.preprocessing import LabelEncoder

# # Setup logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
# )
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI()

# # CORS middleware to allow frontend communication
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Input data model for SMILES
# class InputData(BaseModel):
#     SMILES: str

# # ViT Model Definition
# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels=3, patch_size=4, embedding_dim=48):
#         super().__init__()
#         self.patcher = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
#         self.flatten = nn.Flatten(2, 3)

#     def forward(self, x):
#         x = self.patcher(x)
#         x = self.flatten(x)
#         return x.permute(0, 2, 1)

# class MultiheadSelfAttentionBlock(nn.Module):
#     def __init__(self, embedding_dim=48, num_heads=4, attn_dropout=0):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(embedding_dim)
#         self.multihead_attn = nn.MultiheadAttention(
#             embed_dim=embedding_dim,
#             num_heads=num_heads,
#             dropout=attn_dropout,
#             batch_first=True
#         )

#     def forward(self, x):
#         x_norm = self.layer_norm(x)
#         attn_output, _ = self.multihead_attn(x_norm, x_norm, x_norm, need_weights=False)
#         return attn_output

# class MLPBlock(nn.Module):
#     def __init__(self, embedding_dim=48, mlp_size=3072, dropout=0.1):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(embedding_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, mlp_size),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_size, embedding_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.mlp(self.layer_norm(x))

# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, embedding_dim=48, num_heads=4, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
#         super().__init__()
#         self.msa_block = MultiheadSelfAttentionBlock(embedding_dim, num_heads, attn_dropout)
#         self.mlp_block = MLPBlock(embedding_dim, mlp_size, mlp_dropout)

#     def forward(self, x):
#         x = self.msa_block(x) + x
#         x = self.mlp_block(x) + x
#         return x

# class ViT(nn.Module):
#     def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=3,
#                  embedding_dim=48, num_transformer_layers=12, num_heads=4,
#                  mlp_size=3072, attn_dropout=0, mlp_dropout=0.1, embedding_dropout=0.1):
#         super().__init__()
#         assert img_size % patch_size == 0
#         self.num_patches = (img_size // patch_size) ** 2
#         self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
#         self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
#         self.embedding_dropout = nn.Dropout(embedding_dropout)
#         self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
#         self.encoder_blocks = nn.ModuleList([
#             TransformerEncoderBlock(embedding_dim, num_heads, mlp_size, mlp_dropout, attn_dropout)
#             for _ in range(num_transformer_layers)
#         ])
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(embedding_dim),
#             nn.Linear(embedding_dim, num_classes)
#         )

#     def forward(self, x):
#         x = self.patch_embedding(x)
#         batch_size = x.shape[0]
#         class_token = self.class_embedding.expand(batch_size, -1, -1)
#         x = torch.cat((class_token, x), dim=1)
#         x = self.embedding_dropout(x + self.position_embedding)
#         for block in self.encoder_blocks:
#             x = block(x)
#         return self.classifier(x[:, 0])

# # Function to load ViT model and label encoder
# def load_model_vit(model_path, label_path):
#     try:
#         # Load label encoder
#         label_data = joblib.load(label_path)
#         logger.debug(f"Loaded label data: {label_data}, type: {type(label_data)}")
        
#         # Check if label_data is a dictionary
#         if isinstance(label_data, dict) and 'label_encoder' in label_data:
#             label_encoder = label_data['label_encoder']
#             if isinstance(label_encoder, LabelEncoder) and hasattr(label_encoder, 'classes_'):
#                 le_classes = label_encoder.classes_
#             else:
#                 logger.warning("LabelEncoder in pickle file has no classes_, using default classes")
#                 le_classes = np.array(['active', 'inactive', 'intermediate'])
#         else:
#             logger.warning("Unexpected label data format, using default classes")
#             le_classes = np.array(['active', 'inactive', 'intermediate'])
        
#         logger.info(f"Label encoder classes: {le_classes}, type: {type(le_classes)}")
#         if not isinstance(le_classes, (list, np.ndarray)):
#             raise ValueError(f"Expected list or numpy.ndarray for le_classes, got {type(le_classes)}")
#         if len(le_classes) != 3:
#             raise ValueError(f"Expected 3 classes, got {len(le_classes)}")
        
#         # Initialize model
#         model = ViT(
#             img_size=32,
#             patch_size=4,
#             in_channels=3,
#             num_classes=3,
#             embedding_dim=48,
#             num_transformer_layers=12,
#             num_heads=4,
#             mlp_size=3072,
#             attn_dropout=0,
#             mlp_dropout=0.1,
#             embedding_dropout=0.1
#         )
        
#         # Load state dict
#         model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#         model.eval()
#         logger.info("ViT model loaded successfully")
        
#         return model, le_classes
#     except Exception as e:
#         logger.error(f"Failed to load ViT model or label encoder: {e}")
#         raise

# # Function to convert Morgan fingerprint to image
# def morgan_to_image(x):
#     try:
#         logger.debug(f"Input fingerprint shape: {x.shape}")
#         if len(x) != 1024:
#             raise ValueError(f"Expected 1024-bit fingerprint, got {len(x)}")
#         flat = np.pad(x, (0, 3 * 32 * 32 - len(x)), constant_values=0)
#         image = flat.reshape(3, 32, 32)
#         logger.debug(f"Output image shape: {image.shape}")
#         return image
#     except Exception as e:
#         logger.error(f"Error in morgan_to_image: {e}")
#         raise

# # Function to generate 2D structure image from SMILES
# def smiles_to_base64_image(smiles: str):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             logger.error(f"Invalid SMILES: {smiles}")
#             return None
#         AllChem.Compute2DCoords(mol)
#         img = Draw.MolToImage(mol, size=(300, 300))
#         buffered = BytesIO()
#         img.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         logger.debug("2D structure image generated successfully")
#         return img_str
#     except Exception as e:
#         logger.error(f"Error generating 2D structure for SMILES {smiles}: {e}")
#         return None

# # Use environment variables for model paths
# MODEL_PATH = os.getenv("MODEL_PATH", "/opt/model/vit_model.pth")
# LABEL_PATH = os.getenv("LABEL_PATH", "/opt/model/vit_model.pkl")
# logger.debug(f"Loading ViT model from {MODEL_PATH} and label encoder from {LABEL_PATH}")
# vit, le_classes = load_model_vit(MODEL_PATH, LABEL_PATH)

# # ViT prediction endpoint
# @app.post('/predict/vit')
# async def predict(data: InputData):
#     try:
#         smiles = data.SMILES.strip()
#         logger.info(f"Processing SMILES for ViT prediction: {smiles}")
        
#         # Convert SMILES to molecule
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             logger.error(f"Invalid SMILES: {smiles}")
#             raise HTTPException(status_code=400, detail="Invalid SMILES")
        
#         # Generate Morgan fingerprint
#         try:
#             morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
#             fingerprint = morgan_gen.GetFingerprint(mol)
#             logger.debug(f"Fingerprint generated: {fingerprint}")
#         except Exception as e:
#             logger.error(f"Error generating fingerprint: {e}")
#             raise HTTPException(status_code=400, detail="Error during fingerprint generation")
        
#         fp_array = np.array(fingerprint, dtype=np.float32)
#         logger.debug(f"Fingerprint array shape: {fp_array.shape}, max value: {fp_array.max()}")
#         fp_array = fp_array / (fp_array.max() + 1e-6)
        
#         # Convert to image and predict
#         image = morgan_to_image(fp_array)
#         image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
#         logger.debug(f"Image tensor shape: {image_tensor.shape}")
        
#         with torch.no_grad():
#             output = vit(image_tensor)
#             logger.debug(f"Model output shape: {output.shape}")
#             predicted_class_idx = output.argmax(1).item()
#             logger.debug(f"Predicted class index: {predicted_class_idx}")
        
#         # Map prediction to label
#         try:
#             if predicted_class_idx >= len(le_classes):
#                 raise ValueError(f"Predicted index {predicted_class_idx} out of range for le_classes {le_classes}")
#             predicted_label = le_classes[predicted_class_idx]
#             logger.debug(f"Predicted label: {predicted_label}")
#         except Exception as e:
#             logger.error(f"Error mapping predicted label: {e}")
#             raise ValueError(f"Label mapping failed: {e}")
        
#         # Generate 2D structure image
#         structure_image = smiles_to_base64_image(smiles)
#         if structure_image is None:
#             logger.warning(f"Failed to generate structure image for SMILES {smiles}")
        
#         result = {
#             "smiles": smiles,
#             "activity": predicted_label,
#             "image": structure_image
#         }
#         logger.info(f"ViT prediction result: {result}")
#         return result

#     except HTTPException as http_exc:
#         raise http_exc
#     except Exception as e:
#         logger.error(f"Internal server error: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {e}")




import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, Draw
from io import BytesIO
from PIL import Image
import base64
import joblib
import os
import requests
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class InputData(BaseModel):
    SMILES: str

# Download model files from GitHub
MODEL_DIR = "/tmp/models"
MODEL_PATH = os.path.join(MODEL_DIR, "vit_model.pth")
LABEL_PATH = os.path.join(MODEL_DIR, "vit_model.pkl")

def download_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for file_name in ["vit_model.pth", "vit_model.pkl"]:
        url = f"https://github.com/kmitvatsaveml/drug-design-backend/raw/main/models/{file_name}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(MODEL_DIR, file_name), "wb") as f:
                f.write(response.content)
            logger.debug(f"Downloaded {file_name} to {MODEL_DIR}")
        else:
            logger.error(f"Failed to download {file_name}: Status {response.status_code}")
            raise Exception(f"Failed to download {file_name}")

try:
    download_model_files()
except Exception as e:
    logger.error(f"Model download failed: {e}")
    raise

# ViT Model Classes
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embedding_dim=48):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        return x

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=48, num_heads=4, dropout=0.1):
        super(MultiheadSelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=48, mlp_size=3072, dropout=0.1):
        super(MLPBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.mlp(x)
        x = self.norm(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=48, num_heads=4, mlp_size=3072, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=dropout
        )

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=3,
        embedding_dim=48,
        num_transformer_layers=12,
        num_heads=4,
        mlp_size=3072,
        dropout=0.1
    ):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim
        )
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                dropout=dropout
            ) for _ in range(num_transformer_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x

def load_model_vit(model_path, label_path):
    try:
        label_data = joblib.load(label_path)
        logger.debug(f"Loaded label data: {label_data}, type: {type(label_data)}")
        if isinstance(label_data, dict) and 'label_encoder' in label_data:
            label_encoder = label_data['label_encoder']
            le_classes = label_encoder.classes_ if isinstance(label_encoder, LabelEncoder) else np.array(['active', 'inactive', 'intermediate'])
        else:
            logger.warning("Unexpected label data format, using default classes")
            le_classes = np.array(['active', 'inactive', 'intermediate'])
        logger.info(f"Label encoder classes: {le_classes}")
        model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=3, embedding_dim=48, num_transformer_layers=12, num_heads=4, mlp_size=3072)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info("ViT model loaded successfully")
        return model, le_classes
    except Exception as e:
        logger.error(f"Failed to load ViT model or label encoder: {e}")
        raise

def morgan_to_image(morgan_fp, size=(32, 32)):
    morgan_array = np.array(morgan_fp, dtype=np.uint8).reshape(32, 64)
    morgan_array = morgan_array[:, :32]
    image = Image.fromarray(morgan_array * 255)
    image = image.convert('RGB')
    return image

def smiles_to_base64_image(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        img = Draw.MolToImage(mol, size=(300, 300))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        logger.error(f"Error generating image for SMILES {smiles}: {e}")
        raise

logger.debug(f"Loading ViT model from {MODEL_PATH} and label encoder from {LABEL_PATH}")
vit, le_classes = load_model_vit(MODEL_PATH, LABEL_PATH)

@app.post('/predict/vit')
async def predict(data: InputData):
    try:
        smiles = data.SMILES
        logger.debug(f"Received SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES string: {smiles}")
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        morgan_fp = fpgen.GetFingerprint(mol)
        morgan_fp = np.array(morgan_fp, dtype=np.float32)
        
        image = morgan_to_image(morgan_fp)
        image = image.resize((32, 32))
        image_array = np.array(image) / 255.0
        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            outputs = vit(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        predicted_label = le_classes[predicted_class]
        base64_image = smiles_to_base64_image(smiles)
        
        logger.info(f"Prediction for SMILES {smiles}: {predicted_label}")
        return {
            "smiles": smiles,
            "activity": predicted_label,
            "image": base64_image
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}