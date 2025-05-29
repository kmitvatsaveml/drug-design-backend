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

# Updated ViT Model Classes
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embedding_dim=48):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.patcher(x).flatten(2).transpose(1, 2)
        return x

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=48, num_heads=4, dropout=0.1):
        super(MultiheadSelfAttentionBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        norm_x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(norm_x, norm_x, norm_x)
        x = x + self.dropout(attn_output)
        return x

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=48, mlp_size=3072, dropout=0.1):
        super(MLPBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.layer_norm(x)
        x = x + self.mlp(norm_x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=48, num_heads=4, mlp_size=3072, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.msa_block = MultiheadSelfAttentionBlock(
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
        x = self.msa_block(x)
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
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dim)
        )
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                dropout=dropout
            ) for _ in range(num_transformer_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.class_embedding.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding
        for block in self.encoder_blocks:
            x = block(x)
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