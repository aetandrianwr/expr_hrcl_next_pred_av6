"""
Configuration file for GeoLife next-location prediction.
"""
import torch


class Config:
    """Configuration class for the project."""
    
    # Data paths
    DATA_DIR = "data/geolife"
    TRAIN_FILE = "geolife_transformer_7_train.pk"
    VAL_FILE = "geolife_transformer_7_validation.pk"
    TEST_FILE = "geolife_transformer_7_test.pk"
    
    # Random seed
    SEED = 42
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Vocabulary sizes (from data analysis - max value + 1)
    VOCAB_SIZES = {
        'location': 1187,
        'user': 46,
        'weekday': 7,
        'time_slot': 1440,  # minutes in a day
        'duration': 2880,
        'diff': 8,
        'h3_res5': 174,
        'h3_res6': 278,
        'h3_res7': 463,
        'h3_res8': 795,  # Fixed: was 785
        's2_level11': 314,
        's2_level13': 673,  # Fixed: was 666
        's2_level14': 926,  # Fixed: was 923
        's2_level15': 1249,  # Fixed: was 1246
    }
    
    # Model hyperparameters
    D_MODEL = 80  # embedding dimension (reduced from 96)
    N_HEADS = 5
    N_LAYERS = 3
    D_FF = 160  # feedforward dimension
    DROPOUT = 0.25
    MAX_SEQ_LEN = 52
    
    # Hierarchical location encoding
    USE_HIERARCHICAL_LOCATION = True
    HIERARCHY_LEVELS = ['h3_res5', 'h3_res6', 'h3_res7', 'h3_res8']
    
    # Training hyperparameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.002  # Increased for OneCycleLR
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 80
    PATIENCE = 20
    GRAD_CLIP = 1.0
    
    # Learning rate schedule
    WARMUP_STEPS = 500
    USE_COSINE_SCHEDULE = True
    
    # Label smoothing
    LABEL_SMOOTHING = 0.1
    
    # Checkpoint and logging
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    SAVE_EVERY = 1  # Save checkpoint every N epochs
    
    # Evaluation
    EVAL_BATCH_SIZE = 512

    def __repr__(self):
        return '\n'.join(f'{k}: {v}' for k, v in self.__dict__.items() if not k.startswith('_'))
