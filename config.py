import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_WHEEL_ROOT = os.path.join(CONFIG_DIR, 'emotion-wheel')

PATH_TO_LLM = {
    'Qwen25': os.path.join(CONFIG_DIR, 'data', 'llm', 'Qwen25'),
}

DATA_DIR = {   
    'SUBESCO': os.path.join(CONFIG_DIR, 'data', 'SUBESCO'),
    'SAVEE':   os.path.join(CONFIG_DIR, 'data', 'SAVEE'),
    'RAVDESS': os.path.join(CONFIG_DIR, 'data', 'RAVDESS-Speech'),
    'AESDD':   os.path.join(CONFIG_DIR, 'data', 'AESDD'),
    'EmoDB':   os.path.join(CONFIG_DIR, 'data', 'EmoDB'),
    'URDU':    os.path.join(CONFIG_DIR, 'data', 'URDU'),
    'CaFE':    os.path.join(CONFIG_DIR, 'data', 'CaFE'),
    'ShEMO':   os.path.join(CONFIG_DIR, 'data', 'ShEMO'),
    'mer2023': os.path.join(CONFIG_DIR, 'data', 'mer2023'),
    'mer2024': os.path.join(CONFIG_DIR, 'data', 'mer2024'),
}

PATH_TO_LABEL = {
    'ShEMO':    os.path.join(DATA_DIR['ShEMO'], 'label.npz'),
    'SAVEE':    os.path.join(DATA_DIR['SAVEE'], 'label.npz'),
    'RAVDESS':  os.path.join(DATA_DIR['RAVDESS-Speech'], 'label.npz'),
    'AESDD':    os.path.join(DATA_DIR['AESDD'], 'label.npz'),
    'EmoDB':    os.path.join(DATA_DIR['EmoDB'], 'label.npz'),
    'URDU':     os.path.join(DATA_DIR['URDU'], 'label.npz'),
    'CaFE':     os.path.join(DATA_DIR['CaFE'], 'label.npz'),
    'SUBESCO':  os.path.join(DATA_DIR['SUBESCO'], 'label.npz'),
    'mer2023':  os.path.join(DATA_DIR['mer2023'], 'label.npz'),
    'mer2024':  os.path.join(DATA_DIR['mer2024'], 'label.npz'),
}
