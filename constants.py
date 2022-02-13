from pathlib import Path

import seaborn as sns

MODELS_OBJECTS_PATH = Path("models/objects")
MODELS_INTRA_TABLES = Path("models/intra_tabels")
MODELS_PATH = Path("models")
DATA_PATH = Path("data")
MODELS_GRAPHS = Path("models/graphs")
# TRANSFER_SIZE_LIST = [0, 500, 1000, 1500]
TRANSFER_SIZE_LIST = [100, 200, 300, 400, 500]
DATASETS = ['cow1', 'worm1', 'worm2', 'human1', 'human2', 'human3', 'mouse1', 'mouse2']

ACC_HEATMAP_DICT = {"cmap": "RdBu_r", "square": True, "linewidths": 3, "annot": True, "vmin": 0.5, "vmax": 1,
                    "cbar_kws": {'label': 'ACC'}}

F1_HEATMAP_DICT = {"cmap": sns.diverging_palette(145, 300, s=60, as_cmap=True), "square": True, "linewidths": 3,
                   "annot": True, "vmin": 0, "vmax": 1,
                   "cbar_kws": {'label': 'F1 Score'}}

IMPORTANT_FEATURES = ['miRNAPairingCount_Seed_GU', 'miRNAMatchPosition_1', 'miRNAPairingCount_Total_GU',
                      'Energy_MEF_local_target', 'MRNA_Target_G_comp', 'MRNA_Target_GG_comp', 'miRNAMatchPosition_4',
                      'miRNAMatchPosition_5', 'miRNAPairingCount_Seed_bulge_nt', 'miRNAPairingCount_Seed_GC',
                      'miRNAMatchPosition_2', 'miRNAPairingCount_Seed_mismatch', 'miRNAPairingCount_X3p_GC',
                      'Seed_match_compact_interactions_all']
SEQUANCE_FEATURES = ['mRNA_start', 'label', 'mRNA_name',
                     'target sequence', 'microRNA_name', 'miRNA sequence', 'full_mrna']

FEATURES_TO_DROP = ['mRNA_start', 'label', 'mRNA_name', 'target sequence', 'microRNA_name', 'miRNA sequence',
                    'full_mrna',
                    'canonic_seed', 'duplex_RNAplex_equals', 'non_canonic_seed', 'site_start', 'num_of_pairs',
                    'mRNA_end', 'constraint']

GOOD_MODEL_SHAP_FEATURES = ['Energy_MEF_local_target', 'Energy_MEF_Duplex', 'miRNAMatchPosition_1',
                            'miRNAMatchPosition_9', 'miRNAPairingCount_Total_GU']

VS4_REG_DICT = {"worm": ["worm1", "worm2"], "cow": ["cow1"], "human": ["human1", "human2", "human3"],
                "mouse": ["mouse1", "mouse2"]}
