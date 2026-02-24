import os

# show image filename in UI (debug)
SHOW_IMAGE_NAME = False

# path setup
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# data files
LOCALIZE_JSON = os.path.join(DATA_DIR, 'localize_small.json')
REPORT_METADATA_JSON = os.path.join(DATA_DIR, 'radgame_report.json')

# image directories
LOCALIZE_IMAGE_BASE = os.path.join(BASE_DIR, 'local_sampled_old')
REPORT_IMAGE_BASE = os.path.join(BASE_DIR, 'rex_sampled_additional_cases')

# ---------------------------------------------------------------------------
# Report scoring backend
# ---------------------------------------------------------------------------
# Set to "gpt" (default — uses OpenAI o3) or "medgemma" (uses finetuned
# MedGemma-4B with LoRA adapter via the CRIMSON scoring framework).
REPORT_SCORER = os.environ.get("REPORT_SCORER", "medgemma")

# MedGemma / CRIMSON model paths (only used when REPORT_SCORER == "medgemma")
MEDGEMMA_BASE_MODEL = os.environ.get(
    "MEDGEMMA_BASE_MODEL",
    "google/medgemma-4b-it",
)
MEDGEMMA_LORA_PATH = os.environ.get(
    "MEDGEMMA_LORA_PATH",
    "/n/lw_groups/hms/dbmi/yu/lab/sir855/CRIMSON/data/finetuned_medgemma/checkpoints/checkpoint-21400",
)
MEDGEMMA_CACHE_DIR = os.environ.get(
    "MEDGEMMA_CACHE_DIR",
    "/n/lw_groups/hms/dbmi/yu/lab/sir855/CRIMSON/models/medgemma",
)
MEDGEMMA_MAX_NEW_TOKENS = int(os.environ.get("MEDGEMMA_MAX_NEW_TOKENS", "4096"))

# ---------------------------------------------------------------------------
# Test mode — when True, localize and report always show these fixed cases
# first (in order) so you can anticipate ground truth during development.
# ---------------------------------------------------------------------------
TEST_MODE = os.environ.get("RADGAME_TEST_MODE", "true").lower() in ("1", "true", "yes")

TEST_LOCALIZE_CASES = [
    "1059090736492172890440690893294928964_qnqec4.png",       # 2 findings: interstitial infiltrate + pleural thickening
    "3337838038438312879412295722317051049_2_m1m86n.png",     # 8 findings: pleural thickening, volume loss, cardiomegaly, pacemaker, etc.
    "13224141948247255586026463437846237918_zt30no.png",       # 1 finding: bibasal bronchiectasis (good localisation target)
]

TEST_REPORT_CASES = [
    # 3 positive findings: cardiomegaly, tortuous aorta, T7 compression fracture
    "pGRDN2M6YB40JVKXT_aGRDNPXT6PESJ8PN9_s1.2.826.0.1.3680043.8.498.16966475801975550915919249203392254470",
    # 3 positive findings: left base atelectasis, right pleural effusion, thoracic degenerative changes
    "pGRDN0INSFKP87VBD_aGRDNNF6XJLWHO9E1_s1.2.826.0.1.3680043.8.498.36718788597288494251583097359735026737",
    # 3 positive findings: calcified mediastinal nodes, RUL scarring/volume loss, calcified granuloma
    "pGRDN0T0I2H9WNP7I_aGRDN6RC4UVXXAA7T_s1.2.826.0.1.3680043.8.498.27747643274283460905315749455107466194",
]