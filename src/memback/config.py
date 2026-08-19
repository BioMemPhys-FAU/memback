import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "model"

_ENV_ROOT = os.environ.get("MEMBACK_ROOT")
if _ENV_ROOT:
    _root = Path(_ENV_ROOT).expanduser().resolve()
    DATA_DIR = _root / "data"
    MODEL_DIR = _root / "model"

model_path = str(MODEL_DIR / "memback_0.1.1_state_dict.pt")
hdb_path = str(DATA_DIR / "hdb" / "lipid.hdb")
charmm_ff_path = str(DATA_DIR / "forcefields" / "charmm36-feb2026_cgenff-5.0.ff")
rtp_path = str(Path(charmm_ff_path) / "lipid.rtp")
itp_db_path = str(DATA_DIR / "charmm_lipid_itps")
itp_m3_db_path = str(DATA_DIR / "m3_itps")
map_path = str(DATA_DIR / "maps" / "all_maps.map")
bond_map_path = str(DATA_DIR / "maps" / "all_maps.bnd")


def check_data_files():
    """Return a list of required data paths that are missing (empty if all present)."""
    required = {
        "model checkpoint": model_path,
        "hydrogen database": hdb_path,
        "CHARMM36 force field": charmm_ff_path,
        "CHARMM lipid .itp database": itp_db_path,
        "AA->CG map": map_path,
        "CG bond map": bond_map_path,
    }
    return [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]


bead_classes = {'C': 0, 'N': 1, 'P': 2, 'Q': 3, 'W': 4, 'X': 5, 'D': 6, 'UNK': 7}

bead_sizes = {'R':0, 'S': 1, 'T':2, 'UNK':3}
# In armstrong
bead_radii = {'R':2.35, 'S':2.05, 'T':1.7, 'W': 2.35}

max_atom_number = 6

elements_in_lipid = {'C': 0, 'O': 1, 'P': 2, 'N': 3, 'H': 4, 'S': 5, 'UNK': 6}

sterochemistry_map = {'cis':0, 'trans':1, 'S':2, 'R':3, 'gauche':4, 'UNK':5}

martini3_water_names = ["W"]

martini3_ion_names = ["ION"]

martini3_excluded_residues = ["W", "ION"]

# Martini3 ion bead name, Charmm ion residue name and atom name
martini3_to_charmm_ions = {
    "NA":  ("SOD", "SOD"),   # sodium
    "CL":  ("CLA", "CLA"),   # chloride
    "CA":  ("CAL", "CAL"),   # calcium
    # "BR":  ("BR",  "BR"),   # bromide: not in Charmm
    # "IOD":  ("IOD",  "IOD"),   # iodine: not in Charmm
    # "ZN":  ("ZN2", "ZN"),   # zinc: not in Martini3
    # "CS":  ("CES", "CES"),   # cesium: not in Martini3
    # "RB":  ("RUB", "RUB"),   # rubidium: not in Martini3
    # "BA":  ("BAR", "BAR"),   # barium: not in Martini3
    # "CD":  ("CD2", "CD"),   # cadmium: not in Martini3
    # "MG":  ("MG",  "MG"),   # magnesium: not in Martini3
    # "K":   ("POT", "POT"),    # potassium: not in Martini3
    # "LI":  ("LIT", "LIT"),   # lithium: not in Martini3
}

# Contains only residues where the name differs between force fields.
martini3_to_charmm_lipids = {
    # Sterols
    "CHOL": "CHL1",
    # SM lipids
    "KSM": "ASM",
    "XSM": "LSM",
    # di-C10 tails
    # "DJPA": "DDPA", # Not in Charmm
    "DJPC": "DDPC",
    # "DJPE": "DDPE", # Not in Charmm
    # "DJPG": "DDPG", # Not in Charmm
    # "DJPS": "DDPS", # Not in Charmm

    # di-C12 tails
    "DUPA": "DLPA",
    "DUPC": "DLPC",
    "DUPE": "DLPE",
    "DUPG": "DLPG",
    "DUPS": "DLPS",
    # di-C18:2 tails
    # "DLPA": "DLIPA", # Not in Charmm
    "DLPC": "DLIPC",
    "DLPE": "DLIPE",
    # "DLPG": "DLIPG", # Not in Charmm
    # "DLPS": "DLIPS", # Not in Charmm
    # "DLPI": "DLIPI", # Not in Martini3

    # di-C22:6 tails
    # "DDPA": "DDOPA", # Not in Charmm
    "DDPC": "DDOPC",
    "DDPE": "DDOPE",
    # "DDPG": "DDOPG", # Not in Charmm
    "DDPS": "DDOPS",
    # C16:0/18:3 tails
    "PFPA" : "PLEPA",
    "PFPC" : "PLEPC",
    "PFPE" : "PLEPE",
    "PFPG" : "PLEPG",
    "PFPS" : "PLEPS",
    # C16:0/22:6 tails
    # "PDPA" : "PDOPA", # Not in Charmm
    "PDPC" : "PDOPC",
    "PDPE" : "PDOPE",
    # "PDPG" : "PDOPG", # Not in Charmm
    # "PDPS" : "PDOPS", # Not in Charmm

    # C18:1/22:6 tails
    # "ODPA" : "ODOPA", # Not in Charmm
    "ODPC" : "ODOPC",
    # "ODPE" : "ODOPE", # Not in Charmm
    # "ODPG" : "ODOPG", # Not in Charmm
    # "ODPS" : "ODOPS", # Not in Charmm

    # C16:0/18:3 tails
    "LFPA" : "LLPA",
    "LFPC" : "LLPC",
    "LFPE" : "LLPE",
    "LFPG" : "LLPG",
    "LFPS" : "LLPS",
    # CL lipids
    "TMCL": "TMCL2",
    "TOCL": "TOCL2",
    # POPIs
    "POP1": "POPI13",
    "POP4": "POPI14",
    "POP5": "POPI15",
    "POP2": "POPI2C",  # Could be POPI2D
    "POP6": "POPI24",  # Could be POPI25
    "POP7": "POPI2A",  # Could be POPI2B
    "POP3": "POPI33",  # Could be POPI34 or POPI35
    # SAPIs
    "SAP1": "SAPI13",
    "SAP4": "SAPI14",
    "SAP5": "SAPI15",
    "SAP2": "SAPI2C",  # Could be SAPI2D
    "SAP6": "SAPI24",  # Could be SAPI25
    "SAP7": "SAPI2A",  # Could be SAPI2B
    "SAP3": "SAPI33",  # Could be SAPI34 or SAPI35
}


# CHARMM name may not always have a unique Martini3 counterpart.
charmm_to_martini3 = {v: k for k, v in martini3_to_charmm_lipids.items()}
