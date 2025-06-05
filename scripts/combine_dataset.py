import os
import glob
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- 1) Modifica questi valori con il percorso corretto della tua cartella principale ---
BASE_DIR = "data_frames"      # cartella che contiene DFDC/ e ffpp/
DATASETS = ["DFDC", "ffpp"]   # i due dataset da unire
SPLITS = ["train", "val", "test"]
CLASSES = ["REAL", "FAKE"]    # lowercase come nella tua struttura

# Cartella di destinazione "combined"
DEST_BASE = os.path.join(BASE_DIR, "combined", "_".join(DATASETS))


def _copy_one_folder(task):
    """
    Funzione worker: prende in ingresso una tupla (src_folder, dst_folder),
    e copia l'intera directory src_folder in dst_folder. Se fallisce,
    restituisce una stringa di errore, altrimenti None.
    """
    src_folder, dst_folder = task
    try:
        # Usa copytree per copiare l'intera struttura di src_folder in dst_folder
        # dirs_exist_ok=True permetterà di fondere contenuto se la cartella esiste già
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
    except Exception as e:
        return f"❗ Errore copiando {src_folder} → {dst_folder}: {e}"
    return None


def merge_datasets_parallel():
    # -------------------------------------------------------------------
    # 2) Raccogliamo TUTTE le coppie (src_folder, dst_folder) da copiare:
    # -------------------------------------------------------------------
    all_tasks = []  # ciascun elemento: (src_folder, dst_folder)

    for ds in DATASETS:
        for split in SPLITS:
            for cls in CLASSES:
                src_class_dir = os.path.join(BASE_DIR, ds, split, cls)
                if not os.path.isdir(src_class_dir):
                    continue

                # Ogni subfolder in src_class_dir corrisponde a una "cartella di video"
                for folder_name in os.listdir(src_class_dir):
                    src_folder = os.path.join(src_class_dir, folder_name)
                    if not os.path.isdir(src_folder):
                        continue

                    # Per evitare collisioni di nomi (se due dataset hanno lo stesso folder_name),
                    # prefissiamo con il nome del dataset
                    dst_folder_name = f"{ds}_{folder_name}"
                    dst_folder = os.path.join(DEST_BASE, split, cls, dst_folder_name)

                    all_tasks.append((src_folder, dst_folder))

    # -------------------------------------------------------------------
    # 3) Creiamo tutte le directory di destinazione (train/val/test • real/fake)
    # -------------------------------------------------------------------
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(DEST_BASE, split, cls), exist_ok=True)

    # -------------------------------------------------------------------
    # 4) Copia parallela con ProcessPoolExecutor e barra di avanzamento tqdm
    # -------------------------------------------------------------------
    n_items = len(all_tasks)
    if n_items == 0:
        print("Nessun folder trovato da copiare. Controlla la struttura di DFDC/ e ffpp/.")
        return

    max_workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map restituisce un iteratore di risultati da _copy_one_folder
        for err in tqdm(
            executor.map(_copy_one_folder, all_tasks),
            total=n_items,
            desc="Copying all folders",
        ):
            if err:
                print(err)

    print(f"✔️  Fatto! Copiate {n_items} cartelle in '{DEST_BASE}'.")


if __name__ == "__main__":
    merge_datasets_parallel()
    
    # Analyze for each real/fake folder inside train/val/test the number of images
    for split in SPLITS:
        for class_sub in CLASSES:
            folder_path = os.path.join(DEST_BASE, split, class_sub)
            if os.path.isdir(folder_path):
                folders = glob.glob(os.path.join(folder_path, '*'))
                print(f"Found {len(folders)} folders in {folder_path}.")
                total=0
                for folder in folders:
                    if os.path.isdir(folder):
                        images = glob.glob(os.path.join(folder, '*.png'))
                        total += len(images)
                        # print(f"Folder: {os.path.basename(folder)}, Images: {len(images)}")
                print(f"Total images in {folder_path}: {total}")
