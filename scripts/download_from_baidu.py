#!/usr/bin/env python3
from bypy import ByPy

def download_shared_folder(share_id: str, pwd: str, remote_path: str, local_path: str):
    """
    Scarica il contenuto di una cartella condivisa su Baidu Pan (share link).
    - share_id: l’ID che compare nell’URL di condivisione (dopo /s/ e prima del ?).
    - pwd: password della condivisione (se impostata).
    - remote_path: "/" per la root della condivisione, oppure "/nome_sottocartella".
    - local_path: percorso locale in cui salvare i file.
    """

    bp = ByPy()
    # la prima volta ByPy() fa riferimento a ~/.bypy/bypy.json per i token; se manca,
    # bisogna lanciare bypy login da terminale una volta sola.

    # Chiama il metodo share_download di bypy
    # attenzione: la firma è share_download(share_id, share_pwd, path_remoto, path_locale)
    ret = bp.share_download(share_id, pwd, remote_path, local_path)
    if ret == 0:
        print(f"[OK] Scaricamento completato in: {local_path}")
    else:
        print(f"[ERRORE] share_download restituito codice {ret}")

if __name__ == "__main__":
    # Parametri del tuo link
    SHARE_ID = "1NAMUHcZvsIm7l6hMHeEQjQ"
    PASSWORD = "ogjn"
    REMOTE_PATH = "/"   # radice della cartella condivisa. Se vuoi solo una sottocartella, p.es. "/rgb"
    LOCAL_DIR = "./baidu_rgb_download"

    # Crea la cartella locale se non esiste
    import os
    os.makedirs(LOCAL_DIR, exist_ok=True)

    download_shared_folder(SHARE_ID, PASSWORD, REMOTE_PATH, LOCAL_DIR)
