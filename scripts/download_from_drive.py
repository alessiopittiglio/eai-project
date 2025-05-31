import gdown

# 1) URL della cartella condivisa
folder_url = "https://drive.google.com/file/d/1mZ9NNtgW_4oo9S996uQh9-SmRYaLxPnb/view?usp=sharing"

# 2) Cartella locale di destinazione (puoi cambiarla come vuoi)
output_dir = "./mia_cartella_pubblica"

# 3) Scarica tutto il contenuto della cartella (senza autenticazione)
gdown.download_folder(folder_url, output=output_dir, quiet=False)

print("Scaricamento completato in:", output_dir)
