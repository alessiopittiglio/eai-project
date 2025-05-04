OUTPUT_NOTEBOOK="./models/Xception3d/output_notebook.ipynb"

if [ ! -f "$OUTPUT_NOTEBOOK" ]; then
    nohup papermill notebook.ipynb "$OUTPUT_NOTEBOOK" > log.txt 2>&1 &
else
    echo "$OUTPUT_NOTEBOOK already exists. Exiting."
fi
