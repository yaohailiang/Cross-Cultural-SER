import os
import glob
import shutil
import numpy as np
import pandas as pd
from evaluation import func_read_batch_calling_model, clue_merge_batchcalling

def find_npz_subdir(dataset_path):
    for entry in sorted(os.listdir(dataset_path)):
        full = os.path.join(dataset_path, entry)
        if os.path.isdir(full) and glob.glob(os.path.join(full, '*.npz')):
            return full
    return None

def make_output_dir_name(nosub_basename: str) -> str:
    if 'nosubtitle-npz' in nosub_basename:
        return nosub_basename.replace('nosubtitle-npz', 'nosubtitle+cap-npz')
    else:
        return nosub_basename + '+cap-npz'

if __name__ == "__main__":
    root_dir = "<INSERT_ROOT_NOSUB_DIR>"
    subtitle_csv_dir = "<INSERT_SUBTITLE_CSV_DIR>"

    models = ['Flamingo', 'GLM4', 'Qwen-audio', 'Qwen2-audio', 'SALMOON']
    datasets = [
        'AESDD', 'CaFE', 'EmoDB', 'mer2023', 'mer2024',
        'RAVDESS-Speech', 'SAVEE', 'ShEMO', 'SUBESCO', 'URDU'
    ]

    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')

    for model in models:
        for dataset in datasets:
            ds_path = os.path.join(root_dir, model, dataset)
            if not os.path.isdir(ds_path):
                print(f"[warn] model={model}, dataset={dataset} directory not found, skipping")
                continue

            nosub_dir = find_npz_subdir(ds_path)
            if nosub_dir is None:
                print(f"[warn] {model}/{dataset} no .npz subdir found, skipping")
                continue

            nosub_basename = os.path.basename(nosub_dir)
            out_basename = make_output_dir_name(nosub_basename)
            out_dir = os.path.join(ds_path, out_basename)
            os.makedirs(out_dir, exist_ok=True)

            csv_path = os.path.join(subtitle_csv_dir, f"{dataset}.csv")
            if not os.path.isfile(csv_path):
                print(f"[warn] Subtitle CSV not found: {csv_path}, skipping {model}/{dataset}")
                continue

            df = pd.read_csv(csv_path).drop_duplicates(subset='name', keep='first')
            name2subtitle = dict(zip(df['name'], df['english']))

            for fp in glob.glob(os.path.join(nosub_dir, '*.npz')):
                name = os.path.splitext(os.path.basename(fp))[0]
                save_fp = os.path.join(out_dir, f"{name}.npz")

                if os.path.exists(save_fp):
                    print(f"[skip] already processed: {model}/{dataset}/{name}")
                    continue

                data = np.load(fp, allow_pickle=True)
                key = data.files[0]
                reason = data[key].tolist()

                if name not in name2subtitle:
                    print(f"[warn] {model}/{dataset}/{name} no subtitle, copying original")
                    shutil.copy(fp, save_fp)
                    continue

                subtitle = name2subtitle[name]
                clue_merge_batchcalling(
                    name2reason={name: reason},
                    store_npz=save_fp,
                    name2subtitle={name: subtitle},
                    llm=llm,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params
                )
                print(f"[ok] fused and saved: {model}/{dataset}/{name}")

    print("All models and datasets processed.")
