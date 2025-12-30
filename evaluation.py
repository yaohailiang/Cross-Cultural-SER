import os
import glob
import tqdm
import numpy as np

import config
from toolkit.utils.read_files import *
from video_llama.evaluation.ew_metric import *
from video_llama.evaluation.wheel import *

dataset_config = {
    "aesdd": {
        "name": "AESDD",
        "emotions": ['anger', 'disgust', 'fear', 'happiness', 'sadness']
    },
    "cafe": {
        "name": "CaFE",
        "emotions": ['anger', 'disgust', 'happiness', 'neutral', 'fear', 'surprise', 'sadness']
    },
    "emodb": {
        "name": "EmoDB",
        "emotions": ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'sadness', 'neutral']
    },
    "mer2023": {
        "name": "mer2023",
        "emotions": ['neutral', 'anger', 'happiness', 'sadness', 'worry', 'surprise']
    },
    "mer2024": {
        "name": "mer2024",
        "emotions": ['neutral', 'anger', 'happiness', 'sadness', 'worry', 'surprise']
    },
    "ravdess": {
        "name": "RAVDESSEmotionalspeechaudio",
        "emotions": ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    },
    "savee": {
        "name": "SAVEE",
        "emotions": ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    },
    "shemo": {
        "name": "ShEMO",
        "emotions": ['anger', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    },
    "subesco": {
        "name": "SUBESCO",
        "emotions": ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
    },
    "urdu": {
        "name": "URDU",
        "emotions": ['anger', 'happiness', 'neutral', 'sadness']
    }
}

def func_read_datasetname(input_dir):
    for key in dataset_config:
        if key in input_dir.lower():
            dataset = dataset_config[key]["name"]
            print(f"Processing dataset: {dataset}")
            return dataset
    raise ValueError(f"Unsupported dataset or dataset not found in input directory: {input_dir}")

def func_extract_emo2idx_idx2emo(dataset):
    for conf in dataset_config.values():
        if conf["name"] == dataset:
            emos = conf["emotions"]
            emo2idx = {emo: idx for idx, emo in enumerate(emos)}
            idx2emo = {idx: emo for idx, emo in enumerate(emos)}
            return emo2idx, idx2emo
    raise ValueError(f"Unsupported dataset: {dataset}")

def func_extract_name2gt_testset(dataset):
    if dataset not in config.PATH_TO_LABEL:
        raise ValueError(f"No label path found for dataset {dataset}")
    
    label_path = config.PATH_TO_LABEL[dataset]
    label_data = np.load(label_path, allow_pickle=True)
    name2gt = label_data['name2gt'].item()

    for name, data in name2gt.items():
        gt = data['emo']
        if isinstance(gt, str):
            name2gt[name] = gt
        elif isinstance(gt, dict):
            name2gt[name] = gt.get('emo', 'unknown')
    
    print(f'Processed {len(name2gt)} samples (test set).')
    return name2gt

def main_discrete_zeroshot(input_dir="<INSERT_INPUT_DIR>", openset_output_dir="<INSERT_OUTPUT_DIR>", save_name="<INSERT_SAVE_FILE_NAME>.npz"):
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    dataset = func_read_datasetname(input_dir)
    _, idx2emo = func_extract_emo2idx_idx2emo(dataset)
    name2gt = func_extract_name2gt_testset(dataset)

    for name in name2gt:
        gt = name2gt[name]
        if not isinstance(gt, str):
            name2gt[name] = idx2emo[gt]

    results_storage = {}
    whole_mscore, whole_hitrate = [], []
    modelname = os.path.basename(input_dir)
    aggregated_name2pred = {}

    os.makedirs(openset_output_dir, exist_ok=True)
    epoch_files = glob.glob(os.path.join(input_dir, '*.npz'))
    print(f'Detected {len(epoch_files)} npz files, starting processing...')

    for idx, epoch_root in enumerate(tqdm.tqdm(epoch_files), start=1):
        if 'openset' in epoch_root:
            continue

        print(f'\n[{idx}/{len(epoch_files)}] Processing epoch file: {epoch_root}')
        openset_npz = os.path.join(
            openset_output_dir,
            os.path.basename(epoch_root[:-4]) + '-openset.npz'
        )
        if not os.path.exists(openset_npz):
            extract_openset_batchcalling(
                reason_npz=epoch_root,
                store_npz=openset_npz,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params
            )
        print(f'Openset npz saved to: {openset_npz}')
        if os.path.exists(openset_npz):
            openset_data = np.load(openset_npz, allow_pickle=True)
            if 'filenames' in openset_data and 'fileitems' in openset_data:
                filenames = openset_data['filenames']
                fileitems = openset_data['fileitems']
                for name, item in zip(filenames, fileitems):
                    aggregated_name2pred[name] = item

    print("\nAggregating predictions completed.")
    print(f"Total predictions: {len(aggregated_name2pred)}")
    hitrate, mscore = hitrate_metric_calculation(name2gt=name2gt, name2pred=aggregated_name2pred)
    whole_mscore.append(mscore)
    whole_hitrate.append(hitrate)

    new_storage = create_nested_dict(
        keys=[modelname, 'aggregated', 'whole'],
        value=[mscore, hitrate, 0]
    )
    merge_dicts(results_storage, new_storage)

    save_dir = config.DATA_DIR[dataset]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    np.savez_compressed(save_path, results_storage=results_storage)

    print(f'\n{dataset} best mscore: {max(whole_mscore):.4f}; best hitrate: {max(whole_hitrate):.4f}')
    print(f'Results saved to {save_path}')

if __name__ == '__main__':
    import fire
    fire.Fire(main_discrete_zeroshot)
