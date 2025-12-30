import os
import argparse
import librosa
import torch
import pandas as pd
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

torch.manual_seed(1234)

def func_read_key_from_csv(csv_path, key):
    values = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        if key not in row:
            values.append("")
        else:
            value = row[key]
            if pd.isna(value): 
                value = ""
            values.append(value)
    return values

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subtitle', action='store_true', default=False, help='whether to add subtitle during inference')
    args = parser.parse_args()

    model_path = "<INSERT_MODEL_PATH>"
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map={"": "cuda:0"})
    model.tie_weights()

    sampling_rate = processor.feature_extractor.sampling_rate

    step_root = "<INSERT_ROOT_PATH>"
    audio_root = os.path.join(step_root, "<INSERT_AUDIO_FOLDER>")
    reason_path = os.path.join(step_root, "<INSERT_REASON_CSV>")
    tran_path = os.path.join(step_root, "<INSERT_SUBTITLE_CSV>")

    process_names = func_read_key_from_csv(reason_path, 'name')
    print(f'process names: {len(process_names)}')

    name2eng = {}
    names = func_read_key_from_csv(tran_path, 'name')
    engs = func_read_key_from_csv(tran_path, 'english')
    for (name, eng) in zip(names, engs):
        name2eng[name] = eng

    save_root = os.path.join("<INSERT_OUTPUT_DIR>", f"output-qwen2-eng-{'subtitle' if args.subtitle else 'nosubtitle'}")
    os.makedirs(save_root, exist_ok=True)

    for ii, name in enumerate(process_names):
        print(f'Processing {ii + 1}/{len(process_names)}: {name}')
        wav_path = os.path.join(audio_root, name + '.wav')
        subtitle = name2eng[name]

        conversation = [
            {'role': 'system', 'content': 'You are an expert in emotion analysis.'},
            {'role': 'user', 'content': [
                {"type": "audio", "audio_url": f"file://{wav_path}"},
                {"type": "text", "text": (
                    f"The subtitle for this audio is: {subtitle}. As an expert in the field of emotions, please focus on the acoustic information in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the audio."                    
                    if args.subtitle else
                    "As an expert in the field of emotions, please focus on the acoustic information in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the audio."
                )}
            ]}
        ]

        audios = [librosa.load(wav_path, sr=sampling_rate)[0]]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=sampling_rate)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        model = model.to(device)

        generate_ids = model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(response)

        save_path = os.path.join(save_root, name + '.npy')
        np.save(save_path, response)
