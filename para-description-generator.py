import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_pitch(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = np.max(pitches, axis=0)
    pitch = pitch[pitch > 0]
    pitch_std = np.std(pitch)
    return pitch, pitch_std

def extract_energy(y):
    rms = librosa.feature.rms(y=y)[0]
    energy_std = np.std(rms)
    return rms, energy_std

def extract_speech_rate(y, sr):
    intervals = librosa.effects.split(y, top_db=30)
    speech_duration = sum(end - start for start, end in intervals) / sr
    total_duration = len(y) / sr
    return speech_duration / total_duration

def pitch_std_to_description(pitch_std, pitch_quantiles):
    if pitch_std <= pitch_quantiles[0]:
        return "Low pitch fluctuation"
    elif pitch_std <= pitch_quantiles[1]:
        return "Medium pitch fluctuation"
    else:
        return "High pitch fluctuation"

def energy_std_to_description(energy_std, energy_quantiles):
    if energy_std <= energy_quantiles[0]:
        return "Low energy fluctuation"
    elif energy_std <= energy_quantiles[1]:
        return "Medium energy fluctuation"
    else:
        return "High energy fluctuation"

def speech_rate_to_description(speech_rate, sorted_rates):
    total = len(sorted_rates)
    idx = np.searchsorted(sorted_rates, speech_rate, side='right')
    split_slow = total // 3
    split_medium = 2 * total // 3
    if idx <= split_slow:
        return "Slow speech rate"
    elif idx <= split_medium:
        return "Medium speech rate"
    else:
        return "Fast speech rate"

def batch_process_audio_files(folder_path, output_csv):
    wav_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".wav")]
    all_data = []
    for file_path in tqdm(wav_files, desc="Extracting features"):
        try:
            y, sr = load_audio(file_path)
            pitch, pitch_std = extract_pitch(y, sr)
            rms, energy_std = extract_energy(y)
            speech_rate = extract_speech_rate(y, sr)
            all_data.append({
                "path": file_path,
                "pitch_std": pitch_std,
                "energy_std": energy_std,
                "speech_rate": speech_rate
            })
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    df = pd.DataFrame(all_data)
    pitch_quantiles = np.percentile(df["pitch_std"], [33, 66])
    energy_quantiles = np.percentile(df["energy_std"], [33, 66])
    sorted_speech_rates = np.sort(df["speech_rate"].values)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating descriptions"):
        try:
            filename = os.path.basename(row["path"]).replace(".wav", "")
            pitch_desc = pitch_std_to_description(row["pitch_std"], pitch_quantiles)
            energy_desc = energy_std_to_description(row["energy_std"], energy_quantiles)
            speech_desc = speech_rate_to_description(row["speech_rate"], sorted_speech_rates)
            description = (
                f"This audio shows: {pitch_desc}, "
                f"{energy_desc}, and {speech_desc}."
            )
            results.append({
                "name": filename,
                "description": description
            })
        except Exception as e:
            print(f"Failed to generate description for {row['path']}: {e}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    speech_counts = result_df["description"].str.extract(r"and (.*?)\.", expand=False).value_counts()
    print("\nDistribution summary:")
    print(speech_counts)

if __name__ == "__main__":
    folder_path = "<INSERT_AUDIO_FOLDER_PATH>"
    output_csv = "<INSERT_OUTPUT_CSV_PATH>"
    batch_process_audio_files(folder_path, output_csv)
