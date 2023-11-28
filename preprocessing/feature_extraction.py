import librosa
import librosa.feature
import time
import os
import pandas as pd
import numpy as np


def process_music_files_with_segments(root_dir, num_segments=None):
    """
    Process audio files in specified directory, extracting features with timing.
    Processes each file entirely or in segments, as specified.

    Args:
        root_dir (str): Root directory containing genre subdirectories with audio files.
        num_segments (int, optional): Number of segments to divide each audio file into.
                                      If None or <=0, processes the full audio file.

    Returns:
        pandas.DataFrame: DataFrame containing extracted features from audio files.
    """
    columns = ['track id', 'genre', 'segment', 'chromagram', 'rms', 'spectral_centroid', 'spectral_bandwidth',
               'rolloff', 'zero_crossing_rate', 'harmonics', 'tempo'] + [f'mfcc_{i + 1}' for i in range(20)]
    feature_data_list = []
    start_time = time.time()

    for genre in ['rock', 'pop', 'electronic', 'classical']:
        genre_dir = os.path.join(root_dir, genre)
        genre_start_time = time.time()

        print(f"Processing {genre}")
        track_id = 1

        for i in range(1, 101):
            filename = f"{i}.mp3"
            file_path = os.path.join(genre_dir, filename)
            if os.path.exists(file_path):
                y_full, sr = librosa.load(file_path)
                total_samples = len(y_full)
                file_start_time = time.time()

                if num_segments is None or num_segments <= 0:
                    features = extract_features(y_full, sr)
                    feature_series = pd.Series([track_id, genre, 0] + list(features), index=columns)
                    feature_data_list.append(feature_series)
                else:
                    samples_per_segment = total_samples // num_segments
                    for seg_num in range(num_segments):
                        start_sample = seg_num * samples_per_segment
                        end_sample = start_sample + samples_per_segment
                        y_segment = y_full[start_sample:end_sample]
                        features = extract_features(y_segment, sr)
                        feature_series = pd.Series([track_id, genre, seg_num + 1] + list(features), index=columns)
                        feature_data_list.append(feature_series)

                file_process_time = time.time() - file_start_time
                total_process_time = time.time() - start_time
                print(
                    f"Processed '{genre}/{filename}' in {file_process_time:.2f} sec. Total time: {total_process_time:.2f} sec.")

                track_id += 1

        genre_process_time = time.time() - genre_start_time
        print(
            f"Finished processing {genre} in {genre_process_time:.2f} sec. Total time: {time.time() - start_time:.2f} sec.")

    feature_data = pd.concat(feature_data_list, axis=1).transpose()
    feature_data.columns = columns

    return feature_data


def extract_features(y, sr):
    """
    Extract various audio features from an audio.

    This function computes a set of features from a given audio segment, including
    chroma-stft, root-mean-square (rms), spectral centroid, spectral bandwidth,
    spectral rolloff, zero-crossing rate, harmonics, tempo, and Mel-frequency cepstral coefficients (MFCCs).
    Args:
        y (numpy.ndarray): The audio time series (audio segment).
        sr (int): The sampling rate of the audio time series.
    Returns:
        numpy.ndarray: A numpy array containing the extracted audio features.
        The features include mean values of chroma-stft, rms, spectral centroid,
        spectral bandwidth, spectral rolloff, zero-crossing rate, harmonics, tempo,
        and the mean of each MFCC (total 20 MFCCs).
    """
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    features = np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(
        zcr), np.mean(harmonic), tempo
    features = np.hstack((features, mfcc.mean(axis=1)))
    return features
