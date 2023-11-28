from sound_exploration import sound_utils
from preprocessing import dataset_preparation, feature_extraction

# file_path = 'resources/emotify/emotifymusic/rock/1.mp3'
# data, sr = sound_utils.load_audio(file_path)
# sound_utils.plot_waveform(data, sr)
# sound_utils.plot_fourier_transform(data, sr)
# sound_utils.plot_spectrogram(data, sr)
# sound_utils.plot_mel_spectrogram(data, sr)
# sound_utils.plot_mfcc(data, sr)

dataset_path = "resources/emotify/data.csv"
emotify_music_path = "resources/emotify/emotifymusic"

# dataset = dataset_preprocessing.preprocess_emotify_dataset(dataset_path)
# dataset_preprocessing.save_preprocessed_dataset(dataset, "resources/emotify/preprocessed_dataset.csv")

processed_data_with_segments = feature_extraction.process_music_files_with_segments(emotify_music_path, 20)
processed_data_with_segments.to_csv("resources/emotify/music_emotion_features_20_seg.csv")
