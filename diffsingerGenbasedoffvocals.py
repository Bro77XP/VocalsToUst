import whisper
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import torch


def fetch_lyrics_from_genius(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        lyrics_div = soup.find('div', class_='lyrics')
        if lyrics_div:
            return lyrics_div.get_text()
        return None
    except Exception:
        return None


def load_whisper_model(model_name: str = "base", device: str = "auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    try:
        print(f"Loading Whisper model '{model_name}' on {device}...")
        model = whisper.load_model(model_name, device=device)
        print("Whisper model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load Whisper model: {e}")
        print("Falling back to CPU...")
        try:
            model = whisper.load_model(model_name, device="cpu")
            print("Whisper model loaded on CPU successfully!")
            return model
        except Exception as e2:
            print(f"Failed to load Whisper model on CPU: {e2}")
            raise e2


def transcribe_audio(audio_path, model=None, language=None):
    if model is None:
        model = load_whisper_model("base")
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
        task="transcribe",
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        best_of=5,
        beam_size=5,
    )
    return result


def extract_pitch(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    hop_length = 256
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        hop_length=hop_length, sr=sr, frame_length=2048, fill_na=np.nan
    )
    f0 = librosa.util.fix_length(f0, size=len(y) // hop_length + 1)
    f0 = pd.Series(f0).interpolate().values
    return f0, voiced_flag, sr, hop_length


def detect_onsets(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=onset_env, units='frames'
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


def snap_to_nearest_onset(time, onset_times, max_offset=0.03):
    if len(onset_times) == 0:
        return time
    diffs = np.abs(onset_times - time)
    nearest_idx = np.argmin(diffs)
    nearest_onset = onset_times[nearest_idx]
    if abs(nearest_onset - time) <= max_offset:
        return nearest_onset
    return time


def snap_segments_to_onsets(segments, onset_times):
    print("Snapping word boundaries to nearby onsets...")
    for segment in segments:
        new_start = snap_to_nearest_onset(segment['start'], onset_times, max_offset=0.03)
        new_end = snap_to_nearest_onset(segment['end'], onset_times, max_offset=0.03)
        if new_end <= new_start:
            new_end = segment['end']
        if new_start >= new_end:
            new_start = segment['start']
        segment['start'] = new_start
        segment['end'] = new_end
    return segments


def fix_overlapping_segments(segments):
    for i in range(len(segments) - 1):
        if segments[i]['end'] > segments[i + 1]['start']:
            midpoint = (segments[i]['end'] + segments[i + 1]['start']) / 2
            segments[i]['end'] = midpoint
            segments[i + 1]['start'] = midpoint
    return segments


def detect_voiced_segments(audio_path, sr=22050, hop_length=256):
    y, sr = librosa.load(audio_path, sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    silence_threshold = np.percentile(rms, 25)

    voiced_frames = rms > silence_threshold
    voiced_segments = []
    start_frame = None
    for i, is_voiced in enumerate(voiced_frames):
        if is_voiced and start_frame is None:
            start_frame = i
        elif not is_voiced and start_frame is not None:
            end_frame = i
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
            voiced_segments.append((start_time, end_time))
            start_frame = None

    if start_frame is not None:
        end_time = librosa.frames_to_time(len(voiced_frames), sr=sr, hop_length=hop_length)
        voiced_segments.append((start_time, end_time))

    return voiced_segments


def generate_ust(lyrics_segments, pitch_data, output_path, audio_duration):
    f0, voiced_flag, sr, hop_length = pitch_data
    lyrics_segments = sorted(lyrics_segments, key=lambda x: x['start'])

    REFERENCE_TEMPO = 120.0
    ref_tps = 480 * REFERENCE_TEMPO / 60
    ticks_per_measure = 480 * 4

    notes = []

    for segment in lyrics_segments:
        start_time = segment['start']
        end_time = segment['end']
        lyric = segment.get('text', segment.get('word', '')).strip()
        if not lyric:
            continue

        gap = start_time - (notes[-1]['end_time'] if notes else 0.0)
        if gap > 0.05:
            notes.append({
                'type': 'rest',
                'start_time': start_time - gap,
                'end_time': start_time,
                'length_sec': gap,
            })

        duration = end_time - start_time
        if duration <= 0:
            duration = 0.05

        if 'tone' in segment:
            note_num = segment['tone']
        else:
            start_frame = int(start_time * sr / hop_length)
            end_frame = int(end_time * sr / hop_length)
            if start_frame < len(f0) and end_frame <= len(f0):
                segment_f0 = f0[start_frame:end_frame]
                valid_f0 = segment_f0[voiced_flag[start_frame:end_frame]]
                if len(valid_f0) > 0:
                    med_f0 = np.median(valid_f0)
                    if med_f0 > 0:
                        note_num = int(round(69 + 12 * np.log2(med_f0 / 440)))
                        note_num = max(36, min(96, note_num))
                    else:
                        note_num = 60
                else:
                    note_num = 60
            else:
                note_num = 60

        notes.append({
            'type': 'note',
            'start_time': start_time,
            'end_time': end_time,
            'length_sec': duration,
            'lyric': lyric,
            'note_num': note_num,
        })

    total_ticks = sum(max(1, int(n['length_sec'] * ref_tps)) for n in notes)
    if audio_duration > 0:
        derived_tempo = (total_ticks / audio_duration) * 60 / 480
    else:
        derived_tempo = REFERENCE_TEMPO
    print(f"Derived tempo: {derived_tempo:.1f} BPM (from {len(notes)} notes)")

    derived_tps = 480 * derived_tempo / 60

    ust_content = f"""[#VERSION]
UST Version 1.2
[#SETTING]
Tempo={derived_tempo:.2f}
TimeSignature=4/4
Tracks=1
ProjectName=Generated from Vocals
VoiceDir=%VOICE%\\Default
OutFile=
CacheDir=%CACHE%
Tool1=wavtool.exe
Tool2=resampler.exe
Mode2=True
"""

    note_index = 0

    for note in notes:
        length = max(1, int(note['length_sec'] * derived_tps))

        if note['type'] == 'rest':
            ust_content += f"[#{note_index:04d}]\n"
            ust_content += f"Length={length}\n"
            ust_content += "Lyric=R\n"
            ust_content += "NoteNum=60\n"
            ust_content += "PreUtterance=\n"
            ust_content += "VoiceOverlap=\n"
            ust_content += "Intensity=100\n"
            ust_content += "Modulation=0\n"
            ust_content += "Velocity=100\n"
            ust_content += "Flags=\n"
        else:
            ust_content += f"[#{note_index:04d}]\n"
            ust_content += f"Length={length}\n"
            ust_content += f"Lyric={note['lyric']}\n"
            ust_content += f"NoteNum={note['note_num']}\n"
            ust_content += "PreUtterance=\n"
            ust_content += "VoiceOverlap=\n"
            ust_content += "Velocity=100\n"
            ust_content += "Intensity=100\n"
            ust_content += "Modulation=0\n"
            ust_content += "Flags=g0B0H0P86\n"
            ust_content += "PBS=-40;0\n"
            ust_content += "PBW=80\n"
            ust_content += "PBY=0\n"
            ust_content += "PBM=\n"
            ust_content += "VBR=75,175,25,10,10,0,0\n"

        note_index += 1

    last_end_time = notes[-1]['end_time'] if notes else 0.0
    trailing_sec = audio_duration - last_end_time
    if trailing_sec > 0.05:
        trailing_ticks = int(trailing_sec * derived_tps)
        while trailing_ticks > 0:
            chunk = min(trailing_ticks, ticks_per_measure)
            ust_content += f"[#{note_index:04d}]\n"
            ust_content += f"Length={chunk}\n"
            ust_content += "Lyric=R\n"
            ust_content += "NoteNum=60\n"
            ust_content += "PreUtterance=\n"
            ust_content += "VoiceOverlap=\n"
            ust_content += "Intensity=100\n"
            ust_content += "Modulation=0\n"
            ust_content += "Velocity=100\n"
            ust_content += "Flags=\n"
            note_index += 1
            trailing_ticks -= chunk

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ust_content)


def get_correct_words(script_dir, use_local_lyrics, use_genius):
    correct_words = []
    if use_local_lyrics:
        print("Using local lyrics from lyrics.txt...")
        lyrics_file = script_dir / "lyrics.txt"
        if lyrics_file.exists():
            with open(lyrics_file, 'r', encoding='utf-8') as f:
                lyrics_text = f.read()
            correct_words = [word.strip() for word in lyrics_text.split() if word.strip()]
            print(f"Loaded {len(correct_words)} words from lyrics.txt.")
        else:
            print("lyrics.txt not found. Using transcribed lyrics.")
    elif use_genius:
        print("Fetching lyrics from Genius...")
        genius_url = "https://genius.com/Mili-jpn-worldexecuteme-lyrics"
        correct_lyrics = fetch_lyrics_from_genius(genius_url)
        if correct_lyrics:
            correct_words = [word.strip() for word in correct_lyrics.split() if word.strip()]
            print(f"Fetched {len(correct_words)} words from Genius.")
        else:
            print("Failed to fetch lyrics from Genius.")
    else:
        print("Using transcribed lyrics.")
    return correct_words


def get_lyrics_segments(transcription):
    segments = []
    for segment in transcription['segments']:
        if 'words' in segment:
            segments.extend(segment['words'])
        else:
            segments.append(segment)
    return segments


def replace_lyrics_text(segments, correct_words):
    if len(correct_words) != len(segments):
        print(f"Warning: {len(correct_words)} provided words vs {len(segments)} transcribed segments. Timing may be off.")
    for i, segment in enumerate(segments):
        if i < len(correct_words):
            segment['text'] = correct_words[i]
            segment['word'] = correct_words[i]


def find_audio_file(script_dir):
    audio_path_wav = script_dir / "vocals.wav"
    audio_path_mp3 = script_dir / "vocals.mp3"

    if audio_path_wav.exists():
        print("Found vocals.wav")
        return audio_path_wav
    elif audio_path_mp3.exists():
        print("Found vocals.mp3")
        return audio_path_mp3
    return audio_path_wav


def main():
    use_genius = False
    use_local_lyrics = False
    use_silence_detection = True
    whisper_model_size = "base"

    script_dir = Path(__file__).parent
    audio_path = find_audio_file(script_dir)

    print(f"Current directory: {Path.cwd()}")
    print(f"Looking for: {script_dir / 'vocals.wav'} or {script_dir / 'vocals.mp3'}")

    if not audio_path.exists():
        print("Audio file not found. Please provide either 'vocals.wav' or 'vocals.mp3' in the script directory.")
        return

    print("Loading Whisper model...")
    try:
        whisper_model = load_whisper_model(whisper_model_size)
    except Exception as e:
        print(f"Failed to load specified Whisper model '{whisper_model_size}': {e}")
        print("Trying base model...")
        try:
            whisper_model = load_whisper_model("base")
        except Exception as e2:
            print(f"Failed to load base model: {e2}")
            print("Cannot continue without Whisper model. Exiting.")
            return

    correct_words = get_correct_words(script_dir, use_local_lyrics, use_genius)

    print("Transcribing audio with Whisper...")
    transcription = transcribe_audio(str(audio_path), model=whisper_model)
    lyrics_segments = get_lyrics_segments(transcription)

    if correct_words:
        print(f"Replacing transcribed text with {len(correct_words)} provided words...")
        replace_lyrics_text(lyrics_segments, correct_words)

    print(f"Using {len(lyrics_segments)} segments.")

    if use_silence_detection:
        print("Detecting voiced segments...")
        voiced_segments = detect_voiced_segments(str(audio_path))
        print(f"Detected {len(voiced_segments)} voiced segments.")

        filtered = []
        for segment in lyrics_segments:
            s, e = segment['start'], segment['end']
            for vs, ve in voiced_segments:
                if s < ve and e > vs:
                    filtered.append(segment)
                    break

        print(f"Filtered to {len(filtered)} segments within voiced regions.")
        lyrics_segments = filtered

    print("Detecting onsets...")
    onset_times = detect_onsets(str(audio_path))
    print(f"Detected {len(onset_times)} onsets.")

    lyrics_segments = snap_segments_to_onsets(lyrics_segments, onset_times)
    lyrics_segments = fix_overlapping_segments(lyrics_segments)

    print("Extracting pitch...")
    pitch_data = extract_pitch(str(audio_path))

    y, sr = librosa.load(str(audio_path), sr=None)
    audio_duration = len(y) / sr
    print(f"Audio duration: {audio_duration:.2f}s")

    ust_file = audio_path.parent / (audio_path.stem + ".ust")
    print(f"Generating UST file: {ust_file}")
    generate_ust(lyrics_segments, pitch_data, str(ust_file), audio_duration)

    print("Done! Open the UST file in OpenUTAU and render with DiffSinger.")


if __name__ == "__main__":
    main()
