import os
import subprocess

# List of audio files
audio_files = [
    "Standard recording 14.mp3",
    "Standard recording 15.mp3",
    "Standard recording 16.mp3",
    "Standard recording 17.mp3",
    "Standard recording 18.mp3",
    "Standard recording 19.mp3",
    "Standard recording 20.mp3",
    "Standard recording 21.mp3",
    "Standard recording 22.mp3"
]

output_dir = "output_subtitles"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

for file in audio_files:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue

    # Create a boosted version of the file
    boosted_file = f"boosted_{file}"
    ffmpeg_cmd = [
        "ffmpeg", "-i", file, "-af", "volume=3.0",
        "-c:a", "libmp3lame", boosted_file, "-y"
    ]
    
    print(f"Boosting volume for {file}...")
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Generate subtitles using Whisper
    subtitle_file = os.path.join(output_dir, f"{file.replace('.mp3', '.srt')}")
    whisper_cmd = [
        "whisper", boosted_file, "--model", "medium", "--task", "translate",
        "--output_format", "srt", "--output_dir", output_dir
    ]

    print(f"Transcribing and translating {boosted_file}...")
    subprocess.run(whisper_cmd)

    print(f"Subtitles saved: {subtitle_file}")

print("\n✅ All files processed. Check the 'output_subtitles' folder.")
