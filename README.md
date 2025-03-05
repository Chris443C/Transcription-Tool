# Boost Audio and Generate English Subtitles

This script automatically **amplifies audio files**, **transcribes**, and **translates** subtitles to **English** using OpenAI's Whisper.

## **Features**
✅ Increases the audio volume for better transcription  
✅ Uses **Whisper** to transcribe and translate subtitles into **English**  
✅ Saves the subtitles in **SRT format**  
✅ Processes multiple audio files automatically  

---

## **Prerequisites**
### **1. Install FFmpeg**
Before running the script, **install FFmpeg**:

- **Ubuntu/Debian:**
  ```bash
  sudo apt install ffmpeg -y
MacOS (Homebrew):
bash
Copy
Edit
brew install ffmpeg
Windows (Chocolatey):
bash
Copy
Edit
choco install ffmpeg
Or download manually from FFmpeg.org.

2. Install Required Python Packages
Run the following command to install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Usage Instructions
Place all your audio files in the same folder as the script.
Run the script:
bash
Copy
Edit
python boost_and_transcribe.py
The script will:
Increase the volume of each audio file.
Transcribe & translate audio into English subtitles.
Save subtitles in the output_subtitles/ folder.
Supported Files
This script processes MP3 files. If you need other formats, convert them to MP3 first using:

bash
Copy
Edit
ffmpeg -i input.wav -acodec mp3 output.mp3
Output
The boosted audio files will have the prefix:
nginx
Copy
Edit
boosted_Standard recording 14.mp3
The English subtitles (SRT files) will be stored in:
Copy
Edit
output_subtitles/Standard recording 14.srt
Example Workflow
Step 1: Run the script
bash
Copy
Edit
python boost_and_transcribe.py
Step 2: Check the output
bash
Copy
Edit
ls output_subtitles/
Example output:

python-repl
Copy
Edit
Standard recording 14.srt
Standard recording 15.srt
Standard recording 16.srt
...
Step 3: (Optional) Add Subtitles to Video
If you want to embed the subtitles into a video file:

bash
Copy
Edit
ffmpeg -i video.mp4 -vf subtitles=output_subtitles/Standard recording 14.srt -c:a copy output.mp4
Troubleshooting
If Whisper is not found, install it with:
bash
Copy
Edit
pip install openai-whisper
If FFmpeg is missing, reinstall using the instructions above.
If the script doesn’t find your audio files, make sure they are in the same directory.
Credits
Uses OpenAI Whisper for transcription
Uses FFmpeg for audio processing
