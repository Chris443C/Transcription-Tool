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

brew install ffmpeg
Windows (Chocolatey):

choco install ffmpeg
Or download manually from FFmpeg.org.

2. Install Required Python Packages
Run the following command to install dependencies:


pip install -r requirements.txt
Usage Instructions
Place all your audio files in the same folder as the script.
Run the script:

python boost_and_transcribe.py
The script will:
Increase the volume of each audio file.
Transcribe & translate audio into English subtitles.
Save subtitles in the output_subtitles/ folder.
Supported Files
This script processes MP3 files. If you need other formats, convert them to MP3 first using:


ffmpeg -i input.wav -acodec mp3 output.mp3
Output
The boosted audio files will have the prefix:

boosted_Standard recording 14.mp3
The English subtitles (SRT files) will be stored in:

output_subtitles/Standard recording 14.srt
Example Workflow
Step 1: Run the script

python boost_and_transcribe.py
Step 2: Check the output

ls output_subtitles/
Example output:

python-repl

Standard recording 14.srt
Standard recording 15.srt
Standard recording 16.srt
...
Step 3: (Optional) Add Subtitles to Video
If you want to embed the subtitles into a video file:


ffmpeg -i video.mp4 -vf subtitles=output_subtitles/Standard recording 14.srt -c:a copy output.mp4
Troubleshooting
If Whisper is not found, install it with:

pip install openai-whisper
If FFmpeg is missing, reinstall using the instructions above.
If the script doesn’t find your audio files, make sure they are in the same directory.
Credits
Uses OpenAI Whisper for transcription
Uses FFmpeg for audio processing
markdown
Copy
Edit

### **Next Steps**
1. Save this file as `README.md` in your project directory.
2. Run the script following the instructions.
3. Let me know if you need additional details or modifications! 🚀
