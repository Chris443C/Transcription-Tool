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
  ```
- **MacOS (Homebrew):**
  ```bash
  brew install ffmpeg
  ```
- **Windows (Chocolatey):**
  ```bash
  choco install ffmpeg
  ```

Or download manually from [FFmpeg.org](https://ffmpeg.org/download.html).

### **2. Install Required Python Packages**
Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```

---

## **Usage Instructions**
1. **Place all your audio files** in the same folder as the script.
2. **Run the script**:
   ```bash
   python boost_and_transcribe.py
   ```
3. The script will:
   - **Increase the volume** of each audio file.
   - **Transcribe & translate** audio into English subtitles.
   - **Save subtitles** in the `output_subtitles/` folder.

---

## **Supported Files**
This script processes **MP3 files**. If you need other formats, convert them to MP3 first using:
```bash
ffmpeg -i input.wav -acodec mp3 output.mp3
```

---

## **Output**
- The **boosted audio files** will have the prefix:  
  ```
  boosted_Standard recording 14.mp3
  ```
- The **English subtitles (SRT files)** will be stored in:  
  ```
  output_subtitles/Standard recording 14.srt
  ```

---

## **Example Workflow**
### **Step 1: Run the script**
```bash
python boost_and_transcribe.py
```
### **Step 2: Check the output**
```bash
ls output_subtitles/
```
Example output:
```
Standard recording 14.srt
Standard recording 15.srt
Standard recording 16.srt
...
```

### **Step 3: (Optional) Add Subtitles to Video**
If you want to **embed** the subtitles into a video file:
```bash
ffmpeg -i video.mp4 -vf subtitles=output_subtitles/Standard recording 14.srt -c:a copy output.mp4
```

---

## **Troubleshooting**
- If **Whisper is not found**, install it with:
  ```bash
  pip install openai-whisper
  ```
- If **FFmpeg is missing**, reinstall using the instructions above.
- If the script **doesn’t find your audio files**, make sure they are in the same directory.

---

## **Credits**
- Uses **OpenAI Whisper** for transcription
- Uses **FFmpeg** for audio processing
