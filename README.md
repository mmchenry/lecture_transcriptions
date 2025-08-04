# Lecture Video Transcription with OpenAI Whisper

This project provides a Python environment for transcribing lecture videos using OpenAI's Whisper model, along with additional tools for video processing, data analysis, and visualization.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Homebrew (for ffmpeg installation)
- macOS (tested on macOS 14.5.0)

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd transcribe_lectures
   ```

2. **Install ffmpeg** (if not already installed):
   ```bash
   brew install ffmpeg
   ```

3. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Test the installation**:
   ```bash
   python test_whisper.py
   ```

## 📦 Installed Packages

### Core Transcription Packages
- **openai-whisper**: OpenAI's speech recognition model for transcription
- **ffmpeg-python**: Python bindings for ffmpeg video/audio processing
- **librosa**: Audio analysis library
- **soundfile**: Audio file I/O
- **torch & torchaudio**: PyTorch for deep learning (required by Whisper)

### Data Processing & Analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **opencv-python**: Computer vision and video processing

### Visualization
- **matplotlib**: Basic plotting and visualization
- **plotly**: Interactive plotting and dashboards

### Development Tools
- **jupyter**: Jupyter notebooks for interactive development
- **ipykernel**: Python kernel for Jupyter

## 🎯 Usage Examples

### Basic Transcription
```python
import whisper

# Load a model (choose based on your needs)
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Transcribe audio/video file
result = model.transcribe("path/to/your/video.mp4")
print(result["text"])
```

### Video Processing with OpenCV
```python
import cv2
import numpy as np

# Read video file
cap = cv2.VideoCapture("path/to/video.mp4")

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Your video processing code here
    # e.g., extract audio, analyze frames, etc.
    
cap.release()
```

### Data Analysis with Pandas
```python
import pandas as pd
import numpy as np

# Create a DataFrame from transcription results
transcription_data = {
    'timestamp': [0, 10, 20, 30],
    'text': ['Hello', 'world', 'this is', 'a test']
}
df = pd.DataFrame(transcription_data)
print(df)
```

### Visualization with Plotly
```python
import plotly.express as px
import plotly.graph_objects as go

# Create interactive plots
fig = px.line(df, x='timestamp', y='text', title='Transcription Timeline')
fig.show()
```

## 🔧 Environment Management

### Activate the environment:
```bash
source venv/bin/activate
```

### Deactivate the environment:
```bash
deactivate
```

### Update packages:
```bash
pip install --upgrade -r requirements.txt
```

## 📁 Project Structure
```
transcribe_lectures/
├── venv/                 # Virtual environment
├── requirements.txt      # Python dependencies
├── test_whisper.py      # Whisper test script
├── main.py              # Main application file
├── audio_ops.py         # Audio processing utilities
├── video_ops.py         # Video processing utilities
└── README.md           # This file
```

## 🎓 Whisper Model Options

Choose the appropriate Whisper model based on your needs:

- **tiny**: Fastest, least accurate (39M parameters)
- **base**: Good balance of speed/accuracy (74M parameters)
- **small**: Better accuracy, slower (244M parameters)
- **medium**: High accuracy, slower (769M parameters)
- **large**: Best accuracy, slowest (1550M parameters)
- **large-v3**: Latest large model with improved accuracy
- **turbo**: Optimized for speed with good accuracy

## 🚨 Troubleshooting

### Common Issues:

1. **ffmpeg not found**: Install with `brew install ffmpeg`
2. **CUDA/GPU issues**: Whisper will use CPU by default on macOS
3. **Memory issues**: Use smaller models (tiny, base) for large files
4. **Import errors**: Ensure virtual environment is activated

### Performance Tips:

- Use smaller models for quick testing
- Process videos in chunks for large files
- Consider using GPU acceleration if available
- Pre-process audio to reduce noise

## 📚 Additional Resources

- [OpenAI Whisper Documentation](https://github.com/openai/whisper)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 🤝 Contributing

Feel free to contribute by:
- Adding new transcription features
- Improving video processing capabilities
- Enhancing visualization tools
- Adding more examples and documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.