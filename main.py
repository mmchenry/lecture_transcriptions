# %% 
""" Packages & parameters"""
import os

video_filename = "4.1 Control systems.mov"

# Path to video files
video_root_path = '/Users/mmchenry/Documents/Teaching/E109/E109 Lecture videos'

# Path to data files
data_root_path = '/Users/mmchenry/Documents/Teaching/E109/Study guide'

# Path to video catalog
catalog_path = os.path.join(data_root_path, 'video_catalog.csv')

# Create path to audio file
audio_root_path = os.path.join(data_root_path, 'audio_files')


# %% 
""" Create video catalog """
# Create video catalog
from video_ops import create_video_catalog
video_catalog = create_video_catalog(video_root_path, catalog_path)


# %% 
""" Extract audio from videos """
from audio_ops import extract_audio
audio_path = extract_audio(catalog_path, video_root_path, audio_root_path)
print(f"Audio extraction results: {len(audio_path)} files processed")


# %% 
""" Transcribe audio """
# Creates shell script for running whisper
# Whisper is a package that can be used to transcribe audio files.
from audio_ops import transcribe_audio
transcription_results = transcribe_audio(catalog_path, audio_root_path, data_root_path)
print(f"Transcription results: {len(transcription_results)} files processed")


# %% 
""" Detect slide transitions from video """
from video_ops import detect_slide_transitions
slide_transitions = detect_slide_transitions(catalog_path, video_root_path, data_root_path)
print(f"Slide transition results: {len(slide_transitions)} files processed")


# %% 
""" Align Transcript with Slide Timings """
from audio_ops import align_transcript
alignment_results = align_transcript(catalog_path, data_root_path)
print(f"Alignment results: {len(alignment_results)} files processed")


# %% 
""" Extract Slide Images """
from slide_extractor import extract_slide_images
slide_images = extract_slide_images(catalog_path, video_root_path, data_root_path)
print(f"Slide images extracted: {slide_images} images")


# %% 
""" Generate Website """
from website_generator import generate_lecture_website
website_results = generate_lecture_website(catalog_path, data_root_path)
print(f"Website generated with {len(website_results)} lectures")



# %% 
