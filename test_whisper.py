#!/usr/bin/env python3
"""
Test script to verify Whisper installation and basic functionality.
"""

import whisper
import os

def test_whisper():
    """Test basic Whisper functionality."""
    print("Testing OpenAI Whisper installation...")
    
    # List available models
    print("Available Whisper models:")
    for model_name in whisper.available_models():
        print(f"  - {model_name}")
    
    # Load a small model for testing
    print("\nLoading 'tiny' model for testing...")
    model = whisper.load_model("tiny")
    print("✓ Model loaded successfully!")
    
    print("\nWhisper is ready to use for transcription!")
    print("You can now use it to transcribe lecture videos.")

if __name__ == "__main__":
    test_whisper() 