#!/usr/bin/env python3
"""
Slide Extractor
Extracts individual slide images from videos at detected transition points
"""

import cv2
import os
import pandas as pd
from pathlib import Path
import numpy as np


def extract_slide_images(catalog_path, video_root_path, data_root_path):
    """
    Extract slide images from videos at detected transition points
    """
    # Create output directory
    output_dir = os.path.join(data_root_path, 'slide_images')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load catalog
    catalog = pd.read_csv(catalog_path)

    # Filter catalog to only include rows where generate_webpage is 1 (if column exists)
    if 'generate_webpage' in catalog.columns:
        catalog = catalog[catalog['generate_webpage'] == 1]
    else:
        # If column doesn't exist, process all videos that have slide transitions
        print("Note: 'generate_webpage' column not found, processing all videos with slide transitions")
    
    extracted_count = 0
    
    for index, row in catalog.iterrows():
        video_filename = row['filename']
        base_name = os.path.splitext(video_filename)[0]
        
        # Check if slide transitions file exists
        transitions_path = Path(data_root_path) / "slide_transitions" / f"{base_name}.csv"
        if not transitions_path.exists():
            print(f"Skipping {base_name} - no slide transitions file")
            continue
        
        # Load slide transitions
        slide_transitions = pd.read_csv(transitions_path)
        
        # Get video path
        video_path = Path(video_root_path) / video_filename
        
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            continue
        
        print(f"Extracting slides from: {video_filename}")
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Extract frame at each transition point
            for _, slide in slide_transitions.iterrows():
                slide_num = slide.get('slide_number', 1)
                start_time = slide.get('start_time', 0)
                
                # Calculate frame number
                frame_number = int(start_time * fps)
                
                # Ensure frame number is within valid range
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_number >= total_frames:
                    frame_number = total_frames - 1
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame as image
                    output_filename = f"{base_name}_slide_{int(slide_num):03d}.png"
                    output_file = output_path / output_filename
                    
                    # Resize for web (maintain aspect ratio)
                    height, width = frame.shape[:2]
                    max_width = 1200
                    max_height = 800
                    
                    if width > max_width or height > max_height:
                        scale = min(max_width / width, max_height / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Save image
                    cv2.imwrite(str(output_file), frame)
                    print(f"  Extracted slide {slide_num} at {start_time:.1f}s")
                    extracted_count += 1
                else:
                    print(f"  Failed to extract slide {slide_num}")
            
            cap.release()
            
        except Exception as e:
            print(f"Error extracting slides from {video_filename}: {e}")
            continue
    
    print(f"Extracted {extracted_count} slide images to {output_path}")
    return extracted_count


def extract_slide_images_advanced(catalog_path, video_root_path, data_root_path, output_dir="lecture_website/assets/slides"):
    """
    Advanced slide extraction with multiple frames per slide and quality selection
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load catalog
    catalog = pd.read_csv(catalog_path)
    
    extracted_count = 0
    
    for index, row in catalog.iterrows():
        video_filename = row['filename']
        base_name = os.path.splitext(video_filename)[0]
        
        # Check if slide transitions file exists
        transitions_path = Path(data_root_path) / "slide_transitions" / f"{base_name}.csv"
        if not transitions_path.exists():
            continue
        
        # Load slide transitions
        slide_transitions = pd.read_csv(transitions_path)
        
        # Get video path
        video_path = Path(video_root_path) / video_filename
        
        if not video_path.exists():
            continue
        
        print(f"Extracting slides from: {video_filename}")
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Extract frames for each slide
            for i, slide in slide_transitions.iterrows():
                slide_num = slide.get('slide_number', i+1)
                start_time = slide.get('start_time', 0)
                end_time = slide.get('end_time', 0)
                
                # Extract multiple frames and select the best one
                frames = []
                frame_times = []
                
                # Sample frames throughout the slide duration
                slide_duration = end_time - start_time
                num_samples = min(5, max(1, int(slide_duration / 2)))  # Sample every 2 seconds, max 5 frames
                
                for j in range(num_samples):
                    sample_time = start_time + (j * slide_duration / num_samples)
                    frame_number = int(sample_time * fps)
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    
                    if ret:
                        frames.append(frame)
                        frame_times.append(sample_time)
                
                if frames:
                    # Select the best frame (highest contrast/edge density)
                    best_frame = select_best_frame(frames)
                    best_time = frame_times[frames.index(best_frame)]
                    
                    # Save the best frame
                    output_filename = f"{base_name}_slide_{int(slide_num):03d}.png"
                    output_file = output_path / output_filename
                    
                    # Resize for web
                    height, width = best_frame.shape[:2]
                    max_width = 1200
                    max_height = 800
                    
                    if width > max_width or height > max_height:
                        scale = min(max_width / width, max_height / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        best_frame = cv2.resize(best_frame, (new_width, new_height))
                    
                    # Save image
                    cv2.imwrite(str(output_file), best_frame)
                    print(f"  Extracted slide {slide_num} at {best_time:.1f}s")
                    extracted_count += 1
            
            cap.release()
            
        except Exception as e:
            print(f"Error extracting slides from {video_filename}: {e}")
            continue
    
    print(f"Extracted {extracted_count} slide images")
    return extracted_count


def select_best_frame(frames):
    """
    Select the best frame from a list of frames based on quality metrics
    """
    if len(frames) == 1:
        return frames[0]
    
    best_score = 0
    best_frame = frames[0]
    
    for frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate quality metrics
        # 1. Edge density (more edges = more content)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Contrast (standard deviation of pixel values)
        contrast = np.std(gray)
        
        # 3. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Combined score
        score = edge_density * 0.4 + (contrast / 255.0) * 0.3 + (sharpness / 1000.0) * 0.3
        
        if score > best_score:
            best_score = score
            best_frame = frame
    
    return best_frame


if __name__ == "__main__":
    # Example usage
    catalog_path = "/Users/mmchenry/Documents/Teaching/E109/Study guide/video_catalog.csv"
    video_root_path = "/Users/mmchenry/Documents/Teaching/E109/E109 Lecture videos"
    data_root_path = "/Users/mmchenry/Documents/Teaching/E109/Study guide"
    
    extract_slide_images(catalog_path, video_root_path, data_root_path) 