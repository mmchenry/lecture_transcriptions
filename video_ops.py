import cv2
import numpy as np
import pandas as pd
import os
import glob


def create_video_catalog(video_root_path, catalog_path, video_extensions=['.mov', '.mp4']):
    """
    Create a catalog of video files in the specified directory.
    
    Args:
        video_root_path (str): Path to the directory containing video files
        catalog_path (str): Path where the CSV catalog will be saved
        video_extensions (list): List of file extensions to search for (default: ['.mov', '.mp4'])
    
    Returns:
        pd.DataFrame: The created catalog dataframe or existing catalog if already present
    """
    # Check if catalog already exists
    if os.path.exists(catalog_path):
        print(f"Catalog already exists at '{catalog_path}'. Skipping creation.")
        return pd.read_csv(catalog_path)
    
    # Find all video files with the specified extensions
    video_files = []
    for ext in video_extensions:
        video_pattern = os.path.join(video_root_path, f"*{ext}")
        video_files.extend(glob.glob(video_pattern))
    
    # Extract just the filenames
    filenames = [os.path.basename(file) for file in video_files]
    
    # Extract lecture numbers from filenames
    lecture_nums = []
    import re
    
    for filename in filenames:
        # Remove file extension for processing
        name_without_ext = os.path.splitext(filename)[0]
        
        # Try multiple patterns to extract lecture numbers (including decimals)
        # Pattern 1: Decimal number at start followed by space, dash, or underscore
        match = re.match(r'^(\d+\.\d+)[\s\-_]', name_without_ext)
        if not match:
            # Pattern 2: Integer at start followed by space, dash, or underscore
            match = re.match(r'^(\d+)[\s\-_]', name_without_ext)
        if not match:
            # Pattern 3: Decimal number at start followed by any non-alphanumeric character
            match = re.match(r'^(\d+\.\d+)[^\w]', name_without_ext)
        if not match:
            # Pattern 4: Integer at start followed by any non-alphanumeric character
            match = re.match(r'^(\d+)[^\w]', name_without_ext)
        if not match:
            # Pattern 5: Decimal number at start (end of string or followed by letter)
            match = re.match(r'^(\d+\.\d+)(?:\s|$|[a-zA-Z])', name_without_ext)
        if not match:
            # Pattern 6: Integer at start (end of string or followed by letter)
            match = re.match(r'^(\d+)(?:\s|$|[a-zA-Z])', name_without_ext)
        
        if match:
            lecture_nums.append(float(match.group(1)))
        else:
            # Debug: print filename that doesn't match any pattern
            print(f"Could not extract lecture number from: '{filename}'")
            lecture_nums.append(0.0)  # Default value if no number found
    
    # Create dataframe with the specified columns
    df = pd.DataFrame({
        'lecture_num': lecture_nums,
        'filename': filenames,
        'extract_audio': 0,
        'transcribe': 0,
        'detect_transitions': 0,
        'align': 0
    })
    
    # Sort by lecture number, then by filename for files with same lecture number
    df = df.sort_values(['lecture_num', 'filename'])
    
    # Save the dataframe to CSV
    df.to_csv(catalog_path, index=False)
    
    print(f"Video catalog created with {len(filenames)} files and saved to '{catalog_path}'")
    
    return df


def detect_slide_transitions(catalog_path, video_root_path, data_root_path):
    """
    Detect slide transitions from videos listed in the catalog using improved algorithms
    """
    # Load catalog
    catalog = pd.read_csv(catalog_path)

    # Filter catalog to only include rows where detect_transitions is 1
    catalog = catalog[catalog['detect_transitions'] == 1]

    processed_files = []

    # Loop through each row in catalog
    for index, row in catalog.iterrows():
        video_filename = row['filename']
        
        # Get video path
        video_path = os.path.join(video_root_path, video_filename)
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
            
        print(f"Detecting slide transitions in: {video_filename}")
 
        # Create output filename by replacing video extension with csv
        output_filename = os.path.splitext(video_filename)[0] + '.csv'
        # Create slide transitions directory if it doesn't exist
        slide_transitions_dir = os.path.join(data_root_path, 'slide_transitions')
        os.makedirs(slide_transitions_dir, exist_ok=True)
        output_path = os.path.join(slide_transitions_dir, output_filename)

        try:
            # Create output directory if it doesn't exist
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Could not open video file: {video_path}")
                continue

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            print(f"  Video duration: {duration:.1f} seconds, FPS: {fps:.1f}")
            
            frame_count = 0
            prev_frame = None
            slide_changes = []
            
            # More sensitive parameters for lecture slides
            sample_rate = max(1, int(fps / 4))  # Sample every 0.25 seconds (more frequent)
            
            # Multiple thresholds for different types of changes
            thresholds = {
                'major_change': 0.08,    # Major slide changes
                'minor_change': 0.03,    # Minor content changes
                'text_change': 0.02,     # Text additions/changes
                'highlight': 0.015       # Highlights/annotations
            }
            
            # Minimum gap between transitions (reduced for more sensitivity)
            min_gap_seconds = 1.0
            
            print(f"  Sampling every {sample_rate} frames (every {sample_rate/fps:.2f} seconds)")
            print(f"  Using multiple thresholds: {thresholds}")

            # Detect slide transitions
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process every Nth frame to speed up detection
                if frame_count % sample_rate != 0:
                    frame_count += 1
                    continue

                # Convert frame to grayscale and resize for faster processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240))  # Resize for faster processing

                if prev_frame is not None:
                    # Method 1: Histogram comparison
                    hist1 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    
                    # Method 2: Pixel-wise difference
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff) / 255.0
                    
                    # Method 3: Edge-based comparison
                    edges1 = cv2.Canny(prev_frame, 30, 100)  # Lower thresholds for more edges
                    edges2 = cv2.Canny(gray, 30, 100)
                    edge_diff = cv2.absdiff(edges1, edges2)
                    edge_score = np.sum(edge_diff) / (edge_diff.shape[0] * edge_diff.shape[1])
                    edge_score_norm = edge_score / 255.0
                    
                    # Method 4: Local binary pattern (LBP) for texture changes
                    def get_lbp_features(img):
                        # Simple LBP implementation
                        lbp = np.zeros_like(img)
                        for i in range(1, img.shape[0]-1):
                            for j in range(1, img.shape[1]-1):
                                center = img[i, j]
                                code = 0
                                code |= (img[i-1, j-1] > center) << 7
                                code |= (img[i-1, j] > center) << 6
                                code |= (img[i-1, j+1] > center) << 5
                                code |= (img[i, j+1] > center) << 4
                                code |= (img[i+1, j+1] > center) << 3
                                code |= (img[i+1, j] > center) << 2
                                code |= (img[i+1, j-1] > center) << 1
                                code |= (img[i, j-1] > center) << 0
                                lbp[i, j] = code
                        return lbp
                    
                    lbp1 = get_lbp_features(prev_frame)
                    lbp2 = get_lbp_features(gray)
                    lbp_diff = cv2.absdiff(lbp1, lbp2)
                    lbp_score = np.mean(lbp_diff) / 255.0
                    
                    # Combined score with different weights
                    combined_score = (
                        (1 - hist_diff) * 0.25 +      # Histogram changes
                        mean_diff * 0.25 +            # Pixel changes
                        edge_score_norm * 0.25 +      # Edge changes
                        lbp_score * 0.25              # Texture changes
                    )
                    
                    timestamp = frame_count / fps
                    
                    # Check against multiple thresholds
                    transition_detected = False
                    transition_type = None
                    
                    if combined_score > thresholds['major_change']:
                        transition_type = 'major'
                        transition_detected = True
                    elif combined_score > thresholds['minor_change']:
                        transition_type = 'minor'
                        transition_detected = True
                    elif combined_score > thresholds['text_change']:
                        transition_type = 'text'
                        transition_detected = True
                    elif combined_score > thresholds['highlight']:
                        transition_type = 'highlight'
                        transition_detected = True
                    
                    # Additional check: if we haven't detected anything in a while, be more sensitive
                    if not slide_changes or (timestamp - slide_changes[-1]) > 10.0:
                        # If no transition in 10+ seconds, lower the threshold
                        if combined_score > 0.01:  # Very low threshold
                            transition_type = 'timeout'
                            transition_detected = True
                    
                    if transition_detected:
                        # Check if enough time has passed since last transition
                        if not slide_changes or (timestamp - slide_changes[-1]) > min_gap_seconds:
                            slide_changes.append(timestamp)
                            print(f"    {transition_type.capitalize()} transition at {timestamp:.1f}s (score: {combined_score:.4f})")

                prev_frame = gray.copy()
                frame_count += 1
                
                # Progress indicator
                if frame_count % (sample_rate * 20) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"    Progress: {progress:.1f}%")

            cap.release()

            # Post-process: remove transitions that are too close together
            if len(slide_changes) > 1:
                filtered_changes = [slide_changes[0]]
                for i in range(1, len(slide_changes)):
                    if slide_changes[i] - filtered_changes[-1] >= min_gap_seconds:
                        filtered_changes.append(slide_changes[i])
                slide_changes = filtered_changes

            # Save slide transitions to CSV
            if slide_changes:
                df = pd.DataFrame({
                    "slide_number": range(1, len(slide_changes) + 1), 
                    "start_time": slide_changes,
                    "end_time": slide_changes[1:] + [duration] if len(slide_changes) > 1 else [duration]
                })
            else:
                # If no transitions detected, create a single segment
                df = pd.DataFrame({
                    "slide_number": [1],
                    "start_time": [0.0],
                    "end_time": [duration]
                })
                
            df.to_csv(output_path, index=False)

            print(f"✓ Slide transitions saved to: {output_path}")
            print(f"  Found {len(slide_changes)} slide transitions")
            
            processed_files.append({
                'video_file': video_filename,
                'slide_transitions_file': output_path,
                'num_transitions': len(slide_changes),
                'duration': duration
            })
            
        except Exception as e:
            print(f"Error processing {video_filename}: {e}")
            continue

    print(f"\nSlide transition detection complete! Processed {len(processed_files)} files.")
    return processed_files


def analyze_video_for_transitions(video_path, sample_seconds=30):
    """
    Analyze a video to suggest optimal parameters for slide transition detection
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Analyzing video: {os.path.basename(video_path)}")
    print(f"Duration: {duration:.1f}s, FPS: {fps:.1f}")
    
    # Sample frames from different parts of the video
    sample_frames = []
    sample_times = [0, duration/4, duration/2, 3*duration/4, duration-1]
    
    for time in sample_times:
        frame_num = int(time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))
            sample_frames.append((time, gray))
    
    cap.release()
    
    # Analyze differences between sample frames
    differences = []
    for i in range(len(sample_frames)-1):
        time1, frame1 = sample_frames[i]
        time2, frame2 = sample_frames[i+1]
        
        # Calculate various difference metrics
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        diff = cv2.absdiff(frame1, frame2)
        mean_diff = np.mean(diff) / 255.0
        
        edges1 = cv2.Canny(frame1, 50, 150)
        edges2 = cv2.Canny(frame2, 50, 150)
        edge_diff = cv2.absdiff(edges1, edges2)
        edge_score = np.sum(edge_diff) / (edge_diff.shape[0] * edge_diff.shape[1])
        
        combined_score = (1 - hist_diff) * 0.4 + mean_diff * 0.4 + (edge_score / 255.0) * 0.2
        
        differences.append({
            'time_gap': time2 - time1,
            'hist_diff': hist_diff,
            'mean_diff': mean_diff,
            'edge_score': edge_score / 255.0,
            'combined_score': combined_score
        })
    
    # Calculate statistics
    combined_scores = [d['combined_score'] for d in differences]
    mean_score = np.mean(combined_scores)
    std_score = np.std(combined_scores)
    
    print(f"\nAnalysis results:")
    print(f"  Mean difference score: {mean_score:.3f}")
    print(f"  Standard deviation: {std_score:.3f}")
    print(f"  Min score: {min(combined_scores):.3f}")
    print(f"  Max score: {max(combined_scores):.3f}")
    
    # Suggest threshold
    suggested_threshold = mean_score + 2 * std_score
    print(f"\nSuggested threshold: {suggested_threshold:.3f}")
    print(f"  (This will detect changes that are 2+ standard deviations above the mean)")
    
    # Analyze content type
    if mean_score < 0.05:
        print("  Content appears to be static slides with minimal changes")
        print("  Consider using a lower threshold (0.05-0.10)")
    elif mean_score < 0.15:
        print("  Content appears to be typical lecture slides")
        print("  Current threshold should work well")
    else:
        print("  Content appears to have frequent changes")
        print("  Consider using a higher threshold (0.20-0.30)")
    
    return {
        'suggested_threshold': suggested_threshold,
        'mean_score': mean_score,
        'std_score': std_score,
        'differences': differences
    }

