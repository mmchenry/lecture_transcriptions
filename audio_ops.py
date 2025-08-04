import os
import subprocess
import pandas as pd
import whisper
import json


def extract_audio(catalog_path, video_root_path, audio_root_path):
    """
    Extract audio from videos listed in the catalog
    """
    # Load catalog
    catalog = pd.read_csv(catalog_path)

    # Filter catalog to only include rows where extract_audio is 1
    catalog = catalog[catalog['extract_audio'] == 1]

    processed_audio_paths = []

    # Loop through each row in catalog
    for index, row in catalog.iterrows():
        # Get video path and create audio path for current file
        video_filename = row['filename']

        # Get video path
        video_path = os.path.join(video_root_path, video_filename)

        # Create audio path
        audio_path = os.path.join(audio_root_path, 
                                 os.path.basename(video_path).replace('.mov', '.mp3'))

        # Construct ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,  # Input video file
            '-vn',            # No video
            '-acodec', 'mp3', # Audio codec
            audio_path        # Output audio file
        ]

        # Run ffmpeg command
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Audio extracted to: {audio_path}")
            processed_audio_paths.append(audio_path)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {video_filename}: {e}")
            continue

    return processed_audio_paths


def transcribe_audio(catalog_path, audio_root_path, data_root_path):
    """
    Transcribe audio files using Whisper
    """
    
    
    # Load catalog
    catalog = pd.read_csv(catalog_path)

    # Filter catalog to only include rows where transcribe is 1
    catalog = catalog[catalog['transcribe'] == 1]

    # Load Whisper model (you can change this to 'base', 'small', 'medium', 'large' for different accuracy/speed)
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    transcribed_files = []

    # Loop through each row in catalog
    for index, row in catalog.iterrows():
        video_filename = row['filename']
        
        # Create audio filename (replace .mov with .mp3)
        audio_filename = os.path.basename(video_filename).replace('.mov', '.mp3')
        audio_path = os.path.join(audio_root_path, audio_filename)
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue
            
        print(f"Transcribing: {audio_filename}")
        
        try:
            # Transcribe the audio
            result = model.transcribe(audio_path)
            
            # Create output directory for transcriptions
            transcript_dir = os.path.join(data_root_path, 'transcriptions')
            os.makedirs(transcript_dir, exist_ok=True)
            
            # Save transcription as text file
            transcript_filename = audio_filename.replace('.mp3', '_transcript.txt')
            transcript_path = os.path.join(transcript_dir, transcript_filename)
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            
            # Save detailed results as JSON (includes timestamps, confidence scores, etc.)
            json_filename = audio_filename.replace('.mp3', '_transcript.json')
            json_path = os.path.join(transcript_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Transcription saved to: {transcript_path}")
            print(f"✓ Detailed results saved to: {json_path}")
            
            transcribed_files.append({
                'audio_file': audio_filename,
                'transcript_file': transcript_path,
                'json_file': json_path,
                'text': result['text']
            })
            
        except Exception as e:
            print(f"Error transcribing {audio_filename}: {e}")
            continue

    print(f"\nTranscription complete! Processed {len(transcribed_files)} files.")
    return transcribed_files 

def align_transcript(catalog_path, data_root_path):
    """
    Align transcript with slide timings
    """
    # Load catalog
    catalog = pd.read_csv(catalog_path)

    # Filter catalog to only include rows where align is 1
    catalog = catalog[catalog['align'] == 1]

    aligned_results = []

    for index, row in catalog.iterrows():
        video_filename = row['filename']
        
        # Create filenames based on video filename
        base_name = os.path.splitext(video_filename)[0]
        slide_transitions_filename = f"{base_name}.csv"
        transcript_filename = f"{base_name}_transcript.json"
        
        print(f"Aligning transcript for: {video_filename}")

        # Check if required files exist
        slide_transitions_path = os.path.join(data_root_path, 'slide_transitions', slide_transitions_filename)
        transcript_path = os.path.join(data_root_path, 'transcriptions', transcript_filename)
        
        if not os.path.exists(slide_transitions_path):
            print(f"  Slide transitions file not found: {slide_transitions_path}")
            continue
            
        if not os.path.exists(transcript_path):
            print(f"  Transcript file not found: {transcript_path}")
            continue

        try:
            # Load slide transitions
            slide_transitions = pd.read_csv(slide_transitions_path)
            
            # Load transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Extract transcript segments
            segments = transcript_data.get('segments', [])
            
            if not segments:
                print(f"  No transcript segments found in {transcript_filename}")
                continue
            
            # Create alignment results
            aligned_segments = []
            
            for segment in segments:
                segment_start = segment.get('start', 0)
                segment_end = segment.get('end', 0)
                segment_text = segment.get('text', '').strip()
                
                # Find which slide this segment belongs to
                slide_number = 1
                slide_start_time = 0.0
                slide_end_time = float('inf')
                
                for _, slide_row in slide_transitions.iterrows():
                    slide_start = slide_row.get('start_time', 0)
                    slide_end = slide_row.get('end_time', float('inf'))
                    
                    # Check if segment overlaps with this slide
                    if segment_start >= slide_start and segment_start < slide_end:
                        slide_number = slide_row.get('slide_number', 1)
                        slide_start_time = slide_start
                        slide_end_time = slide_end
                        break
                
                # Calculate overlap percentage with the slide
                segment_duration = segment_end - segment_start
                slide_duration = slide_end_time - slide_start_time
                
                if slide_duration > 0:
                    overlap_start = max(segment_start, slide_start_time)
                    overlap_end = min(segment_end, slide_end_time)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    overlap_percentage = (overlap_duration / segment_duration) * 100
                else:
                    overlap_percentage = 0
                
                aligned_segments.append({
                    'segment_start': segment_start,
                    'segment_end': segment_end,
                    'segment_text': segment_text,
                    'slide_number': slide_number,
                    'slide_start_time': slide_start_time,
                    'slide_end_time': slide_end_time,
                    'overlap_percentage': overlap_percentage,
                    'confidence': segment.get('avg_logprob', 0)
                })
            
            # Create output directory
            aligned_dir = os.path.join(data_root_path, 'aligned_transcripts')
            os.makedirs(aligned_dir, exist_ok=True)
            
            # Save aligned transcript
            aligned_filename = f"{base_name}_aligned.csv"
            aligned_path = os.path.join(aligned_dir, aligned_filename)
            
            aligned_df = pd.DataFrame(aligned_segments)
            aligned_df.to_csv(aligned_path, index=False)
            
            # Create summary statistics
            total_segments = len(aligned_segments)
            total_slides = len(slide_transitions)
            avg_segments_per_slide = total_segments / total_slides if total_slides > 0 else 0
            
            # Create summary file
            summary_filename = f"{base_name}_alignment_summary.txt"
            summary_path = os.path.join(aligned_dir, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Alignment Summary for {video_filename}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total transcript segments: {total_segments}\n")
                f.write(f"Total slides: {total_slides}\n")
                f.write(f"Average segments per slide: {avg_segments_per_slide:.1f}\n\n")
                
                f.write("Slide-by-slide breakdown:\n")
                f.write("-" * 30 + "\n")
                
                for slide_num in range(1, total_slides + 1):
                    slide_segments = [s for s in aligned_segments if s['slide_number'] == slide_num]
                    slide_start = slide_transitions[slide_transitions['slide_number'] == slide_num]['start_time'].iloc[0]
                    slide_end = slide_transitions[slide_transitions['slide_number'] == slide_num]['end_time'].iloc[0]
                    
                    f.write(f"Slide {slide_num} ({slide_start:.1f}s - {slide_end:.1f}s): {len(slide_segments)} segments\n")
                    
                    # Show first few words of each segment
                    for i, segment in enumerate(slide_segments[:3]):  # Show first 3 segments
                        text_preview = segment['segment_text'][:50] + "..." if len(segment['segment_text']) > 50 else segment['segment_text']
                        f.write(f"  {i+1}. [{segment['segment_start']:.1f}s] {text_preview}\n")
                    
                    if len(slide_segments) > 3:
                        f.write(f"  ... and {len(slide_segments) - 3} more segments\n")
                    f.write("\n")
            
            print(f"✓ Aligned transcript saved to: {aligned_path}")
            print(f"✓ Summary saved to: {summary_path}")
            print(f"  {total_segments} segments aligned with {total_slides} slides")
            
            aligned_results.append({
                'video_file': video_filename,
                'aligned_file': aligned_path,
                'summary_file': summary_path,
                'total_segments': total_segments,
                'total_slides': total_slides,
                'avg_segments_per_slide': avg_segments_per_slide
            })
            
        except Exception as e:
            print(f"Error aligning transcript for {video_filename}: {e}")
            continue

    print(f"\nAlignment complete! Processed {len(aligned_results)} files.")
    return aligned_results