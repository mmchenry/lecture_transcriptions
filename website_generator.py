#!/usr/bin/env python3
"""
Website Generator for Lecture Materials
Converts transcriptions, slide transitions, and alignments into a structured website
"""

import os
import pandas as pd
import json
import re
from pathlib import Path
import shutil
from datetime import datetime


class LectureWebsiteGenerator:
    def __init__(self, data_root_path, output_dir):
        self.data_root_path = Path(data_root_path)
        self.output_dir = Path(output_dir)
        self.lectures = []
        
    def generate_website(self, catalog_path):
        """Generate the complete website from lecture materials"""
        print("Generating lecture website...")
        
        # Create output directory structure
        self._create_directory_structure()
        
        # Load catalog
        catalog = pd.read_csv(catalog_path)
        
        # Process each lecture
        for index, row in catalog.iterrows():
            video_filename = row['filename']
            base_name = os.path.splitext(video_filename)[0]
            
            print(f"Processing lecture: {base_name}")
            
            # Check if we have the required files
            if not self._check_required_files(base_name):
                print(f"  Skipping {base_name} - missing required files")
                continue
            
            # Generate lecture content
            lecture_data = self._process_lecture(base_name, row)
            if lecture_data:
                self.lectures.append(lecture_data)
        
        # Generate index and navigation
        self._generate_index_page()
        self._generate_navigation()
        self._generate_styles()
        self._generate_html_preview()
        
        print(f"Website generated successfully in: {self.output_dir}")
        return self.lectures
    
    def _create_directory_structure(self):
        """Create the website directory structure"""
        dirs = [
            self.output_dir,
            self.output_dir / "lectures",
            self.output_dir / "assets",
            self.output_dir / "assets" / "css",
            self.output_dir / "assets" / "js",
            self.output_dir / "assets" / "images",
            self.output_dir / "assets" / "slides",
            self.output_dir / "assets" / "slide_images"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy slide images to website assets
        self._copy_slide_images()
    
    def _copy_slide_images(self):
        """Copy slide images from data directory to website assets"""
        source_dir = self.data_root_path / "slide_images"
        target_dir = self.output_dir / "assets" / "slide_images"
        
        if not source_dir.exists():
            print(f"Slide images directory not found: {source_dir}")
            return
        
        print("Copying slide images to website assets...")
        copied_count = 0
        
        for image_file in source_dir.glob("*.png"):
            target_file = target_dir / image_file.name
            try:
                import shutil
                shutil.copy2(image_file, target_file)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {image_file.name}: {e}")
        
        print(f"Copied {copied_count} slide images")
    
    def _check_required_files(self, base_name):
        """Check if required files exist for a lecture"""
        # Try different filename variations
        possible_transcript_files = [
            f"{base_name}_transcript.json",
            f"{base_name}_transcr ipt.json",  # Handle space in filename
            f"{base_name}_transcript.json"
        ]
        
        possible_transition_files = [
            f"{base_name}.csv",
            f"{base_name}.csv"
        ]
        
        # Check if any combination exists
        transcript_exists = any((self.data_root_path / "transcriptions" / f).exists() 
                               for f in possible_transcript_files)
        transition_exists = any((self.data_root_path / "slide_transitions" / f).exists() 
                               for f in possible_transition_files)
        
        return transcript_exists and transition_exists
    
    def _process_lecture(self, base_name, catalog_row):
        """Process a single lecture and generate its content"""
        try:
            # Find actual transcript file
            transcript_path = None
            possible_transcript_files = [
                f"{base_name}_transcript.json",
                f"{base_name}_transcr ipt.json",  # Handle space in filename
            ]
            
            for filename in possible_transcript_files:
                test_path = self.data_root_path / "transcriptions" / filename
                if test_path.exists():
                    transcript_path = test_path
                    break
            
            if transcript_path is None:
                print(f"  Transcript file not found for {base_name}")
                return None
            
            # Load transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Load slide transitions
            transitions_path = self.data_root_path / "slide_transitions" / f"{base_name}.csv"
            if not transitions_path.exists():
                print(f"  Slide transitions file not found: {transitions_path}")
                return None
                
            slide_transitions = pd.read_csv(transitions_path)
            
            # Load aligned transcript if available
            aligned_path = self.data_root_path / "aligned_transcripts" / f"{base_name}_aligned.csv"
            aligned_data = None
            if aligned_path.exists():
                aligned_data = pd.read_csv(aligned_path)
            
            # Generate lecture content
            lecture_content = self._generate_lecture_content(
                base_name, transcript_data, slide_transitions, aligned_data, catalog_row
            )
            
            # Save lecture file
            lecture_file = self.output_dir / "lectures" / f"{base_name}.md"
            with open(lecture_file, 'w', encoding='utf-8') as f:
                f.write(lecture_content)
            
            return {
                'filename': f"{base_name}.md",
                'title': self._extract_lecture_title(base_name),
                'lecture_num': catalog_row.get('lecture_num', 0),
                'duration': self._calculate_duration(transcript_data),
                'slide_count': len(slide_transitions),
                'word_count': len(transcript_data.get('text', '').split())
            }
            
        except Exception as e:
            print(f"  Error processing {base_name}: {e}")
            return None
    
    def _generate_lecture_content(self, base_name, transcript_data, slide_transitions, aligned_data, catalog_row):
        """Generate markdown content for a lecture"""
        lecture_title = self._extract_lecture_title(base_name)
        full_text = transcript_data.get('text', '')
        segments = transcript_data.get('segments', [])
        
        # Start building the markdown content
        content = []
        
        # Header
        content.append(f"# {lecture_title}")
        content.append("")
        
        # Metadata
        duration = self._calculate_duration(transcript_data)
        content.append(f"**Duration:** {duration:.1f} minutes")
        content.append(f"**Slides:** {len(slide_transitions)}")
        content.append(f"**Words:** {len(full_text.split())}")
        content.append("")
        
        # Slide-by-slide breakdown (main content)
        content.append("## Lecture Content")
        content.append("")
        for i, slide in slide_transitions.iterrows():
            slide_num = slide.get('slide_number', i+1)
            start_time = slide.get('start_time', 0)
            end_time = slide.get('end_time', 0)
            
            # Slide image
            content.append(f"![Slide {int(slide_num)}](./assets/slide_images/{base_name}_slide_{int(slide_num):03d}.png)")
            content.append("")
            
            # Content for this slide
            slide_text = ""
            if aligned_data is not None:
                slide_segments = aligned_data[aligned_data['slide_number'] == slide_num]
                if not slide_segments.empty:
                    # Combine all text for this slide
                    slide_texts = []
                    for _, segment in slide_segments.iterrows():
                        text = segment['segment_text'].strip()
                        if text:
                            slide_texts.append(text)
                    slide_text = " ".join(slide_texts)
            
            # If no aligned data, extract text from transcript segments
            if not slide_text:
                slide_text = self._extract_text_for_slide(segments, start_time, end_time)
            
            if slide_text:
                content.append("**Content:**")
                content.append("")
                content.append(slide_text)
                content.append("")
            
            # Key points (extracted from transcript)
            key_points = self._extract_key_points(segments, start_time, end_time)
            if key_points:
                content.append("**Key Points:**")
                content.append("")
                for point in key_points:
                    content.append(f"- {point}")
                content.append("")
            
            # Add some spacing between slides instead of separators
            content.append("")
            content.append("")
        
        return "\n".join(content)
    
    def _extract_lecture_title(self, base_name):
        """Extract a readable title from the filename"""
        # Remove lecture number and clean up
        title = re.sub(r'^\d+\.?\d*\s*', '', base_name)
        title = title.replace('_', ' ').replace('-', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        return title
    
    def _calculate_duration(self, transcript_data):
        """Calculate lecture duration in minutes"""
        if 'segments' in transcript_data and transcript_data['segments']:
            last_segment = transcript_data['segments'][-1]
            duration_seconds = last_segment.get('end', 0)
            return duration_seconds / 60
        return 0
    
    def _extract_text_for_slide(self, segments, start_time, end_time):
        """Extract text content for a specific slide from transcript segments"""
        slide_texts = []
        
        for segment in segments:
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            
            # Check if segment overlaps with slide timing
            if segment_start >= start_time and segment_start < end_time:
                text = segment.get('text', '').strip()
                if text:
                    slide_texts.append(text)
        
        return " ".join(slide_texts)
    
    def _extract_key_points(self, segments, start_time, end_time):
        """Extract key points from transcript segments within a time range"""
        key_points = []
        
        for segment in segments:
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            
            # Check if segment overlaps with slide timing
            if segment_start >= start_time and segment_start < end_time:
                text = segment.get('text', '').strip()
                
                # Look for key phrases that might indicate important points
                key_indicators = [
                    'important', 'key', 'main', 'primary', 'essential', 'critical',
                    'remember', 'note', 'notice', 'observe', 'consider',
                    'first', 'second', 'third', 'finally', 'conclusion'
                ]
                
                if any(indicator in text.lower() for indicator in key_indicators):
                    # Clean up the text
                    clean_text = re.sub(r'^\s*[-•]\s*', '', text)
                    if len(clean_text) > 10:  # Only include substantial points
                        key_points.append(clean_text)
        
        return key_points[:5]  # Limit to 5 key points per slide
    
    def _generate_index_page(self):
        """Generate the main index page"""
        content = []
        content.append("# Lecture Materials")
        content.append("")
        content.append("Welcome to the lecture materials website. This site contains transcribed lectures with slide-by-slide breakdowns.")
        content.append("")
        
        # Sort lectures by lecture number
        sorted_lectures = sorted(self.lectures, key=lambda x: x.get('lecture_num', 0))
        
        content.append("## Available Lectures")
        content.append("")
        
        for lecture in sorted_lectures:
            title = lecture['title']
            filename = lecture['filename']
            duration = lecture['duration']
            slide_count = lecture['slide_count']
            word_count = lecture['word_count']
            
            content.append(f"### [{title}](lectures/{filename})")
            content.append("")
            content.append(f"- **Duration:** {duration:.1f} minutes")
            content.append(f"- **Slides:** {slide_count}")
            content.append(f"- **Words:** {word_count:,}")
            content.append("")
        
        # Save index
        index_file = self.output_dir / "index.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
    
    def _generate_navigation(self):
        """Generate navigation files"""
        # Create a simple navigation structure
        nav_content = []
        nav_content.append("---")
        nav_content.append("title: Lecture Navigation")
        nav_content.append("---")
        nav_content.append("")
        
        for lecture in sorted(self.lectures, key=lambda x: x.get('lecture_num', 0)):
            title = lecture['title']
            filename = lecture['filename']
            nav_content.append(f"- [{title}](lectures/{filename})")
        
        nav_file = self.output_dir / "_navigation.md"
        with open(nav_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(nav_content))
    
    def _generate_styles(self):
        """Generate CSS styles for the website"""
        css_content = """
/* Lecture Website Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    background-color: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}

h2 {
    color: #34495e;
    margin-top: 30px;
}

h3 {
    color: #7f8c8d;
    border-left: 4px solid #3498db;
    padding-left: 15px;
}

.slide-content {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 20px;
    margin: 20px 0;
}

.slide-image {
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 5px;
    margin: 10px 0;
}

.metadata {
    background-color: #e9ecef;
    padding: 15px;
    border-radius: 5px;
    margin: 20px 0;
}

.key-points {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 15px;
    margin: 15px 0;
}

.toc {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 20px;
    margin: 20px 0;
}

.toc ul {
    list-style-type: none;
    padding-left: 0;
}

.toc li {
    padding: 5px 0;
    border-bottom: 1px solid #eee;
}

.toc a {
    text-decoration: none;
    color: #495057;
}

.toc a:hover {
    color: #007bff;
}
"""
        
        css_file = self.output_dir / "assets" / "css" / "style.css"
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(css_content)

    def _generate_html_preview(self):
        """Generate a simple HTML preview of the lectures"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lecture Materials Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .lecture { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
        .slide { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        .slide img { max-width: 100%; height: auto; border: 1px solid #ccc; }
        .metadata { background: #e9ecef; padding: 10px; border-radius: 5px; margin: 10px 0; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; }
        h3 { color: #7f8c8d; }
    </style>
</head>
<body>
    <h1>Lecture Materials Preview</h1>
"""
        
        for lecture in self.lectures:
            lecture_file = self.output_dir / "lectures" / lecture['filename']
            if lecture_file.exists():
                with open(lecture_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert markdown to simple HTML
                import re
                
                # Convert headers
                content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
                content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
                content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
                
                # Convert bold
                content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
                
                # Convert italics
                content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
                
                # Convert lists
                content = re.sub(r'^- (.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
                
                # Convert images (fix paths)
                content = re.sub(r'!\[(.+?)\]\(\./assets/slide_images/(.+?)\)', 
                               r'<img src="./assets/slide_images/\2" alt="\1" style="max-width: 100%; height: auto;">', content)
                
                # Convert paragraphs
                content = re.sub(r'^([^<].+)$', r'<p>\1</p>', content, flags=re.MULTILINE)
                
                html_content += f"""
    <div class="lecture">
        {content}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML file
        html_file = self.output_dir / "preview.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML preview generated: {html_file}")


def generate_lecture_website(catalog_path, data_root_path):
    """Main function to generate the lecture website"""
    output_dir = os.path.join(data_root_path, 'lecture_website')  # Change this line to your preferred default
    os.makedirs(output_dir, exist_ok=True)
    generator = LectureWebsiteGenerator(data_root_path, output_dir)
    return generator.generate_website(catalog_path)
