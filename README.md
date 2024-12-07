# 🎨 CinemaLUT: Video Color Essence Extractor

## Overview
CinemaLUT is an innovative Python-powered tool that transforms video footage into custom color lookup tables (LUTs), allowing filmmakers, colorists, and content creators to capture and reproduce the unique color essence of any video.

## Features
- 🎥 Extracts color profile from input video
- 🌈 Generates a professional-grade 33x33x33 color lookup table
- 📊 Performs advanced color analysis using machine learning
- 🖌️ Creates custom color transformations based on video characteristics

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- scikit-learn

### Install Dependencies
```bash
pip install opencv-python numpy scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/CinemaLUT.git
cd CinemaLUT
```

## Usage

### Basic Usage
```python
from video_lut_generator import video_to_lut

# Generate LUT from your video
video_to_lut('path/to/your/video.mp4', 'output_lut.cube')
```

### Advanced Configuration
- Customize sampling count
- Adjust dominant color extraction
- Fine-tune color transformation parameters

## How It Works
1. Sample frames from the input video
2. Analyze color distribution using K-means clustering
3. Extract mean color, standard deviation, and dominant colors
4. Generate a custom 33x33x33 LUT based on video characteristics

## Example Outputs
- Capture the mood of a film
- Reproduce cinematographic color styles
- Create unique color grading presets

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License

## Disclaimer
CinemaLUT provides an approximation of video color characteristics. Professional color grading may require manual refinement.

## Contact
- Created by [Pikel1997]
- Email: kunalkamlesh1212@gmail.com
- Project Link: https://github.com/Pikel1997/CinemaLUT
