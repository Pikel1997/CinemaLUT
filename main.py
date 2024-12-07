import cv2
import numpy as np
import colorsys

def extract_color_profile(video_path, sample_count=1000):
    """
    Extract color characteristics from a video

    Args:
    video_path (str): Path to the input video
    sample_count (int): Number of frames to sample

    Returns:
    dict: Color analysis statistics
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame sampling
    frame_step = max(1, total_frames // sample_count)

    # Color storage
    colors = []

    # Sample frames
    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (200, 200))

        # Flatten the frame and convert to RGB
        pixels = frame.reshape(-1, 3)

        # Convert BGR to RGB
        pixels = pixels[:, ::-1]

        # Normalize colors
        pixels = pixels.astype(np.float32) / 255.0

        colors.extend(pixels)

    cap.release()

    # Convert to numpy array
    colors = np.array(colors)

    # Compute color statistics
    return {
        'mean_color': np.mean(colors, axis=0),
        'std_color': np.std(colors, axis=0),
        'dominant_colors': _get_dominant_colors(colors)
    }

def _get_dominant_colors(colors, num_colors=5):
    """
    Extract dominant colors using K-means clustering

    Args:
    colors (np.array): Input colors
    num_colors (int): Number of dominant colors to extract

    Returns:
    list: Dominant colors
    """
    # Reshape colors for clustering
    pixels = colors.reshape(-1, 3)

    # Perform K-means clustering
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)

    # Get cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_

    return dominant_colors

def generate_custom_lut(color_profile, size=33):
    """
    Generate a LUT based on video color profile

    Args:
    color_profile (dict): Color analysis from the video
    size (int): Size of the 3D LUT grid

    Returns:
    list: Lines of the LUT file
    """
    # Create LUT file header
    lut_lines = [
        f"TITLE \"Custom Video LUT\"",
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0"
    ]

    # Use color profile to adjust transformation
    mean_color = color_profile['mean_color']
    std_color = color_profile['std_color']
    dominant_colors = color_profile['dominant_colors']

    def color_transform(r, g, b):
        """
        Custom color transformation based on video analysis
        """
        # Apply some color shifts based on video characteristics
        shift_r = mean_color[0] * 0.5
        shift_g = mean_color[1] * 0.5
        shift_b = mean_color[2] * 0.5

        # Adjust contrast and color based on standard deviation
        contrast_r = 1 + std_color[0]
        contrast_g = 1 + std_color[1]
        contrast_b = 1 + std_color[2]

        # Transform colors
        new_r = np.clip((r * contrast_r) + shift_r, 0, 1)
        new_g = np.clip((g * contrast_g) + shift_g, 0, 1)
        new_b = np.clip((b * contrast_b) + shift_b, 0, 1)

        return new_r, new_g, new_b

    # Generate color mapping
    for b in np.linspace(0, 1, size):
        for g in np.linspace(0, 1, size):
            for r in np.linspace(0, 1, size):
                # Transform the color
                new_r, new_g, new_b = color_transform(r, g, b)

                # Format the line with 6 decimal places
                lut_lines.append(f"{new_r:.6f} {new_g:.6f} {new_b:.6f}")

    return lut_lines

def video_to_lut(video_path, output_lut='custom_video_lut.cube'):
    """
    Convert video to LUT

    Args:
    video_path (str): Path to input video
    output_lut (str): Path to output LUT file
    """
    # Extract color profile
    color_profile = extract_color_profile(video_path)

    # Generate LUT
    lut_lines = generate_custom_lut(color_profile)

    # Save LUT file
    with open(output_lut, 'w') as f:
        for line in lut_lines:
            f.write(line + '\n')

    print(f"LUT file generated: {output_lut}")

    # Print color profile for reference
    print("\nColor Profile:")
    print("Mean Color:", color_profile['mean_color'])
    print("Color Std Deviation:", color_profile['std_color'])
    print("Dominant Colors:")
    for color in color_profile['dominant_colors']:
        print(color)

# Example usage
if __name__ == "__main__":
    # Replace with your video path
    video_path = "path-to-your-file.mp4"
    video_to_lut(video_path)
