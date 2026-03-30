import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        raise ValueError("No frames to save")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_height, frame_width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def frame_stream(video_path, batch_size=500, overlap=30):
    """
    Generator that yields (frames, start_frame_idx, fps, width, height, is_last) tuples.
    Each batch has up to batch_size frames.
    Consecutive batches overlap by `overlap` frames to handle boundary effects.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buffer = []
    global_idx = 0
    batch_start_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)
        global_idx += 1

        if len(buffer) == batch_size:
            is_last = (global_idx >= total_frames)
            yield buffer.copy(), batch_start_idx, fps, frame_width, frame_height, is_last
            # Keep last `overlap` frames as start of next batch
            buffer = buffer[-overlap:]
            batch_start_idx = global_idx - overlap
            if is_last:
                break

    # Yield remaining frames if any
    if buffer:
        yield buffer, batch_start_idx, fps, frame_width, frame_height, True

    cap.release()


def get_video_info(video_path):
    """Return (fps, total_frames, width, height) without loading frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height