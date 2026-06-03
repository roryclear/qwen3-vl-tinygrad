import torch
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
import cv2

# ---- Load model + processor ----
model_id = "Qwen/Qwen3-VL-2B-Instruct"  

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
model = model.to("cpu")

# ---- Load video frames ----
def load_video_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # sample evenly
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

video_path = "images/goal.mp4"
frames = load_video_frames(video_path)

# ---- Prepare messages with video placeholder ----
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": "Describe what is happening in this video."}
        ]
    }
]

# ---- Tokenize using chat template ----
# This properly formats the prompt with video tokens
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=text,
    videos=frames,
    return_tensors="pt"
)

print("rory inputs =",inputs)

# ---- Generate ----
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )

# ---- Decode ----
response = processor.decode(output[0], skip_special_tokens=True)
print("output =",response)