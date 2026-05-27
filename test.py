from qwen3vl import Qwen3VL
from tinygrad import Tensor, Variable
import cv2
if __name__ == "__main__":
  qwen = Qwen3VL(size="2B")

  # first four are all 256x256
  images = [
      cv2.cvtColor(cv2.imread("images/f40.jpeg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("images/gtr.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("images/bug.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("images/micra.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("images/96_notif.jpg"), cv2.COLOR_BGR2RGB)
  ]

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\nIt is a **red** color. The vehicle is a classic supercar, and the image is a photograph of it parked on a cobblestone surface in front of a brick house.",
                      "Based on the image provided, the car is a **Nissan GT-R**, specifically the **GTR R35** model from the 2014-2019 production run. This is a high-performance sports car known for its powerful engine and distinctive design.\n\nThe car is **red**.",
                      "Based on the image provided, the car is a **Bugatti Chiron**.\n\nIt is a **blue** sports car. This particular model is known for its vibrant blue paint, which is a signature of the Bugatti brand's high-performance vehicles.",
                      "This is a blue Nissan Micra, a compact car. It's a small, economical vehicle that was popular in the 1990s and early 2000s.",
                      "A person wearing a light green hoodie and light-colored pants is standing near a silver car with the driver's side door open."]

  prompts = ["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this? what color is it?<|im_end|>\n<|im_start|>assistant\n",
             "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this? what color is it?<|im_end|>\n<|im_start|>assistant\n",
             "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this? what color is it?<|im_end|>\n<|im_start|>assistant\n",
             "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this? what color is it?<|im_end|>\n<|im_start|>assistant\n",
          "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nTell me the history of this car<|im_end|>\n<|im_start|>assistant\n",
          "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat has been detected on my CCTV camera? Write in one short sentence, only info about the object(s) detected.<|im_end|>\n<|im_start|>assistant\n"]

  z = 0
  qwen.prewarm(images[0].shape, prompts[0])
  for image, expected_output, prompt in zip(images, expected_outputs, prompts):
    z += 1
    if z > 3: continue
    
    output = qwen.forward(prompt=prompt, image=image)
    print("output =",output)
    toks_out = []
    prompt2 = f"<|im_start|>user\nis it fast?<|im_end|>\n<|im_start|>assistant\n"
    prompt2 = qwen.tok.encode(prompt2)
    start_pos = len(prompt2)
    tok = qwen.lang.prefill_jit(tokens=Tensor(prompt2).unsqueeze(0), start_pos=Variable("pos",1,2000).bind(start_pos), temperature=Tensor(0.7).clone())
    print("tok =",tok.numpy(), qwen.tok.decode(tok.numpy()[0]))
    for i in range(10):
      tok = qwen.lang.rollout_jit(tokens=tok.clone(), start_pos=Variable("pos",1,2000).bind(start_pos), temperature=Tensor(0.7).clone())
      start_pos+=1
      print("tok =",tok.numpy()[0][0], qwen.tok.decode(tok.numpy()[0]))
    assert output == expected_output
