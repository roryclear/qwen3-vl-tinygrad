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

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\nIt is **red**.\n\nThe image shows a close-up of the front right side of the car, which is a classic Ferrari F40, a highly sought-after and iconic sports car. The vehicle is parked on a cobblestone surface in front of a brick house, with some green foliage in the background.",
                      "Based on the image provided, the car is a **Nissan GT-R**.\n\nIt is a **red** (or a vibrant, glossy red) high-performance sports car. The vehicle in the photo is a modified version, likely a **GT-R Nismo**, which features a more aggressive and sporty design, including a large rear wing and a front splitter, which are common on cars with enhanced performance capabilities.",
                      "Based on the image provided, the car is a **Bugatti Chiron**.\n\nIt is painted a vibrant **blue** color. The vehicle is captured in motion on a scenic road, and its distinctive features, such as the large front grille, the sleek, low-profile body, and the aggressive headlights, are characteristic of the Bugatti Chiron model.",
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
    qwen.pos = 0
    z += 1
    if z > 3: continue
    
    output = qwen.forward(prompt=prompt, image=image)
    print("output =",output)
    #assert output == expected_output
    prompt2 = f"<|im_start|>user\nwhat is its top speed? just give me the figure in kph only<|im_end|>\n<|im_start|>assistant\n"
    output2 = qwen.forward(prompt=prompt2)
    print("output =",output2)