from qwen3vl import Qwen3VL
import cv2
from tinygrad import Tensor, Variable
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

  expected_outputs = ["Based on the image provided, the vehicle is a **Ferrari F40**.\n\nThe car is **red**.",
                      "Based on the image provided, here is the information about the car:\n\n-   **Car Model**: This is a **Nissan GT-R**, specifically a model from the **Nissan R35** generation, which was produced from 2011 to 2017. The GT-R is known for its powerful performance and sleek design, and the \"R35\" designation is often used to refer to this specific generation.\n-   **Color**: The car is a vibrant **red** or **racing red**. The paint has a glossy finish, which is typical for cars in this class, especially when they are customized or painted for high-performance use.",
                      "Based on the visual details provided in the image, here is the information about the car:\n\n-   **Car Model**: This is a **Bugatti Chiron**. The distinctive features that point to this specific model include the \"BUGATTI\" logo on the front grille, the unique headlights, and the overall design of the body. The Chiron is a high-performance hypercar known for its speed and luxury.\n-   **Color**: The car is **blue**. It appears to be a vibrant, metallic blue, which is one of the most popular colors for Bugatti vehicles.",
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
    assert output == expected_output


