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

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\nIt is **red**.\n\nThe image shows a close-up of the front right side of the car, which is a classic Ferrari F40, a highly sought-after and iconic sports car. The vehicle is parked on a cobblestone surface in front of a brick house, with some green foliage in the background.",
                      "Based on the image provided, the car is a **Nissan GT-R**.\n\nIt is painted a vibrant **red**. The vehicle appears to be a modified version, possibly a high-performance model, given its aggressive styling, such as the large front splitter and the rear wing. The car is positioned at an angle, showcasing its sleek and aerodynamic design.",
                      "Based on the image provided, the car is a **Bugatti Chiron**.\n\nThe car is a **blue** color. It is a high-performance hypercar, and the image shows it on a road, likely in a scenic location. The iconic Bugatti logo is visible on the front grille.",
                      "This is a blue Nissan Micra, a compact car. It's a small, economical vehicle that was popular in the 1990s and early 2000s.",
                      "A person wearing a light green hoodie and light-colored pants is standing near a silver car with the driver's side door open."]

  prompts = ["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this? what color is it? what country was it made in?<|im_end|>\n<|im_start|>assistant\n",
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
    #assert output == expected_output

    prompt2 = f"<|im_start|>user\nwhat is the capital city of that country?<|im_end|>\n<|im_start|>assistant\n"
    output = qwen.forward(prompt=prompt2)
    print(output)

    prompt2 = f"<|im_start|>user\nwhat is that city's number 1 tourist attraction called?<|im_end|>\n<|im_start|>assistant\n"
    output = qwen.forward(prompt=prompt2)
    print(output)
    
    prompt2 = f"<|im_start|>user\ndoes it host events?<|im_end|>\n<|im_start|>assistant\n"
    output = qwen.forward(prompt=prompt2)
    print(output)

    exit()



