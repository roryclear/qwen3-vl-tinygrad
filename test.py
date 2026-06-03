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

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\nIt is a **red** color, with a glossy finish. The car is a classic model from the 1980s, known for its iconic design and performance.",
                      "Based on the image provided, the car is a **Nissan GT-R**.\n\nIt is a **red** color, with a vibrant, glossy finish. The car is shown in a modern, high-performance style, and the image appears to be a studio photograph, as it is set against a plain gray background.",
                      "The car in the image is a **Bugatti Chiron**.\n\nIt is a **blue** color, with a glossy finish. The vehicle is a high-performance supercar, known for its distinctive design and powerful engine, and is captured in a dynamic, motion-filled scene on a paved road under a bright blue sky.",
                      "This is a blue Nissan Micra, a compact car. It's a small, economical vehicle that was popular in the 1990s and early 2000s.",
                      "A person wearing a light green hoodie and light-colored pants is standing near a silver car with the driver's side door open."]

  prompts = ["What car is this? what color is it?",
             "What car is this? what color is it?",
             "What car is this? what color is it?",
             "",
          "",
          ""]

  z = 0
  qwen.prewarm(images[0].shape)
  for image, expected_output, prompt in zip(images, expected_outputs, prompts):
    z += 1
    if z > 3: continue
    output = qwen.generate(prompt=prompt, image=image)
    assert output == expected_output
  
  # test two images in prefill
  qwen.generate(image=images[1], reset=True)
  output = qwen.generate(prompt="What is the first car? is it better than the second one? be brief", image=images[2])
  assert output == "The first car is a red Nissan GT-R, and the second is a blue Bugatti Chiron.\n\nThe Nissan GT-R is not better than the Bugatti Chiron. The Bugatti Chiron is a more expensive and faster vehicle, but it has a higher price tag and is not as reliable as the Nissan GT-R. The Nissan GT-R has a more powerful engine and is more reliable, but it is not as fast as the Bugatti Chiron."
  output = qwen.generate(prompt=f"where was the first one made?")
  assert output == "The red car in the image is a Nissan GT-R, and it was made in Japan."
  output = qwen.generate(prompt=f"what is the capital city of there?")
  assert output == "The capital city of Japan is Tokyo."
  output = qwen.generate(prompt=f"what is the best tourist attraction there? just give the number 1.")
  assert output == "1. Tokyo Tower"
