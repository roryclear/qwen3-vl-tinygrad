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

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\n-   **Color:** The car is a vibrant **red**.\n\nThe F40 is a classic Ferrari model that was built between 1987 and 1990. It is a legendary sports car known for its sleek design, powerful engine, and iconic status in the automotive world. The photograph shows it parked on a cobblestone surface in front of a brick building, with some greenery in the background.",
                      "Based on the image provided, the car is a **Nissan GT-R**.\n\nIt is a **red** color, with a glossy finish that reflects the studio lighting. The car is parked at an angle, showcasing its sleek design, aerodynamic body, and sporty features like a large rear wing and a prominent front splitter.",
                      "Based on the image provided, the car is a **Bugatti Chiron**.\n\nIt is a **blue** sports car. The vehicle is captured in motion on a scenic road, with a backdrop of green hills and a partly cloudy sky, which highlights its sleek design and powerful presence.",
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

  output = qwen.generate(prompt=prompts[0], image=images[0])
  print(output)

  output = qwen.generate(prompt=f"where was it made?")
  assert output == "Based on the visual characteristics and the specific model, the car in the image is a **Ferrari F40**.\n\nThe Ferrari F40 was produced by **Ferrari, the Italian manufacturer**. It was manufactured in two main locations:\n\n-   **1982**: The first production of the F40 began in **Bologna, Italy**. This was the primary production site for the model.\n-   **1983**: The production of the F40 continued at the **Maserati plant in Modena, Italy**.\n\nThe F40 was manufactured in Italy, which is why it is so well-regarded and highly sought after by collectors and enthusiasts."
  output = qwen.generate(prompt=f"what is the capital city of there?")
  assert output == "The capital city of Italy is **Rome**."
  output = qwen.generate(prompt=f"what is the best tourist attraction there? just give the number 1.")
  assert output == "1. The best tourist attraction in Rome is the **Colosseum**."


