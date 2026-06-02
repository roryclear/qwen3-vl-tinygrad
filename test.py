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

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\nIt is a **red** sports car. The vehicle is parked on a cobblestone surface, and the image captures it from a front three-quarter angle, highlighting its iconic design featuring a large rear wing and a distinctive front grille. The car is positioned in front of a brick building and some green foliage.",
                      "Based on the image provided, the car is a **Nissan GT-R**.\n\nThe car is painted a vibrant **red**. It is a high-performance sports car, and the image appears to be a professional studio photograph, likely used for promotional or advertising purposes.",
                      "This is a Bugatti Veyron, and it is blue.",
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
  assert output == "The Ferrari F40 was made in **Italy**.\n\nSpecifically, it was manufactured by **Ferrari's factory in Maranello, Italy**, which is the company's primary production center. The F40 was produced from 1987 to 1992, and it was the first Ferrari to use the \"F40\" nameplate, which was also used for the F40 and F430 models. The F40 was designed and built with the aim of producing a car that was both fast and elegant, and it has since become a highly sought-after classic."
  output = qwen.generate(prompt=f"what is the capital city of there?")
  assert output == "The capital city of Italy is **Rome**.\n\nIt is the largest city in Italy and the political, cultural, and economic center of the country."
  output = qwen.generate(prompt=f"what is the best tourist attraction there? just give the number 1.")
  assert output == "1. The Colosseum"


