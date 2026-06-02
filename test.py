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

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\nIt is a **red** sports car. The vehicle is parked on a cobblestone surface, and the image captures it from a front three-quarter angle, highlighting its iconic design featuring a large rear wing and a sleek, aerodynamic body.",
                      "Based on the image provided, the car is a **Nissan GT-R**.\n\nThe car is **red** in color. It appears to be a modified version, possibly a high-performance variant like the Nissan GT-R Nismo, given the aggressive front grille, large rear spoiler, and black racing-style wheels.",
                      "Based on the image provided, the car is a **Bugatti Chiron**.\n\nIt is a **blue** sports car.",
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

  output = qwen.generate(prompt=f"where was it made?")
  assert output == "The Ferrari F40 was made in **Italy**.\n\nSpecifically, it was manufactured at the **Ferrari factory in Maranello, Italy**. This location is in the Lago di Como region of the Lombardy province, in the northern part of the country. The F40 is a production car that was created as a limited-edition, high-performance model, and it's famously known for its innovative design and engineering."
  output = qwen.generate(prompt=f"what is the capital city of there?")
  assert output == "The capital city of Italy is **Rome**."
  output = qwen.generate(prompt=f"what is the best tourist attraction there? just give the number 1.")
  assert output == "1. The Colosseum"


