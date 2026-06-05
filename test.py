from qwen3vl import Qwen3VL
import cv2
if __name__ == "__main__":
  qwen = Qwen3VL(size="2B", res=(256, 256))

  # first three are all 256x256
  images = [
      cv2.cvtColor(cv2.imread("images/f40.jpeg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("images/gtr.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("images/bug.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("images/micra.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("images/96_notif.jpg"), cv2.COLOR_BGR2RGB)
  ]

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\nIt is a **red** car. The vibrant red color is a signature of the Ferrari F40, a legendary sports car that was produced in the 1980s. The vehicle is a high-performance, mid-engine sports car known for its sleek design and exceptional handling, with the iconic \"ferrari\" badge visible on the front grille.",
                      "Based on the image provided, the car is a **Nissan GT-R** (specifically, the 2018 model with a \"GT-R\" badge, which is a common style for the Nissan GT-R in that era).\n\nIt is a **red** car. The vibrant red paint job is a signature of the Nissan GT-R, a high-performance sports car known for its powerful engine and agile handling. The vehicle is presented in a studio setting, which highlights its sleek design and aerodynamic features.",
                      "Based on the image provided, the car is a **Bugatti Chiron**.\n\nIt is a **blue** car. The vibrant blue color is a signature of the Bugatti Chiron, a high-performance hypercar known for its exceptional speed and luxurious design. The vehicle is captured in a dynamic pose, emphasizing its sleek lines and powerful presence on a scenic road.",
                      "This is a blue Nissan Micra, a compact car. It's a small, economical vehicle that was popular in the 1990s and early 2000s.",
                      "A person wearing a light green hoodie and light-colored pants is standing near a silver car with the driver's side door open."]

  prompts = ["What car is this? what color is it?",
             "What car is this? what color is it?",
             "What car is this? what color is it?",
             "",
          "",
          ""]

  z = 0
  qwen.prewarm()
  for image, expected_output, prompt in zip(images, expected_outputs, prompts):
    z += 1
    if z > 3: continue
    output = qwen.generate(prompt=prompt, image=image)
    assert output == expected_output
  
  # test two images in prefill
  qwen.generate(image=images[1], reset=True)
  output = qwen.generate(prompt="What is the first car? is it better than the second one? be brief", image=images[2])
  print("output =",output)
  assert output == "The first car shown is a red Nissan GT-R, and the second is a blue Bugatti Chiron. The Nissan GT-R is a high-performance sports car known for its powerful engine and handling, while the Bugatti Chiron is a high-end, luxury supercar with advanced technology and speed. The GT-R is generally considered to be more powerful and faster than the Chiron, but the Chiron is more expensive and has a more luxurious interior. So, it depends on what you value most."

  # test image, prompt, image, prompt (about both)
  qwen.generate(image=images[1], prompt="what is this car?", reset=True)
  output = qwen.generate(image=images[2], prompt="what is this car? is it faster than the first one? be brief")
  assert output == "This car is a **2024 Bugatti Chiron**, which is significantly faster than the first Nissan GT-R Nismo shown in the first image.\n\nThe Bugatti Chiron is a hypercar known for its incredible speed and performance, with a 0-60 mph time of just 2.3 seconds, while the Nissan GT-R Nismo has a 0-60 mph time of around 3.5 seconds. The Chiron is also faster and more powerful than the first car."

  # many prompts with context
  output = qwen.generate(prompt=f"where was the first one made?")
  assert output == "The first car shown in the image, the **Nissan GT-R Nismo**, was made in **Japan**.\n\nNissan is a Japanese manufacturer, and the GT-R Nismo is a model produced by Nissan's Japanese operations. It is also notable that the GT-R Nismo was launched in 2012 and is known for its high-performance capabilities and aggressive styling."
  output = qwen.generate(prompt=f"what is the capital city of there?")
  assert output == "The capital city of Japan is **Tokyo**."
  output = qwen.generate(prompt=f"what is the best tourist attraction there? just give the number 1.")
  assert output == "1. **Tokyo Tower**"
