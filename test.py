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

  expected_outputs = ["Based on the image provided, the car is a **Ferrari F40**.\n\nIt is a **red** color, and it's shown in a partial view, with the front portion of the car visible. The vehicle is parked on a cobblestone street in front of a brick building.",
                      "Based on the image provided, the car is a **Nissan GT-R**.\n\nIt is a **red** color. The car is shown in a studio setting with a gray background, and it appears to be a modified version of the GT-R, possibly a \"R35\" or a similar model with aftermarket enhancements to the rear spoiler.",
                      "Based on the image provided, the car is a **Bugatti Chiron**.\n\nIt is a **blue** color. The car is shown in motion on a road, with a scenic landscape in the background under a partly cloudy sky.",
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
  assert output == "The first car is a red Nissan GT-R, and the second is a blue Bugatti Chiron.\n\nThe second car is generally considered better because it is a hypercar with a higher performance and more advanced technology. However, the first car is a high-performance sports car with a strong reputation and is often considered one of the best in its class."

  # test image, prompt, image, prompt (about both)
  qwen.generate(image=images[1], prompt="what is this car?", reset=True)
  output = qwen.generate(image=images[2], prompt="what is this car? is it faster than the first one? be brief")
  assert output == "This is a **Bugatti Chiron**, a supercar known for its exceptional speed and luxury. The Chiron is indeed faster than the Nissan GT-R Nismo, with a top speed of around **320 km/h** (199 mph), compared to the GT-R Nismo's top speed of approximately **305 km/h** (190 mph)."

  # many prompts with context
  output = qwen.generate(prompt=f"where was the first one made?")
  assert output == "The **Nissan GT-R Nismo** was made in **Japan**."
  output = qwen.generate(prompt=f"what is the capital city of there?")
  assert output == "The capital city of Japan is **Tokyo**."
  output = qwen.generate(prompt=f"what is the best tourist attraction there? just give the number 1.")
  assert output == "1. **Tokyo Tower**"
