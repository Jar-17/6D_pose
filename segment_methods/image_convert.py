from PIL import Image
image_base = "haoliang.png"
image_name = image_base.split(".")[0]
img = Image.open(f"data/{image_base}").convert("RGB")
img.save(f"data/{image_name}-clean.jpg")
