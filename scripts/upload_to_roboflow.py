from roboflow import Roboflow
import glob

rf = Roboflow(api_key="YOUR_API_KEY")  # Get from Roboflow settings
project = rf.workspace().project("newcastle-traffic-detection")

# Upload all daytime images
images = sorted(glob.glob("data/dataset/*_1[0-6]*.jpg"))
print(f"Uploading {len(images)} daytime images...")

for img in images:
    project.upload(img)
    print(f"  Uploaded: {img}")

print("Done! Go to Roboflow to start labeling.")