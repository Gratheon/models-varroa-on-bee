from ultralytics import YOLO
import sys

if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]

# Load the trained model (using the path from your successful training output)
model = YOLO('/Users/artjom/git/gratheon/entrance-observer/runs/detect/varroa_detection/varroa_model5/weights/best.pt')

# Run inference
results = model(image_path)

# Display results
for result in results:
    result.show()  # Display to screen
    # result.save(filename='result.jpg')  # Save to disk

print("Prediction complete.")