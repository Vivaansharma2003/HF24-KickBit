from ultralytics import YOLO


data_dir = "litter_dataset"   
output_dir = "litter_detection_weights"  
num_classes = 1  
model = YOLO("yolov8s.pt")  

model.freeze() 


with open("data.yaml", "w") as f:
    f.write(f"train: {data_dir}/images\n")
    f.write(f"val: {data_dir}/test_images\n")  
    f.write(f"nc: {num_classes}\n")
    f.write("names: ['litter']\n")  

# Train the model
model.train(
    data="data.yaml",  
    epochs=100, 
    batch_size=8,  
    imgsz=416, 
    accumulate=4, 
)


model.save(output_dir)