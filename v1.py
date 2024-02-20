# Import libraries
import cv2
import cvzone
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("model/best.pt")

# Open the video file
cap = cv2.VideoCapture("test/tokyo.mp4")

#Object classes
classNames = ['biker', 'car', 'pedestrian', 'traffic light', 'trafficlight-green', 'trafficlight-red', 'truck']

# Loop through the video frames
while True:
    # Read a frame from the video
    success, img = cap.read()
    results = model(img, stream=True)
    results = model.track(img, stream=True, persist=True, tracker="bytetrack.yaml") #or botsort.yaml

    # Initialize the boxes and other class names lists for each frame
    biker_boxes = []
    biker_id = 0
    car_boxes = []
    car_id = 0
    pedestrian_boxes = []
    pedestrian_id = 0
    traffic_light_boxes = []
    traffic_light_id = 0
    trafficlight_green_boxes = []
    trafficlight_green_id = 0
    trafficlight_red_boxes = []
    trafficlight_red_id = 0
    truck_boxes = []
    truck_id = 0


    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = box.conf[0]

            # Classname
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'biker':
                if conf > 0.5:
                    biker_id += 1
                    print(f"Biker ID: {biker_id}")
                    biker_boxes.append((x1, y1, w, h))  #1 Append the biker box

            if currentClass == 'car':
                if conf > 0.5:
                    car_id += 1
                    print(f"Car ID: {car_id}")
                    car_boxes.append((x1, y1, w, h))  #2 Append the car box

            if currentClass == 'pedestrian':
                if conf > 0.5:
                    pedestrian_id += 1
                    print(f"Pedestrian ID: {pedestrian_id}")
                    pedestrian_boxes.append((x1, y1, w, h))  #3 Append the Pedestrian box

            if currentClass == 'traffic light':
                if conf > 0.5:
                    traffic_light_id += 1
                    print(f"Traffic Light ID: {traffic_light_id}")
                    traffic_light_boxes.append((x1, y1, w, h))  #4 Append the Traffic Light box

            if currentClass == 'trafficlight-green':
                if conf > 0.5:
                    trafficlight_green_id += 1
                    print(f"Traffic Light-Green ID: {trafficlight_green_id}")
                    trafficlight_green_boxes.append((x1, y1, w, h))  #5 Append the Traffic Light-Green box

            if currentClass == 'trafficlight-red':
                if conf > 0.5:
                    trafficlight_red_id += 1
                    print(f"Traffic Light-Red ID: {trafficlight_red_id}")
                    trafficlight_red_boxes.append((x1, y1, w, h))  #6 Append the Traffic Light-Red box

            if currentClass == 'truck':
                if conf > 0.5:
                    truck_id += 1
                    print(f"Truck ID: {truck_id}")
                    truck_boxes.append((x1, y1, w, h))  #7 Append the Truck box


    #1 Display the biker boxes
    for biker_box, biker_id in zip(biker_boxes, range(1, biker_id + 1)):
        x1, y1, w, h = biker_box

        # Determine the color based on class presence
        color = (0, 0, 0)  # Default color is black         

        cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=1)
        cvzone.putTextRect(img, f'biker_id {biker_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=color,
                        colorT=(255, 255, 255), colorR=color, offset=5)

    #2 Display the car boxes
    for car_box, car_id in zip(car_boxes, range(1, car_id + 1)):
        x1, y1, w, h = car_box

        # Determine the color based on class presence
        color = (0, 0, 0)  # Default color is black         

        cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=1)
        cvzone.putTextRect(img, f'car_id {car_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=color,
                        colorT=(255, 255, 255), colorR=color, offset=5)

    #3 Display the Pedestrian boxes
    for pedestrian_box, pedestrian_id in zip(pedestrian_boxes, range(1, pedestrian_id + 1)):
        x1, y1, w, h = pedestrian_box

        # Determine the color based on class presence
        color = (0, 0, 0)  # Default color is black         

        cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=1)
        cvzone.putTextRect(img, f'pedestrian_id {pedestrian_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=color,
                        colorT=(255, 255, 255), colorR=color, offset=5)

    #4 Display the Traffic Light boxes
    for traffic_light_box, traffic_light_id in zip(traffic_light_boxes, range(1, traffic_light_id + 1)):
        x1, y1, w, h = traffic_light_box

        # Determine the color based on class presence
        color = (0, 0, 0)  # Default color is black         

        cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=1)
        cvzone.putTextRect(img, f'traffic_light_id {traffic_light_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=color,
                        colorT=(255, 255, 255), colorR=color, offset=5)

    #5 Display the Traffic Light-Green boxes
    for trafficlight_green_box, trafficlight_green_id in zip(trafficlight_green_boxes, range(1, trafficlight_green_id + 1)):
        x1, y1, w, h = trafficlight_green_box

        # Determine the color based on class presence
        color = (0, 0, 0)  # Default color is black         

        cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=1)
        cvzone.putTextRect(img, f'trafficlight_green_id {trafficlight_green_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=color,
                        colorT=(255, 255, 255), colorR=color, offset=5)
        

    #6 Display the Traffic Light-Red boxes
    for trafficlight_red_box, trafficlight_red_id in zip(trafficlight_red_boxes, range(1, trafficlight_red_id + 1)):
        x1, y1, w, h = trafficlight_red_box

        # Determine the color based on class presence
        color = (0, 0, 0)  # Default color is black         

        cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=1)
        cvzone.putTextRect(img, f'trafficlight_red_id {trafficlight_red_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=color,
                        colorT=(255, 255, 255), colorR=color, offset=5)
        

    #7 Display the truck boxes
    for truck_box, truck_id in zip(truck_boxes, range(1, truck_id + 1)):
        x1, y1, w, h = truck_box

        # Determine the color based on class presence
        color = (0, 0, 0)  # Default color is black         

        cvzone.cornerRect(img, (x1, y1, w, h), colorR=color, t=1)
        cvzone.putTextRect(img, f'truck_id {truck_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=color,
                        colorT=(255, 255, 255), colorR=color, offset=5)


    cv2.imshow("Image", img)
    # Exit the loop if the frame time has passed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()