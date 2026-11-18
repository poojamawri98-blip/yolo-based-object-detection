from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")


cap = cv2.VideoCapture("videos/trafficvideo.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print("Original FPS:", fps)


output = cv2.VideoWriter(
    "output_videonew.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

print("Video opened:", cap.isOpened())

# Create a resizable window
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 1200, 700)

while True:

    success, frame = cap.read()
   # print("Current frame:", int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    if not success:
        break

    
    results = model(
        frame,
        conf=0.35,
        imgsz=992
    )

    person_count = 0
    vehicle_count = 0

    # Check every detected object
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls)
            object_name = model.names[class_id]

         
            if object_name == "person":
                person_count += 1

           
            elif object_name in ["car", "bus", "truck"]:
                vehicle_count += 1

           
            if object_name in ["person", "car", "bus", "truck"]:

                x1, y1, x2, y2 = box.xyxy[0]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    object_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

    print("Processed frames:", int(cap.get(cv2.CAP_PROP_POS_FRAMES)))


    cv2.putText(
    frame,
    f"Persons : {person_count}",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    2,
    (0, 255, 0),
    4
)

    cv2.putText(
    frame,
    f"Vehicles : {vehicle_count}",
    (20, 110),
    cv2.FONT_HERSHEY_SIMPLEX,
    2,
    (0, 255, 255),
    4
)

    output.write(frame)

    cv2.imshow("YOLO Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
output.release()
cv2.destroyAllWindows()