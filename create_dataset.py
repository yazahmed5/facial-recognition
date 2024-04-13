import cv2
import os
from tkinter import messagebox, filedialog

def start_capture(name, mode='camimg'):
    path = "./data/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')

    if mode == 'camimg':
        vid = cv2.VideoCapture(0)
        while True:
            ret, img = vid.read()
            new_img = None
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                new_img = img[y:y+h, x:x+w]
            cv2.imshow("FaceDetection", img)
            keyCode = cv2.waitKey(1)
            key = cv2.waitKey(1) & 0xFF
       
            try:
                cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
                num_of_images += 1
            except:
                pass
            if cv2.getWindowProperty("FaceDetection", cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord("q") or key == 27 or num_of_images > 301:
                break
        cv2.destroyAllWindows()				
        return num_of_images

    elif mode == 'uplimg':
        messagebox.showinfo("INSTRUCTIONS", "Select 30 or more images of your face for dataset \n(More image better accuracy)")
        # Allow user to select multiple image files
        file_paths = filedialog.askopenfilenames(title="Upload")
        if len(file_paths) < 30:
            messagebox.showerror("ERROR", "Please select 30 or more images!")
            return

        num_of_images = 0  # Initialize the counter for the number of images processed

        for file_path in file_paths:
            img = cv2.imread(file_path)
            new_img = img.copy()  # Initialize new_img to a copy of the original image
            if img is None:
                print("Error: Unable to read image file:", file_path)
                continue  # Skip to the next image if there's an error reading the file

            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                cv2.putText(img, str(str(num_of_images) + " images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                new_img = img[y:y+h, x:x+w]

            # Save the processed image
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
            num_of_images += 1  # Increment the counter for the number of images processed

        cv2.destroyAllWindows()
        return num_of_images
