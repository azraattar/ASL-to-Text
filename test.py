import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL to Text")
        self.offset = 20
        self.imgsize = 300
        self.cap = cv2.VideoCapture(0)
        self.classifier = Classifier("model/keras_model.h5", "model/labels.txt")
        self.detector = HandDetector(maxHands=1)
        self.folder = "Data/"  # You might want to make this dynamic
        self.counter = 0
        self.labels = ["hey", "i", "a", "m", "z", "r"] # Matches your provided labels

        # UI Elements
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.prediction_label = tk.Label(root, text="Prediction: ", font=("Arial", 12))
        self.prediction_label.pack()

        self.current_word_label = tk.Label(root, text="Current Word: ", font=("Arial", 10))
        self.current_word_label.pack()
        self.current_word = ""

        self.sentence_label = tk.Label(root, text="Sentence: ", font=("Arial", 12))
        self.sentence_label.pack()
        self.current_sentence = ""

        # Frame to hold the word building buttons
        self.word_button_frame = tk.Frame(root)
        self.word_button_frame.pack(pady=5)

        self.build_word_button = tk.Button(self.word_button_frame, text="Add to Word", font=("Arial", 10), command=self.add_to_word)
        self.build_word_button.pack(side='left', padx=5)

        self.add_word_button = tk.Button(self.word_button_frame, text="Add Word to Sentence (Space)", font=("Arial", 10), command=self.add_word_to_sentence)
        self.add_word_button.pack(side='left', padx=5)

        self.clear_button = tk.Button(root, text="Clear Sentence", font=("Arial", 10), command=self.clear_sentence)
        self.clear_button.pack(pady=5)

        self.index = 0 # Initialize index

        self.update_frame()

    def add_to_word(self):
        if 0 <= self.index < len(self.labels):
            self.current_word += self.labels[self.index]
            self.current_word_label.config(text=f"Current Word: {self.current_word}")

    def add_word_to_sentence(self):
        if self.current_word:
            self.current_sentence += self.current_word + " "
            self.sentence_label.config(text=f"Sentence: {self.current_sentence}")
            self.current_word = ""
            self.current_word_label.config(text=f"Current Word: ")

    def clear_sentence(self):
        self.current_sentence = ""
        self.sentence_label.config(text=f"Sentence: {self.current_sentence}")
        self.current_word = ""
        self.current_word_label.config(text=f"Current Word: ")

    def update_frame(self):
        success, img = self.cap.read()
        if success:
            imgOutput = img.copy()
            hands, img = self.detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgwhite = np.ones((self.imgsize, self.imgsize, 3), np.uint8) * 255
                imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    aspectratio = h / w
                    if aspectratio > 1:
                        k = self.imgsize / h
                        wcalculated = math.ceil(k * w)
                        imgresize = cv2.resize(imgCrop, (wcalculated, self.imgsize))
                        wgap = math.ceil((self.imgsize - wcalculated) / 2)
                        imgwhite[:, wgap:wcalculated + wgap] = imgresize
                        prediction, self.index = self.classifier.getPrediction(imgwhite)
                        print(prediction, self.index)
                    else:
                        k = self.imgsize / w
                        hcalculated = math.ceil(k * h)
                        imgresize = cv2.resize(imgCrop, (self.imgsize, hcalculated))
                        hgap = math.ceil((self.imgsize - hcalculated) / 2)
                        imgwhite[hgap:hcalculated + hgap, :] = imgresize
                        prediction, self.index = self.classifier.getPrediction(imgwhite)

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("Image White", imgwhite)
                    if 0 <= self.index < len(self.labels):
                        cv2.putText(imgOutput, self.labels[self.index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
                        self.prediction_label.config(text=f"Prediction: {self.labels[self.index]}")
                    else:
                        self.prediction_label.config(text="Prediction: Unknown")
                else:
                    self.prediction_label.config(text="Prediction: No hand detected")

            cv2.imshow("Image", imgOutput)
            img_tk = self.convert_cv2_to_tkinter(imgOutput)
            self.image_label.imgtk = img_tk
            self.image_label.config(image=img_tk)

        self.root.after(30, self.update_frame)

    def convert_cv2_to_tkinter(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        return img_tk

    def on_closing(self):
        print("Closing Application...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLTranslatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()