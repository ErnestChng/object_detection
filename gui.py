import tkinter as tk
from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk

from object_detector import ObjectDetector


class GUI:
    def __init__(self):
        self.top = tk.Tk()
        self.top.geometry('800x600')
        self.top.title('Object Detection')
        self.top.configure(background='#CDCDCD')
        self.label = Label(self.top, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.sign_image = Label(self.top)

        upload = Button(self.top, text="Upload an image", command=self.upload_image, padx=10, pady=5)
        upload.configure(background='#364156', foreground='black', font=('arial', 10, 'bold'))
        upload.pack(side=BOTTOM, pady=50)
        self.sign_image.pack(side=BOTTOM, expand=True)
        self.label.pack(side=BOTTOM, expand=True)
        heading = Label(self.top, text="Object Detection", pady=20, font=('arial', 20, 'bold'))
        heading.configure(background='#CDCDCD', foreground='black')
        heading.pack()

        self.model = ObjectDetector()

    def classify(self, file_path):
        self.model.detect_objects(file_path, url_type='local')

    def show_classify_button(self, file_path):
        classify_b = Button(self.top, text="Classify Image", command=lambda: self.classify(file_path), padx=10, pady=5)
        classify_b.configure(background='#364156', foreground='black', font=('arial', 10, 'bold'))
        classify_b.place(relx=0.79, rely=0.46)

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((self.top.winfo_width() / 2.25), (self.top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)
            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.label.configure(text='')
            self.show_classify_button(file_path)
        except Exception as e:
            print(e)
            pass

    def run(self):
        self.top.mainloop()
