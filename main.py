import cv2
import zipfile
import os

import numpy as np
import tkinter as tk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from tkinter import filedialog, Canvas, messagebox, ttk
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
from sklearn.mixture import BayesianGaussianMixture

class Imageenhancer:
    def __init__(self, master):
        self.master = master
        master.title("Image Enhancement for Histopathology Images")

        self.algorithm = tk.StringVar()
        self.image_label = tk.Label(root)

        self.label = Label(master, text="Select images to enhance")
        self.label.pack()

        self.canvas_orig = Canvas(master, width=512, height=512)
        self.canvas_orig.pack(side="left")

        self.canvas_proc = Canvas(master, width=512, height=512)
        self.canvas_proc.pack(side="right")

        self.select_button = Button(master, text="Select Images", command=self.select_images)
        self.select_button.pack()

        self.dropdown = ttk.Combobox(root, textvariable=self.algorithm)
        self.dropdown['values'] = ("WCLAHE", "Reinhard's Method", "CDV Multi Modal")
        self.dropdown.pack()
        self.dropdown.current(0)

        self.enhance_button = Button(master, text="enhance Images", command=self.enhance_images, state="disabled")
        self.enhance_button.pack()

        self.save_button = Button(master, text="Save enhanced Images", command=self.save_images, state="disabled")
        self.save_button.pack()

        self.predict_button = Button(master, text="predict image", command=self.predict_images, state="disabled")
        self.predict_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    
    def sigmoid(self, x, alpha, beta, M, C1, C2, C3):
        return C1 + (-alpha / (1 + np.exp(-beta * ((x - M)**C3 / C2))))

    def ssr(self, image, sigma):
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        return (np.log10(image + 1e-8) - np.log10(blurred + 1e-8)) / sigma**2

    def wclahe(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.float64(hsv)

        sigmas = [15, 180, 250]
        params = [
            (1, 0.8, 80, 1, 10, 1),  # Parameters for w1
            (2, 1, 100, 2, 350, 2),  # Parameters for w2
            (-1, 0.8, 150, 0, 10, 1) # Parameters for w3
        ]

        h, s, v = cv2.split(hsv)

        weights = []
        retinex = np.zeros_like(v)
        for sigma, param in zip(sigmas, params):
            retinex = self.ssr(v, sigma)
            weight = self.sigmoid(retinex, *param)
            weights.append(weight)
        weights = np.array(weights)
        weights = weights / np.sum(weights, axis=0)
        msr = np.sum(weights * np.array([self.ssr(v, sigma) for sigma in sigmas]), axis=0)
        msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX)

        hsv[:,:,2] = msr
        hsv = np.uint8(hsv)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l_channel, a_channel, b_channel = cv2.split(image_lab)


        clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8,8))
        l_channel_clahe1 = clahe.apply(l_channel)

        clahe = cv2.createCLAHE(clipLimit=0.001, tileGridSize=(8,8))
        l_channel_clahe2 = clahe.apply(l_channel)

        # Image fusion
        # Step 1:
        X = np.array([l_channel_clahe1.flatten(), l_channel_clahe2.flatten()])

        # Step 2:
        C = np.cov(X)

        # Step 3:
        eigenvalues, eigenvectors = np.linalg.eig(C)
        idx = eigenvalues.argmax()

        # Step 4:
        weights = eigenvectors[:, idx] / np.sum(eigenvectors[:, idx])

        # Step 5:
        F = weights[0]*l_channel_clahe1 + weights[1]*l_channel_clahe2
        F = cv2.normalize(F, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        image_lab_enhanced = cv2.merge((F, a_channel, b_channel))
        image_enhanced = cv2.cvtColor(image_lab_enhanced, cv2.COLOR_LAB2BGR)

        return image_enhanced

    def reinhard(self, source_img, ref_img_path="sample.jpg"):
        ref_img = cv2.imread(ref_img_path)

        source_img_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)
        ref_img_lab = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB)

        Ls, As, Bs = cv2.split(source_img_lab)
        Lr, Ar, Br = cv2.split(ref_img_lab)

        q = (np.std(Lr) - np.std(Ls)) / np.std(Lr)

        if q > 0:
            Ln = np.mean(Ls) + (Ls - np.mean(Ls)) * (1 + q)
            An = As
            Bn = Bs
        else:
            Ln = np.mean(Ls) + (Ls - np.mean(Ls)) * (1 + 0.05)
            An = np.mean(Ar) + (As - np.mean(As))
            Bn = np.mean(Br) + (Bs - np.mean(Bs))

        Ln = Ln.astype(np.uint8)
        An = An.astype(np.uint8)
        Bn = Bn.astype(np.uint8)

        img_normalized_lab = cv2.merge([Ln, An, Bn])
        img_normalized_rgb = cv2.cvtColor(img_normalized_lab, cv2.COLOR_LAB2RGB)

        return img_normalized_rgb

    def cdv_mm(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        mask = np.all((img > 25) & (img < 300), axis=2)
        img_1d = img[mask].reshape(-1, 1)

        gmm = BayesianGaussianMixture(n_components=2).fit(img_1d)

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()

        fg_index = np.argmax(means)
        bg_index = 1 - fg_index

        norm_img = (img - means[bg_index]) / (stds[fg_index] - stds[bg_index])
        norm_img = norm_img * weights[fg_index] + img

        norm_img[norm_img > 300] = img[norm_img > 300]

        return norm_img

    def select_images(self):
        self.image_paths = filedialog.askopenfilenames()
        try:
            self.img = Image.open(self.image_paths[0]).resize((512, 512))
        except:
            messagebox.showinfo("Image enhancement", "Ypur file(s) is damaged or not supported!")

        self.tk_img = ImageTk.PhotoImage(self.img)
        self.canvas_orig.create_image(20, 20, anchor="nw", image=self.tk_img)
        self.enhance_button.config(state="normal")

        # Display the filenames of the selected images
        filenames = "\n".join(os.path.basename(path) for path in self.image_paths)
        self.label.config(text=f"Selected images:\n{filenames}")

    def enhance_images(self):
        self.enhanced_image_paths = []
        for image_path in self.image_paths:
            img = cv2.imread(image_path)
            if self.algorithm.get() == "WCLAHE":
                img = self.wclahe(img)
            elif self.algorithm.get() == "Reinhard's Method":
                img = self.reinhard(img)
            elif self.algorithm.get() == "CDV Multi Modal":
                img = self.cdv_mm(img)

            # Save the enhanced image in the TEMP folder
            if not os.path.exists('TEMP'):
                os.makedirs('TEMP')
            enhanced_image_path = os.path.join('TEMP', os.path.basename(image_path))
            cv2.imwrite(enhanced_image_path, img)
            self.enhanced_image_paths.append(enhanced_image_path)

            # Display the enhanced image
            self.proc_img = Image.open(enhanced_image_path).resize((512, 512))
            self.tk_proc_img = ImageTk.PhotoImage(self.proc_img)
            self.canvas_proc.create_image(20, 20, anchor="nw", image=self.tk_proc_img)

        self.save_button.config(state="normal")
        self.predict_button.config(state="normal")
        messagebox.showinfo("Image enhancement", "Image enhancement is done!")
    
    def predict_images(self):
        class_indices = {'Adenocarcinoma': 0, 'Squamous Cell Carcinoma': 1, 'Benign': 2}
        model = load_model("model.h5")

        img = image.load_img(self.enhanced_image_paths[0], target_size=(400, 400))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize to [0,1] if the model was trained on normalized images

        predictions = model.predict(img_array)

        predicted_class = np.argmax(predictions, axis=1)
        class_labels = {v: k for k, v in class_indices.items()}

        pred = class_labels[predicted_class[0]]
        messagebox.showinfo("Prediction Result", "Lung Cancer Prediction:" + " " + pred)

    def save_images(self):
        # Zip the enhanced images
        zip_file = 'enhanced_images.zip'
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            for enhanced_image_path in self.enhanced_image_paths:
                # Add the file to the zip without the "TEMP" folder in the path
                zipf.write(enhanced_image_path, arcname=os.path.basename(enhanced_image_path))

        # Ask the user where to save the zipped images
        save_path = filedialog.asksaveasfilename(defaultextension=".zip")
        if save_path:
            os.rename(zip_file, save_path)

        print("Enhanced images zipped successfully!")

root = Tk()
my_gui = Imageenhancer(root)
root.mainloop()
