import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# === UTILS ===

def create_directories():
    os.makedirs("dataset/chats", exist_ok=True)
    os.makedirs("dataset/chiens", exist_ok=True)

def copy_images_to_folder(file_paths, folder):
    for path in file_paths:
        shutil.copy(path, folder)

def get_histogram(path):
    image = Image.open(path).resize((100, 100)).convert("RGB")
    return np.array(image.histogram())

def compare_histogram(img1_path, img2_path):
    hist1 = get_histogram(img1_path)
    hist2 = get_histogram(img2_path)
    return np.linalg.norm(hist1 - hist2)

def predict_image(img_path, chat_images, chien_images):
    chat_score = np.mean([compare_histogram(img_path, img) for img in chat_images])
    chien_score = np.mean([compare_histogram(img_path, img) for img in chien_images])
    total = chat_score + chien_score
    chat_percent = 100 * (1 - (chat_score / total))
    chien_percent = 100 - chat_percent
    return chat_percent, chien_percent

# === INTERFACE ===

class ChatChienApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 AI - Prédiction Chat vs Chien")
        self.root.geometry("740x520")
        self.root.configure(bg="#f5f5f5")

        self.icon_chat = ImageTk.PhotoImage(Image.open("icons/icon_chat.png").resize((32, 32)))
        self.icon_chien = ImageTk.PhotoImage(Image.open("icons/icon_chien.png").resize((32, 32)))
        self.icon_next = ImageTk.PhotoImage(Image.open("icons/icon_next.png").resize((32, 32)))
        self.icon_previous = ImageTk.PhotoImage(Image.open("icons/icon_previous.png").resize((32, 32)))
        self.icon_prediction = ImageTk.PhotoImage(Image.open("icons/icon_prediction.png").resize((32, 32)))

        self.chat_imgs_preview = []
        self.chien_imgs_preview = []

        self.step1_window()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def step1_window(self):
        self.clear_window()

        title = tk.Label(self.root, text="Étape 1 : Remplir le DataSet ", font=("Arial", 20, "bold"), bg="#f5f5f5", fg="#FF007F")
        title.pack(pady=10)

        # SECTION CHATS
        chat_frame = tk.Frame(self.root, bg="#d0f0ff", bd=2, relief="groove", padx=10, pady=10)
        chat_frame.pack(pady=10)

        tk.Label(chat_frame, text="Chats", font=("Arial", 14), image=self.icon_chat, compound="left", bg="#d0f0ff").pack()
        tk.Button(chat_frame, text="Ajouter 10 images de chats", command=self.add_images_chat, bg="#b3e0ff").pack(pady=5)
        self.chat_preview_frame = tk.Frame(chat_frame, bg="#d0f0ff")
        self.chat_preview_frame.pack()

        # SECTION CHIENS
        chien_frame = tk.Frame(self.root, bg="#ffe5d0", bd=2, relief="groove", padx=10, pady=10)
        chien_frame.pack(pady=10)

        tk.Label(chien_frame, text="Chiens", font=("Arial", 14), image=self.icon_chien, compound="left", bg="#ffe5d0").pack()
        tk.Button(chien_frame, text="Ajouter 10 images de chiens", command=self.add_images_chien, bg="#ffd1a3").pack(pady=5)
        self.chien_preview_frame = tk.Frame(chien_frame, bg="#ffe5d0")
        self.chien_preview_frame.pack()

        self.btn_next = tk.Button(self.root, text="  Suivant   ", image=self.icon_next, compound="right",
                                  bg="#87cefa", fg="white", font=("Arial", 12 , "bold"), command=self.step2_window, state=tk.DISABLED)
        self.btn_next.pack(pady=30)

    def display_previews(self, paths, target_frame, preview_list, bg_color):
        for widget in target_frame.winfo_children():
            widget.destroy()
        preview_list.clear()
        for path in paths:
            img = Image.open(path).resize((60, 60))
            img_tk = ImageTk.PhotoImage(img)
            lbl = tk.Label(target_frame, image=img_tk, bg=bg_color)
            lbl.image = img_tk
            lbl.pack(side="left", padx=2)
            preview_list.append(lbl)

    def add_images_chat(self):
        files = filedialog.askopenfilenames(title="Choisir 10 images de chats")
        if len(files) != 10:
            messagebox.showerror("Erreur ❌", "Tu dois choisir exactement 10 images.")
            return
        copy_images_to_folder(files, "dataset/chats")
        self.display_previews(files, self.chat_preview_frame, self.chat_imgs_preview, "#d0f0ff")
        messagebox.showinfo("Succès ✅ ", "Images de chats ajoutées !")
        self.check_next_button()

    def add_images_chien(self):
        files = filedialog.askopenfilenames(title="Choisir 10 images de chiens")
        if len(files) != 10:
            messagebox.showerror("Erreur ❌", "Tu dois choisir exactement 10 images.")
            return
        copy_images_to_folder(files, "dataset/chiens")
        self.display_previews(files, self.chien_preview_frame, self.chien_imgs_preview, "#ffe5d0")
        messagebox.showinfo("Succès ✅", "Images de chiens ajoutées !")
        self.check_next_button()

    def check_next_button(self):
        if len(os.listdir("dataset/chats")) >= 10 and len(os.listdir("dataset/chiens")) >= 10:
            self.btn_next.config(state=tk.NORMAL)

    def step2_window(self):
        self.clear_window()
        self.img_path = None

        tk.Label(self.root, text="Étape 2 : Prédire une image", font=("Arial", 20, "bold"), bg="#f5f5f5", fg="#41c124").pack(pady=10)

        self.image_label = tk.Label(self.root, bg="#f5f5f5")
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 16), bg="#f5f5f5")
        self.result_label.pack(pady=10)

        tk.Button(self.root, text="Choisir une image", command=self.choose_image, bg="#cde", font=("Arial", 12)).pack(pady=5)
        button_frame = tk.Frame(self.root, bg="#F0F8FF")
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="  Précédent ", image=self.icon_previous, compound="left",
                                  bg="#87cefa", fg="white", font=("Arial", 12 , "bold"), command=self.step1_window,).pack(side=tk.LEFT, padx=30)
        tk.Button(button_frame, text="  Prédire  ", image=self.icon_prediction, compound="left", 
                                 bg="#FFA07A", fg="white", font=("Arial", 12, "bold"), command=self.do_prediction,).pack(side=tk.LEFT, padx=30)

    def choose_image(self):
        file_path = filedialog.askopenfilename(title="Choisir une image à prédire")
        if not file_path:
            return
        self.img_path = file_path
        img = Image.open(file_path).resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def do_prediction(self):
        if not self.img_path:
            messagebox.showwarning("Attention !", "Aucune image sélectionnée.")
            return

        chat_imgs = [f"dataset/chats/{f}" for f in os.listdir("dataset/chats") if f.endswith((".jpg", ".jpeg", ".png"))]
        chien_imgs = [f"dataset/chiens/{f}" for f in os.listdir("dataset/chiens") if f.endswith((".jpg", ".jpeg", ".png"))]

        if not chat_imgs or not chien_imgs:
            messagebox.showerror("Erreur", "Le dataset est vide.")
            return

        chat_p, chien_p = predict_image(self.img_path, chat_imgs, chien_imgs)
        self.result_label.config(text=f"🔹 Chat : {chat_p:.2f}%\n🔸 Chien : {chien_p:.2f}%")


# === MAIN ===
if __name__ == "__main__":
    create_directories()
    root = tk.Tk()
    app = ChatChienApp(root)
    root.mainloop()
