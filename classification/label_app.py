import tkinter as tk
import sys
import os
from PIL import Image, ImageTk
from tkinter import ttk


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, width=1000, height=800)

        self.lookup = {"a": "alpha",
                       "b": "beta"}

        self.current_label = tk.StringVar()
        self.current_label.set("nolabel")
        self.label_label = ttk.Label(
            master=master, textvariable=self.current_label)

        self.master = master
        self.img_dir = sys.argv[1]
        print(self.img_dir)
        self.index = 0
        self.files = self.get_file_list(self.img_dir)

        self.label_label.pack(side="bottom", pady=10, padx=10)
        self.pack(side="top", fill="both", anchor="center",
                  expand=True, pady=10, padx=10)

        self.show_images(self.files[self.index])
        self.focus_get()

        master.bind("<KeyPress>", lambda event: self.set_label(event))
        master.bind("<space>", lambda event: self.store_label(event))

    def get_file_list(self, img_dir):
        for subdir, dirs, files in os.walk(self.img_dir):
            files = sorted([f for f in files if not f[0]
                            == '.' and f[-5:] == '0.jpg'])
            return files

    def set_label(self, event):
        if event.char in self.lookup:
            self.current_label.set(self.lookup[event.char])

    def store_label(self, event):
        print(event)
        file = self.files[self.index]
        print("I'm storing label "+str(self.current_label)+" for img "+file)

        second, third = self.expand_file_name(file)

        for f in [file, second, third]:
            name, extension = f.split(".")
            os.rename(os.path.join(self.img_dir, f), os.path.join(
                self.img_dir, name+"_"+str(self.current_label)+"."+extension))

        self.index += 1

        self.show_images(self.files[self.index])

    def expand_file_name(self, img_path):

        def patch(string, char):
            new = list(string)
            new[-5] = char
            new = "".join(new)
            return new

        second = patch(img_path, "1")
        third = patch(img_path, "2")

        return second, third

    def show_images(self, img_path):

        second, third = self.expand_file_name(img_path)

        img_paths = [img_path, second, third]
        size = (250/2, 1200/2)
        for idx, path in enumerate(img_paths):
            im = Image.open(os.path.join(self.img_dir, path))
            im.thumbnail(size, Image.ANTIALIAS)
            render = ImageTk.PhotoImage(im)

            img = tk.Label(self, image=render)
            img.image = render
            print(img.winfo_width())
            img.place(x=idx*250/2+200, y=0)


if __name__ == "__main__":
    
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
