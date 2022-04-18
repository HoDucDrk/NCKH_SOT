from tkinter import Tk
import Packages as pk

root = Tk()
win = pk.Window(root)
win()
root.iconbitmap('./Assets/icon.ico')
root.mainloop()