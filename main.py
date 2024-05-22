from modules.gui import Main
import os
import tkinter as tk


def main():
    root = tk.Tk()

    # Import the tcl file with the tk.call method
    root.tk.call('source', os.getcwd() + '/Styles/ttk-Breeze/breeze.tcl')
    Main(root)

    root.mainloop()


if __name__ == "__main__":
    main()
