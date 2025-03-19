import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(12,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(12)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:-1, 1:13]  # 2014-2023 (10 лет)
    y = data[1:, 1:13]   # 2015-2024 (10 лет)
    return X, y

def train_model(model, X_train, y_train, epochs=200):
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
    return history

class WeatherApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Прогноз температуры")
        self.model = None
        self.X = None
        self.y = None
        self.setup_ui()

    def setup_ui(self):
        tk.Button(self.root, text="Загрузить данные", command=self.load_data).pack()
        tk.Button(self.root, text="Обучить модель", command=self.train).pack()
        tk.Button(self.root, text="Прогноз на 2025", command=self.predict).pack()
        tk.Button(self.root, text="Построить график", command=self.plot).pack()

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.X, self.y = load_data(file_path)
                if self.X.shape[0] != self.y.shape[0]:
                    raise ValueError("Несоответствие данных в файле!")
                messagebox.showinfo("Успех", "Данные загружены!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")
        else:
            messagebox.showwarning("Предупреждение", "Файл не выбран!")

    def train(self):
        if self.X is None or self.y is None:
            messagebox.showerror("Ошибка", "Сначала загрузите данные!")
            return
        self.model = create_model()
        self.history = train_model(self.model, self.X, self.y)
        self.model.save("models/trained_model.h5")
        messagebox.showinfo("Успех", "Модель обучена!")

    def predict(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Сначала обучите модель!")
            return
        prediction = self.model.predict(self.X[-1].reshape(1, -1))
        np.savetxt("prediction_2025.txt", prediction)
        messagebox.showinfo("Успех", "Прогноз сохранен в prediction_2025.txt")

    def plot(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Сначала обучите модель!")
            return
        prediction = self.model.predict(self.X[-1].reshape(1, -1))
        plt.figure()
        plt.plot(range(1, 13), prediction[0], marker='o')
        plt.title("Прогноз температуры на 2025 год")
        plt.xlabel("Месяц")
        plt.ylabel("Температура (°C)")
        plt.grid(True)
        plt.savefig("temp_plot.png")
        plt.close()
        img = ImageTk.PhotoImage(Image.open("temp_plot.png"))
        top = tk.Toplevel(self.root)
        tk.Label(top, image=img).pack()
        top.mainloop()

if __name__ == "__main__":
    app = WeatherApp()
    app.root.mainloop()