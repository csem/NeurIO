from fpioa_manager import fm
from Maix import GPIO
import time
import uos
import KPU as kpu
import lcd
import image
import json


def draw_progress(x: int, y: int, percent: int):
    lcd.fill_rectangle(x, y, percent, 5, lcd.GREEN)
    lcd.draw_string(x + 110, y - 2, str(percent) + "%", lcd.GREEN)


def main():
    lcd.init(type=1)
    lcd.clear()
    lcd.rotation(2)
    lcd.draw_string(60, 100, "Test Benchmarking Skeleton...", lcd.WHITE, lcd.BLACK)
    for i in range(101):
        draw_progress(60, 115, i)
        time.sleep(0.1)

    with open("/sd/results.txt", "w") as f:
        f.write(str({"comment": "ok"}))
