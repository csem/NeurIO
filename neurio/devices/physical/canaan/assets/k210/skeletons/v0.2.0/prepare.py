from fpioa_manager import fm
from Maix import GPIO
import time
import uos
import KPU as kpu
import lcd
import image
from board import board_info

fm.register(12, fm.fpioa.GPIO0)
led_b = GPIO(GPIO.GPIO0, GPIO.OUT)
fm.register(13, fm.fpioa.GPIO1)
led_g = GPIO(GPIO.GPIO1, GPIO.OUT)
fm.register(14, fm.fpioa.GPIO2)
led_r = GPIO(GPIO.GPIO2, GPIO.OUT)


def show_color(r: int, g: int, b: int):
    led_r.value(1 - r)
    led_g.value(1 - g)
    led_b.value(1 - b)


def main():
    show_color(1, 0, 0)
    lcd.init(type=1)
    lcd.clear()
    lcd.rotation(0)
    lcd.draw_string(20, 100, "Inference scripts ready.", lcd.WHITE, lcd.BLACK)
    lcd.draw_string(20, 115, "Please put data on SD card.", lcd.WHITE, lcd.BLACK)
    lcd.draw_string(20, 130, "Once done, press 'Y' on computer.", lcd.WHITE, lcd.BLACK)

    show_color(0, 1, 0)
