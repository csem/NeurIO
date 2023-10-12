from fpioa_manager import fm
from Maix import GPIO
import time
import uos
import KPU as kpu
import lcd
import image
from board import board_info


def main():
    lcd.init(type=1)
    lcd.clear()
    lcd.rotation(0)
    lcd.draw_string(20, 100, "Inference scripts ready.", lcd.WHITE, lcd.BLACK)
    lcd.draw_string(20, 115, "Please put data on SD card.", lcd.WHITE, lcd.BLACK)
    lcd.draw_string(20, 130, "Once done, press 'Y' on computer.", lcd.WHITE, lcd.BLACK)

    fm.register(12, fm.fpioa.GPIO0)
    led_r = GPIO(GPIO.GPIO0, GPIO.OUT)