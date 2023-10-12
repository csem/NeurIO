from fpioa_manager import fm
from Maix import GPIO
import uos
import KPU as kpu
import image
import time
# setup sensor
import lcd
import json

DEBUG = True
IMG_WIDTH = 224
IMG_HEIGHT = 224
KPU_OUTPUT = [0, 1, 1, 5]
SD_FOLDER_DATA = "/sd/data"
SD_RESULT_FILE = "/sd/results.txt"
MODEL_PATH = "/sd/test.kmodel"

lcd.init(type=1)
lcd.clear()
lcd.rotation(0)
fm.register(14, fm.fpioa.GPIO0)
led_r = GPIO(GPIO.GPIO0, GPIO.OUT)
led_r.value(1)


def main():
    r_dict = {"model": {}, "images": {}}

    # setup AI
    if DEBUG:
        print("Loading model...")
        lcd.draw_string(60, 60, "Loading model...", lcd.WHITE)

    start = time.ticks_ms()
    m = kpu.load(MODEL_PATH)
    kpu.set_outputs(m, KPU_OUTPUT[0], KPU_OUTPUT[1], KPU_OUTPUT[2], KPU_OUTPUT[3])
    end = time.ticks_ms()
    r_dict["model"] = {"load_ms": time.ticks_diff(end, start)}
    if DEBUG:
        print("OK")
        lcd.draw_string(200, 60, "OK", lcd.GREEN)

    # get list of images for inference
    if DEBUG:
        print("Listing files...")
        lcd.draw_string(60, 80, "Listing files...", lcd.WHITE)
    img_list = sorted([x for x in uos.listdir(SD_FOLDER_DATA)])
    total_files = len(img_list)
    if DEBUG:
        print("ok")
        lcd.draw_string(200, 80, "OK", lcd.GREEN)

    # benchmarking phase
    if DEBUG:
        print("Benchmarking...")
        lcd.draw_string(60, 100, "Benchmarking...", lcd.WHITE)
        led_r.value(0)

    for i in range(total_files):
        filename = SD_FOLDER_DATA + "/" + str(img_list[i])

        r_img_dict = {}
        # compute loading time
        start = time.ticks_ms()
        img = image.Image(filename)
        end = time.ticks_ms()
        r_img_dict["load_ms"] = time.ticks_diff(end, start)

        # compute preprocessing time
        start = time.ticks_ms()
        img = img.resize(IMG_WIDTH, IMG_HEIGHT)
        #img = img.to_grayscale()
        img.pix_to_ai()
        end = time.ticks_ms()
        r_img_dict["preprocess_ms"] = time.ticks_diff(end, start)

        # perform inference and store result
        start = time.ticks_ms()
        feature_map = kpu.forward(m, img)
        end = time.ticks_ms()
        r_img_dict["inference_ms"] = time.ticks_diff(end, start)
        r_img_dict["output"] = feature_map[:]
        r_img_dict["filename"] = filename

        with open(SD_RESULT_FILE, "a") as f:
            f.write(str(r_img_dict))

        if DEBUG:
            led_r.value(1)
            percent = int((i + 1) * 100 / total_files)
            print(percent)
            draw_progress(60, 120, percent)
            time.sleep_ms(10)
            led_r.value(0)

    if DEBUG:
        print("Done")
        lcd.draw_string(200, 100, "DONE", lcd.GREEN)
        led_r.value(1)


def draw_progress(x, y, percent):
    lcd.fill_rectangle(x, y, percent, 5, lcd.GREEN)
    lcd.draw_string(x + 110, y - 5, str(percent) + "%", lcd.GREEN)