from fpioa_manager import fm
from maix import GPIO
import uos
from maix import KPU
import image
import time
# setup sensor
import lcd
import gc
import machine
import ujson

DEBUG = True
SD_FOLDER_DATA = "/sd/data"
SD_RESULT_FILE = "/sd/results.txt"
MODEL_PATH = "/sd/test.kmodel"
JSON_INFERENCE_PATH = "/sd/inference.json"

IMG_WIDTH = 224
IMG_HEIGHT = 224
KPU_OUTPUT = [0, 1, 1, 5]

lcd.init(type=1)
lcd.clear()
lcd.rotation(0)
# register leds
fm.register(12, fm.fpioa.GPIO0)
led_b = GPIO(GPIO.GPIO0, GPIO.OUT)
fm.register(13, fm.fpioa.GPIO1)
led_g = GPIO(GPIO.GPIO1, GPIO.OUT)
fm.register(14, fm.fpioa.GPIO2)
led_r = GPIO(GPIO.GPIO2, GPIO.OUT)
# turn off all leds
led_r.value(1)#off
led_g.value(1)#off
led_b.value(1)#off
LCD_X = 30
LCD_X_STATUS = 170

def init_sequence():
    for i in range(3):
        led_b.value(0)
        time.sleep_ms(100)
        led_r.value(1)  # off
        led_g.value(0)
        time.sleep_ms(100)
        led_g.value(1)  # off
        led_b.value(0)
        time.sleep_ms(100)
        led_b.value(1)  # off

def main():
    # starting signal
    init_sequence()

    r_dict = {"model": {}, "images": {}}
    kpu = KPU()
    # setup AI
    if DEBUG:
        print("Loading model...")
        lcd.draw_string(LCD_X, 60, "Loading model...", lcd.WHITE)

    print("KPU Loading...")
    try:
        start = time.ticks_ms()
        kpu.load(MODEL_PATH)
        end = time.ticks_ms()
        r_dict["model"] = {"load_ms": time.ticks_diff(end, start)}
    except Exception as e:
        print("KPU Loading failed:", e)
        lcd.draw_string(LCD_X_STATUS, 60, "ERROR", lcd.RED)
        raise e

    if DEBUG:
        lcd.draw_string(LCD_X_STATUS, 60, MODEL_PATH, lcd.GREEN)

    print("Model loaded.")

    # get list of images for inference
    if DEBUG:
        print("Listing json files...")
        lcd.draw_string(LCD_X, 80, "Listing files...", lcd.WHITE)
        time.sleep_ms(10)

    with open(JSON_INFERENCE_PATH, 'rb') as f:
         json_data = f.read()
         json_dict = ujson.loads(json_data)

    img_list = [json_dict[i] for i in json_dict]
    total_files = len(img_list)

    if DEBUG:
        print(img_list)
        print(str(total_files) + " files found.")
        lcd.draw_string(LCD_X_STATUS, 80, "found:"+str(total_files), lcd.GREEN)
        time.sleep_ms(10)

    # benchmarking phase
    if DEBUG:
        print("Benchmarking...")
        lcd.draw_string(LCD_X, 100, "Benchmarking...", lcd.WHITE)
        led_r.value(1)
        time.sleep_ms(10)

    for i in range(total_files):
        filename = SD_FOLDER_DATA + "/" + str(img_list[i])
        percent = int((i + 1) * 100 / total_files)

        r_img_dict = {}
        # compute loading time
        if DEBUG:
            print("Benchmarking...")
            lcd.draw_string(LCD_X, 100, "Benchmarking...", lcd.WHITE)
            led_r.value(1)
            draw_progress(LCD_X, 120, percent, "Load "+ str(img_list[i]))

        start = time.ticks_ms()
        img = image.Image(filename)
        end = time.ticks_ms()
        r_img_dict["load_ms"] = time.ticks_diff(end, start)

        # compute preprocessing time
        print("Preprocess")
        if DEBUG:
            draw_progress(LCD_X, 120, percent, "Preprocess")
        #img = img.resize(IMG_WIDTH, IMG_HEIGHT)
        # img = img.to_grayscale()
        start = time.ticks_ms()
        img.pix_to_ai()
        end = time.ticks_ms()
        r_img_dict["preprocess_ms"] = time.ticks_diff(end, start)

        # perform inference and store result
        print("Infer")
        if DEBUG:
            draw_progress(LCD_X, 120, percent, "Infer")

        start = time.ticks_ms()
        kpu.run(img)
        end = time.ticks_ms()
        r_img_dict["inference_ms"] = time.ticks_diff(end, start)

        feature_map = kpu.get_outputs()

        print("OUT")
        if DEBUG:
            draw_progress(LCD_X, 120, percent, "Readout")

        r_img_dict["output"] = feature_map[:]
        r_img_dict["filename"] = filename

        with open(SD_RESULT_FILE, "a") as f:
            f.write(str(r_img_dict))

        if DEBUG:
            led_r.value(0) #on
            draw_progress(LCD_X, 120, percent)
            led_r.value(1) #off

        del r_img_dict
        del feature_map
        gc.collect()

    if DEBUG:
        draw_progress(LCD_X, 120, 100)
        print("Done")
        lcd.draw_string(LCD_X_STATUS, 100, "DONE", lcd.GREEN)
        time.sleep_ms(10)
        led_r.value(1)
        led_g.value(0)

    kpu.deinit()
    time.sleep(2)
    del kpu
    gc.collect()
    del img_list
    del r_dict
    gc.collect()
    time.sleep_ms(200)
    led_g.value(1) # conclude
    gc.collect()

    # restart to clear everything
    machine.reset()

def draw_progress(x: int, y: int, percent: int, message: str = ""):
    lcd.fill_rectangle(x, y-5, 250, 10, (0,0,0))
    lcd.fill_rectangle(x, y, percent, 5, lcd.GREEN)
    lcd.draw_string(x + 110, y - 5, str(percent) + "%", lcd.GREEN)
    lcd.draw_string(x + 150, y - 5, message, lcd.GREEN)
    time.sleep_ms(3)