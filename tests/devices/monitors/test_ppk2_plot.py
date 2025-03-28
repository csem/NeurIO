#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2025
Creation: 27.03.2025
Description: TODO
"""


import pytest
import time
from ppk2_api.ppk2_api import PPK2_MP
from neurio.devices.monitors.power_monitor_mp import PowerProfilerKitII

import unittest
from unittest.mock import patch, MagicMock
import threading
import numpy as np

class TestPowerProfilerKitII(unittest.TestCase):

    def test_recording(self):
        port = "/dev/tty.usbmodemF89AE991B16A2"
        ppk2_test = PPK2_MP(port, buffer_max_size_seconds=2)
        modifiers = ppk2_test.get_modifiers()
        print(f"Modifiers: {modifiers}")
        ppk2_test.use_source_meter()  # set source meter mode
        ppk2_test.set_source_voltage(5000)  # set source voltage in mV
        ppk2_test.start_measuring()  # start measuring
        print("Measuring started")

        # read measured values in a for loop like this:
        samples = [-1]
        for i in range(0, 10):
            read_data = ppk2_test.get_data()
            if read_data != b'':
                samples = ppk2_test.get_samples(read_data)
                print(f"Average of {len(samples)} samples is: {np.sum(samples) / len(samples)}uA")
            print(i)
            print(time.time().__str__(), np.sum(samples)/len(samples))
            time.sleep(0.1)

        ppk2_test.stop_measuring()


    def test_normal(self):
        port = "/dev/tty.usbmodemF89AE991B16A2"
        ppk2_test = PowerProfilerKitII(port, source_voltage=5000)
        ppk2_test.start('test_phase')
        time.sleep(5)
        ppk2_test.stop()

        print(ppk2_test.recordings)
