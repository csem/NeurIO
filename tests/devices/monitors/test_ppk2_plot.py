#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2025
Creation: 27.03.2025
Description: This class can be used with the PPK2 Kit to check that it generates correct plots.
"""
import time
from neurio.devices.monitors.power_monitor import PowerProfilerKitII
import numpy as np
import pytest
import os

class TestPPK2Plot:

    # Only execute this test if the PPK2 is connected to the computer
    @pytest.mark.skipif(not os.path.exists("/dev/tty.usbmodemF89AE991B16A2"), reason="PPK2 not connected")
    def test_ppk2(self):
        port = "/dev/tty.usbmodemF89AE991B16A2"
        ppk2 = PowerProfilerKitII(port=port, source_voltage=5000, baudrate=115200)
        ppk2.toggle_power("ON")
        time.sleep(5.0)
        ppk2.start("PHASE1")
        time.sleep(30)  # simulate algorithm
        ppk2.stop()

        import matplotlib.pyplot as plt

        recording = ppk2.recordings[0]
        times = [s['time'] for s in recording['samples']]
        values = [s['value'] for s in recording['samples']]

        plt.plot(times, values)
        plt.xlabel("Time (s)")
        plt.ylabel("Current (uA)")
        plt.title("PPK2 Measurement")
        plt.show()

        print("Average current: ", np.mean(values))
        time.sleep(2)

        # record again
        ppk2.start("Phase2")
        time.sleep(10)  # simulate algorithm
        ppk2.stop()

        recording = ppk2.recordings[1]
        times = [s['time'] for s in recording['samples']]
        values = [s['value'] for s in recording['samples']]

        plt.plot(times, values)
        plt.xlabel("Time (s)")
        plt.ylabel("Current (uA)")
        plt.title("PPK2 Measurement")
        plt.show()

        print("Average current: ", np.mean(values))
        time.sleep(3)
        ppk2.toggle_power("OFF")

        # pass the test
        assert True


