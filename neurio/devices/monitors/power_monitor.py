#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 15.01.2024
Description: Power Monitoring classes
"""

import time
import threading
import json
import os
import numpy as np

import time
import serial
import struct
import logging
from serial.threaded import ReaderThread, Protocol

class PPK2Command:
    GET_METADATA = 0x19
    AVERAGE_START = 0x06
    AVERAGE_STOP = 0x07
    RESET = 0x20
    SET_POWER_MODE = 0x11
    SET_USER_GAINS = 0x25
    DEVICE_RUNNING_SET = 0x0C
    TRIGGER_SET = 0x01
    NO_OP = 0x00
    REGULATOR_SET = 0x0d


class PPK2Protocol(Protocol):
    def __init__(self):
        super().__init__()
        self.transport = None
        self.buffer = b""
        self.sample_size = 4
        self.ppk2 = None  # Back reference

    def connection_made(self, transport):
        self.transport = transport
        logging.info("Serial connection established.")

    def data_received(self, data):
        self.buffer += data
        while len(self.buffer) >= self.sample_size:
            sample = int.from_bytes(self.buffer[:4], 'little')
            self.buffer = self.buffer[4:]
            if self.ppk2:
                self.ppk2.handle_sample(sample)

    def connection_lost(self, exc):
        logging.warning("Serial connection lost.")


class PPK2:
    def __init__(self, port, source_voltage=5000, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=1)
        self.reader_thread = None
        self.protocol = None

        self.expected_counter = None
        self.remainder = b""
        self.recording = []
        self.modifiers = {
            "r": [1031.64, 101.65, 10.15, 0.94, 0.043],
            "gs": [1, 1, 1, 1, 1],
            "gi": [1, 1, 1, 1, 1],
            "o": [0, 0, 0, 0, 0],
            "s": [0, 0, 0, 0, 0],
            "i": [0, 0, 0, 0, 0],
            "ug": [1, 1, 1, 1, 1],
        }
        self.adc_mult = 1.8 / 163840
        self.vdd_low = 800
        self.vdd_high = 5000

        self.set_source_voltage(source_voltage)

        if self.reader_thread is not None:
            logging.warning("Reader thread already running.")
            return
        self.reader_thread = ReaderThread(self.serial_port, PPK2Protocol)
        self.reader_thread.start()
        self.protocol = self.reader_thread.connect()[1]
        self.protocol.ppk2 = self

    def _convert_source_voltage(self, mV):
        """Convert input voltage to device command"""
        # minimal possible mV is 800
        if mV < self.vdd_low:
            mV = self.vdd_low

        # maximal possible mV is 5000
        if mV > self.vdd_high:
            mV = self.vdd_high

        offset = 32
        # get difference to baseline (the baseline is 800mV but the initial offset is 32)
        diff_to_baseline = mV - self.vdd_low + offset
        base_b_1 = 3
        base_b_2 = 0  # is actually 32 - compensated with above offset

        # get the number of times we have to increase the first byte of the command
        ratio = int(diff_to_baseline / 256)
        remainder = diff_to_baseline % 256  # get the remainder for byte 2

        set_b_1 = base_b_1 + ratio
        set_b_2 = base_b_2 + remainder

        return set_b_1, set_b_2


    def set_source_voltage(self, mV):
        """Inits device - based on observation only REGULATOR_SET is the command.
        The other two values correspond to the voltage level.

        800mV is the lowest setting - [3,32] - the values then increase linearly
        """
        b_1, b_2 = self._convert_source_voltage(mV)
        self.send_command(PPK2Command.REGULATOR_SET, b_1, b_2)
        self.current_vdd = mV

    def start(self):
        self.recording = []
        self.send_command(PPK2Command.AVERAGE_START)

    def stop(self):
        self.send_command(PPK2Command.AVERAGE_STOP)

    def close(self):
        if self.reader_thread:
            self.reader_thread.close()
            self.reader_thread = None
            self.protocol = None
        if self.reader_thread:
            self.reader_thread.close()

    def __del__(self):
        self.disable_DUT()
        self.close()

    def enable_DUT(self):
        self.send_command(PPK2Command.DEVICE_RUNNING_SET, PPK2Command.TRIGGER_SET)

    def disable_DUT(self):
        self.send_command(PPK2Command.DEVICE_RUNNING_SET, PPK2Command.NO_OP)

    def send_command(self, *args):
        self.serial_port.write(bytes(args))

    def handle_sample(self, adc_value: int):
        range_ = (adc_value >> 14) & 0x7
        counter = (adc_value >> 18) & 0x3F
        adc_result = (adc_value & 0x3FFF) * 4
        bits = (adc_value >> 24) & 0xFF

        res = (adc_result - self.modifiers["o"][range_]) * (self.adc_mult / self.modifiers["r"][range_])
        adc = self.modifiers["ug"][range_] * (
            res * (self.modifiers["gs"][range_] * res + self.modifiers["gi"][range_]) +
            (self.modifiers["s"][range_] * (self.current_vdd / 1000) + self.modifiers["i"][range_])
        )

        self.recording.append({"value": 0.5*adc * 1e6, "bits": bits, "time": time.time()})

    def get_metadata(self):
        self.send_command(PPK2Command.GET_METADATA)
        data = self.serial_port.read(512).decode(errors='ignore')
        print("Metadata:", data)
        return data


class PowerProfilerKitII:
    """
    Class that can be used to record the power consumption of a device using the Power Profiler Kit II from Nordic.
    """

    def __init__(self, port: any = "serial", name: str = "ppk2", mode="source", source_voltage: int = 5000,
                 verbose: int = 0, log_dir: str = None, limit_uA=None, **kwargs):

        self.kit = PPK2(port=port, source_voltage=source_voltage, baudrate=115200)
        self.name = name
        self.recordings = []
        if mode not in ["source"]:
            raise Exception(f'Unsupported mode {mode}')
        self.verbose = verbose
        if log_dir is None:
            log_dir = os.path.join("logs-{}".format(time.strftime("%Y%m%d-%H%M%S")))
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.limit_uA = limit_uA
        self.phase = None
        self.running = False

    def set_source_voltage(self, int):
        """
        Set the source voltage of the Power Profiler Kit II.
        :param int: voltage in mV
        """
        self.kit.set_source_voltage(int)
        if self.verbose:
            print(f"Source voltage set to {int} mV")

    def toggle_power(self, state):
        """
        Toggle the power of the Power Profiler Kit II.
        :param state: "ON" or "OFF"
        """
        if state not in ["ON", "OFF"]:
            raise Exception(f'Unsupported state {state}')
        if state == "ON":
            self.kit.enable_DUT()
        elif state == "OFF":
            self.kit.disable_DUT()

    def start(self, phase: str):
        """
        Start recording the power consumption of the device for a given phase.
        :param phase: name of the phase
        """
        # launch a thread that acquires data
        if self.running:
            raise Exception("Already recording. Stop first")
        else:
            self.phase = phase
            self.recording = {"phase": phase, "start_timestamp": time.time(), "samples": []}
            self.kit.start()

    def stop(self):
        """
        Stop recording the power consumption of the device.
        """
        self.kit.stop()
        self.recording["stop_timestamp"] = time.time()
        self.running = False
        # get samples from kit
        times = [s['time'] - self.kit.recording[0]['time'] for s in self.kit.recording]
        values = [s['value'] for s in self.kit.recording]

        self.recording["samples"] = [{"time": t, "value": v} for t, v in zip(times, values)]
        self.recordings.append(self.recording)

        with open(os.path.join(self.log_dir, f"{self.name}_recording.json"), "w") as f:
            json.dump(self.recordings, f)
