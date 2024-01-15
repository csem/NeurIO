#!/user/bin/env python

"""
Test suite for Power Profiler Kit II
"""

import pytest
import time
from neurio.devices.monitors.power_monitor import PowerProfilerKitII

import unittest
from unittest.mock import patch, MagicMock
import threading

class TestPowerProfilerKitII(unittest.TestCase):

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_init(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_modifiers.return_value = None

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='ampere', acquisition_interval=1, verbose=0)
        mock_ppk2_class.assert_called_once_with('COM3', timeout=1, write_timeout=1, exclusive=True)
        mock_ppk2_instance.get_modifiers.assert_called_once()
        assert ppk2_instance.kit == mock_ppk2_instance
        assert ppk2_instance.port == 'COM3'
        assert ppk2_instance.name == 'ppk2'
        assert ppk2_instance.mode == 'ampere'
        assert ppk2_instance.acquisition_interval == 1
        assert ppk2_instance.verbose == 0

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_start_stop(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_data.return_value = b'test_data'

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='ampere', acquisition_interval=1, verbose=0)


        ppk2_instance.start('test_phase')
        self.assertTrue(ppk2_instance.running)
        self.assertIsInstance(ppk2_instance.thread, threading.Thread)

        ppk2_instance.stop()
        self.assertFalse(ppk2_instance.running)

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_record(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_samples.return_value = 'samples_data', 'digital_data'

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='ampere', acquisition_interval=1, verbose=0)
        ppk2_instance.start('test_phase')
        time.sleep(1.2)
        ppk2_instance.stop()
        # assert get data is called 2 times during the sleep
        assert mock_ppk2_instance.get_data.call_count == 2
        assert mock_ppk2_instance.get_samples.call_count == 2
        assert len(ppk2_instance.recordings) == 1
        assert ppk2_instance.recordings[0]['samples'] == ['samples_data', 'samples_data']
        assert ppk2_instance.recordings[0]['digital'] == ['digital_data', 'digital_data']

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_set_source_voltage(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_modifiers.return_value = None

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='ampere', acquisition_interval=1, verbose=0, source_voltage=5000)
        ppk2_instance.kit.set_source_voltage.assert_called_once_with(5000)

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_use_ampere_meter(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_modifiers.return_value = None

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='ampere', acquisition_interval=1, verbose=0)
        mock_ppk2_instance.use_ampere_meter.assert_called_once()

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_use_source_meter(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_modifiers.return_value = None

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='source', acquisition_interval=1, verbose=0)
        mock_ppk2_instance.use_source_meter.assert_called_once()

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_read_data_empty(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_data.return_value = b''

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='ampere', acquisition_interval=1, verbose=0)
        ppk2_instance.start('test_phase')
        time.sleep(1.2)
        ppk2_instance.stop()
        # assert get data is called 2 times during the sleep
        assert mock_ppk2_instance.get_data.call_count == 2
        assert mock_ppk2_instance.get_samples.call_count == 0

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_read_data_not_empty(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_data.return_value = b'test_data'
        mock_ppk2_instance.get_samples.return_value = 'samples_data', 'digital_data'

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='ampere', acquisition_interval=1, verbose=0)
        ppk2_instance.start('test_phase')
        time.sleep(0.5)
        # second read
        mock_ppk2_instance.get_data.return_value = b'test_data2'
        mock_ppk2_instance.get_samples.return_value = 'samples_data2', 'digital_data2'
        time.sleep(1.0)
        ppk2_instance.stop()

        assert mock_ppk2_instance.get_data.call_count == 2
        assert mock_ppk2_instance.get_samples.call_count == 2
        assert len(ppk2_instance.recordings) == 1
        assert ppk2_instance.recordings[0]['samples'] == ['samples_data', 'samples_data2']
        assert ppk2_instance.recordings[0]['digital'] == ['digital_data', 'digital_data2']

    @patch('neurio.devices.monitors.power_monitor.PPK2_API', autospec=True)
    def test_multiple_phase_one_after_the_other_ok(self, mock_ppk2_class):
        mock_ppk2_class.list_devices.return_value = ['COM3']
        mock_ppk2_instance = mock_ppk2_class.return_value
        mock_ppk2_instance.get_data.return_value = b'test_data'
        mock_ppk2_instance.get_samples.return_value = 'samples_data', 'digital_data'

        ppk2_instance = PowerProfilerKitII(port='COM3', name='ppk2', mode='ampere', acquisition_interval=1, verbose=0)
        ppk2_instance.start('test_phase1')
        time.sleep(1.2)
        ppk2_instance.stop()
        ppk2_instance.start('test_phase2')
        time.sleep(1.2)
        ppk2_instance.stop()

        assert mock_ppk2_instance.get_data.call_count == 4
        assert mock_ppk2_instance.get_samples.call_count == 4
        assert len(ppk2_instance.recordings) == 2
        ppk2_instance.recordings[0]["phase"] = 'test_phase1'
        ppk2_instance.recordings[1]["phase"] = 'test_phase2'
        assert ppk2_instance.recordings[0]['samples'] == ['samples_data', 'samples_data']
        assert ppk2_instance.recordings[0]['digital'] == ['digital_data', 'digital_data']
        assert ppk2_instance.recordings[1]['samples'] == ['samples_data', 'samples_data']
        assert ppk2_instance.recordings[1]['digital'] == ['digital_data', 'digital_data']






if __name__ == '__main__':
    unittest.main()

