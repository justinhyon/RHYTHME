import time

import numpy as np
import pandas as pd

import serial
import serial.tools.list_ports

def find_arduino(port=None):
    """Get the name of the port that is connected to Arduino."""
    if port is None:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if p.manufacturer is not None and "Arduino" in p.manufacturer:
                port = p.device
    return port


def handshake_arduino(
    arduino, sleep_time=1, print_handshake_message=False, handshake_code=0
):
    """Make sure connection is established by sending
    and receiving bytes."""
    # Close and reopen
    arduino.close()
    arduino.open()

    # Chill out while everything gets set
    time.sleep(sleep_time)

    # Set a long timeout to complete handshake
    timeout = arduino.timeout
    arduino.timeout = 2

    # Read and discard everything that may be in the input buffer
    _ = arduino.read_all()

    # Send request to Arduino
    arduino.write(bytes([handshake_code]))

    # Read in what Arduino sent
    handshake_message = arduino.read_until()

    # Send and receive request again
    arduino.write(bytes([handshake_code]))
    handshake_message = arduino.read_until()

    # Print the handshake message, if desired
    if print_handshake_message:
        print("Handshake message: " + handshake_message.decode())

    # Reset the timeout
    arduino.timeout = timeout

if __name__ == '__main__':
    HANDSHAKE = 0
    OFF = 2
    SIGNAL = 3


    port = find_arduino()
    arduino = serial.Serial(port, baudrate=115200)
    handshake_arduino(arduino, handshake_code=HANDSHAKE, print_handshake_message=True)

    arduino.write(bytes([SIGNAL]))

    while(True):
        time.sleep(2)
        print("send")

        # arduino.close()
        # arduino.open()
        # arduino.timeout = 1
        # _ = arduino.read_all()
        arduino.write(bytes([SIGNAL]))
        # handshake_message = arduino.read_until()

        # Send and receive request again
        # arduino.write(bytes([SIGNAL]))
        # handshake_message = arduino.read_until()
        # Send request to Arduino
        # handshake_arduino(arduino, handshake_code=SIGNAL, print_handshake_message=True)
        # message = arduino.read_until()


